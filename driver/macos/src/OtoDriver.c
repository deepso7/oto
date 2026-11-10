// SPDX-License-Identifier: MIT OR Apache-2.0
//
// OtoMicrophone.driver: a Core Audio AudioServerPlugIn that is a loopback.
//
// Structure follows Apple's NullAudio sample (one plug-in object, devices, streams, a
// GetZeroTimeStamp clock), written from scratch and trimmed to what Oto needs.
//
// Objects
//   1  plug-in
//   2  "Oto Microphone"  UID dev.oto.microphone  visible input device, 2 ch
//   3    its input stream
//   4  "Oto Sink"        UID dev.oto.sink        hidden output device, 2 ch
//   5    its output stream
//
// The sink is hidden (kAudioDevicePropertyIsHidden = 1, cannot be default), so it does not
// show in kAudioHardwarePropertyDevices or system pickers. An app finds it with
// kAudioHardwarePropertyTranslateUIDToDevice("dev.oto.sink") and opens it like any device.
//
// Both formats are fixed: 48 kHz, 2 channels, packed native-endian float32.
//
// Loopback
//   Both devices share one clock: a single anchor host time set when the first of the two
//   devices starts IO, so a sample time S means the same instant on both devices and they
//   cannot drift. They also report the same kAudioDevicePropertyClockDomain.
//
//   The sink's WriteMix stores frame S into ring slot S % RING_FRAMES and stamps the slot
//   with S. The mic's ReadInput for frame S returns that slot only if its stamp is still S;
//   any other stamp (sink stopped, never written, overwritten) reads as silence. Stale audio
//   is therefore impossible, and the ring needs no clearing on stop. Stamps are reset when
//   the shared clock restarts from zero, so an old session cannot alias a new one.
//
//   The ring is single-writer (sink IO thread), single-reader (mic IO thread), lock-free.
//
// Loopback delay
//   In the timestamp domain the delay is 0 frames: frame S played to the sink appears as
//   frame S on the mic. In wall-clock terms, the HAL asks for output one IO buffer ahead
//   of now and delivers input one IO buffer behind now, so audio an app renders into the
//   sink reaches a mic client after about
//       sink IO buffer + mic IO buffer
//   Measured with driver/macos/tools/oto-verify on macOS 27 (M-series): 128 + 128 frames ->
//   5.33 ms mean, 512 + 512 frames -> 21.33 ms mean, bit-exact. Latency and safety offsets are
//   reported as 0.

#include <CoreAudio/AudioServerPlugIn.h>
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <string.h>
#include <os/log.h>

// ---------------------------------------------------------------------------------------------
// Constants

enum {
    kObj_PlugIn = kAudioObjectPlugInObject, // 1
    kObj_Mic = 2,
    kObj_MicStream = 3,
    kObj_Sink = 4,
    kObj_SinkStream = 5,
};

#define SAMPLE_RATE 48000.0
#define CHANNELS 2
#define BYTES_PER_FRAME (CHANNELS * sizeof(Float32))
#define ZTS_PERIOD 16384u   // frames between zero time stamps
#define RING_FRAMES 32768u  // power of two, > any IO buffer the HAL allows
#define CLOCK_DOMAIN 0x4f746f31u // 'Oto1'

#define MANUFACTURER "Oto"
#define MIC_NAME "Oto Microphone"
#define MIC_UID "dev.oto.microphone"
#define SINK_NAME "Oto Sink"
#define SINK_UID "dev.oto.sink"
#define MODEL_UID "dev.oto.loopback"

// ---------------------------------------------------------------------------------------------
// State

static AudioServerPlugInHostRef gHost = NULL;
static _Atomic UInt32 gRefCount = 0;

static pthread_mutex_t gStateMutex = PTHREAD_MUTEX_INITIALIZER;
static UInt32 gMicIOCount = 0;  // StartIO minus StopIO, per device
static UInt32 gSinkIOCount = 0;
static _Atomic UInt64 gAnchorHostTime = 0; // shared clock anchor, valid while any IO runs
static Float64 gHostTicksPerFrame = 0;

static Float32 gRing[RING_FRAMES * CHANNELS];
static _Atomic UInt64 gStamp[RING_FRAMES]; // sample time held by each slot, EMPTY = none
#define EMPTY UINT64_MAX

static os_log_t gLog;

static void ring_reset(void) {
    for (UInt32 i = 0; i < RING_FRAMES; i++) atomic_store_explicit(&gStamp[i], EMPTY, memory_order_relaxed);
    memset(gRing, 0, sizeof gRing);
    atomic_thread_fence(memory_order_release);
}

static void ring_write(Float64 sampleTime, const Float32* src, UInt32 frames) {
    if (sampleTime < 0) return;
    UInt64 s = (UInt64)(sampleTime + 0.5);
    for (UInt32 f = 0; f < frames; f++, s++) {
        UInt32 slot = (UInt32)(s & (RING_FRAMES - 1));
        atomic_store_explicit(&gStamp[slot], EMPTY, memory_order_relaxed);
        atomic_thread_fence(memory_order_release);
        for (UInt32 c = 0; c < CHANNELS; c++) gRing[slot * CHANNELS + c] = src[f * CHANNELS + c];
        atomic_store_explicit(&gStamp[slot], s, memory_order_release);
    }
}

static void ring_read(Float64 sampleTime, Float32* dst, UInt32 frames) {
    if (sampleTime < 0) {
        memset(dst, 0, frames * BYTES_PER_FRAME);
        return;
    }
    UInt64 s = (UInt64)(sampleTime + 0.5);
    for (UInt32 f = 0; f < frames; f++, s++) {
        UInt32 slot = (UInt32)(s & (RING_FRAMES - 1));
        Float32* out = dst + f * CHANNELS;
        if (atomic_load_explicit(&gStamp[slot], memory_order_acquire) == s) {
            for (UInt32 c = 0; c < CHANNELS; c++) out[c] = gRing[slot * CHANNELS + c];
            atomic_thread_fence(memory_order_acquire);
            if (atomic_load_explicit(&gStamp[slot], memory_order_relaxed) == s) continue;
        }
        for (UInt32 c = 0; c < CHANNELS; c++) out[c] = 0.0f;
    }
}

// ---------------------------------------------------------------------------------------------
// Helpers

static bool is_device(AudioObjectID o) { return o == kObj_Mic || o == kObj_Sink; }
static bool is_stream(AudioObjectID o) { return o == kObj_MicStream || o == kObj_SinkStream; }
static AudioObjectID stream_of(AudioObjectID dev) { return dev == kObj_Mic ? kObj_MicStream : kObj_SinkStream; }
static AudioObjectID device_of(AudioObjectID st) { return st == kObj_MicStream ? kObj_Mic : kObj_Sink; }
// Scope the device's stream lives in.
static AudioObjectPropertyScope dev_scope(AudioObjectID dev) {
    return dev == kObj_Mic ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput;
}
static bool scope_matches(AudioObjectID dev, AudioObjectPropertyScope s) {
    return s == kAudioObjectPropertyScopeGlobal || s == dev_scope(dev);
}

static AudioStreamBasicDescription stream_format(void) {
    AudioStreamBasicDescription f = {0};
    f.mSampleRate = SAMPLE_RATE;
    f.mFormatID = kAudioFormatLinearPCM;
    f.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagsNativeEndian | kAudioFormatFlagIsPacked;
    f.mBytesPerPacket = BYTES_PER_FRAME;
    f.mFramesPerPacket = 1;
    f.mBytesPerFrame = BYTES_PER_FRAME;
    f.mChannelsPerFrame = CHANNELS;
    f.mBitsPerChannel = 32;
    return f;
}

// Writing helpers: each copies at most inSize bytes and reports the full size.
#define PUT(type, value)                                                     \
    do {                                                                     \
        if (inSize < sizeof(type)) return kAudioHardwareBadPropertySizeError; \
        *(type*)outData = (value);                                           \
        *outSize = sizeof(type);                                             \
        return noErr;                                                        \
    } while (0)

static OSStatus put_ids(const AudioObjectID* ids, UInt32 n, UInt32 inSize, UInt32* outSize, void* outData) {
    UInt32 fit = inSize / sizeof(AudioObjectID);
    if (fit > n) fit = n;
    memcpy(outData, ids, fit * sizeof(AudioObjectID));
    *outSize = fit * sizeof(AudioObjectID);
    return noErr;
}

// ---------------------------------------------------------------------------------------------
// Property core. All of HasProperty / GetPropertyDataSize / GetPropertyData go through get_prop.
// For size and existence queries it is called with a scratch buffer; every value is either
// plain data or an immortal CFSTR constant, so nothing leaks.

static OSStatus get_prop(AudioObjectID obj, const AudioObjectPropertyAddress* a, UInt32 qSize,
                         const void* qData, UInt32 inSize, UInt32* outSize, void* outData, bool probe) {
    AudioObjectPropertySelector sel = a->mSelector;

    // ---- plug-in
    if (obj == kObj_PlugIn) {
        switch (sel) {
        case kAudioObjectPropertyBaseClass: PUT(AudioClassID, kAudioObjectClassID);
        case kAudioObjectPropertyClass: PUT(AudioClassID, kAudioPlugInClassID);
        case kAudioObjectPropertyOwner: PUT(AudioObjectID, kAudioObjectUnknown);
        case kAudioObjectPropertyManufacturer: PUT(CFStringRef, CFSTR(MANUFACTURER));
        case kAudioObjectPropertyOwnedObjects:
        case kAudioPlugInPropertyDeviceList: {
            static const AudioObjectID devs[] = {kObj_Mic, kObj_Sink};
            return put_ids(devs, 2, inSize, outSize, outData);
        }
        case kAudioPlugInPropertyBoxList:
        case kAudioPlugInPropertyClockDeviceList: *outSize = 0; return noErr;
        case kAudioPlugInPropertyTranslateUIDToDevice: {
            AudioObjectID r = kAudioObjectUnknown;
            if (!probe) {
                if (qSize < sizeof(CFStringRef) || qData == NULL) return kAudioHardwareBadPropertySizeError;
                CFStringRef uid = *(const CFStringRef*)qData;
                if (uid && CFStringCompare(uid, CFSTR(MIC_UID), 0) == kCFCompareEqualTo) r = kObj_Mic;
                else if (uid && CFStringCompare(uid, CFSTR(SINK_UID), 0) == kCFCompareEqualTo) r = kObj_Sink;
            }
            PUT(AudioObjectID, r);
        }
        case kAudioPlugInPropertyResourceBundle: PUT(CFStringRef, CFSTR(""));
        }
        return kAudioHardwareUnknownPropertyError;
    }

    // ---- devices
    if (is_device(obj)) {
        bool mic = obj == kObj_Mic;
        switch (sel) {
        case kAudioObjectPropertyBaseClass: PUT(AudioClassID, kAudioObjectClassID);
        case kAudioObjectPropertyClass: PUT(AudioClassID, kAudioDeviceClassID);
        case kAudioObjectPropertyOwner: PUT(AudioObjectID, kObj_PlugIn);
        case kAudioObjectPropertyName: PUT(CFStringRef, mic ? CFSTR(MIC_NAME) : CFSTR(SINK_NAME));
        case kAudioObjectPropertyManufacturer: PUT(CFStringRef, CFSTR(MANUFACTURER));
        case kAudioDevicePropertyDeviceUID: PUT(CFStringRef, mic ? CFSTR(MIC_UID) : CFSTR(SINK_UID));
        case kAudioDevicePropertyModelUID: PUT(CFStringRef, CFSTR(MODEL_UID));
        case kAudioDevicePropertyTransportType: PUT(UInt32, kAudioDeviceTransportTypeVirtual);
        case kAudioDevicePropertyRelatedDevices: {
            AudioObjectID self = obj;
            return put_ids(&self, 1, inSize, outSize, outData);
        }
        case kAudioDevicePropertyClockDomain: PUT(UInt32, CLOCK_DOMAIN);
        case kAudioDevicePropertyDeviceIsAlive: PUT(UInt32, 1);
        case kAudioDevicePropertyDeviceIsRunning: {
            pthread_mutex_lock(&gStateMutex);
            UInt32 r = (mic ? gMicIOCount : gSinkIOCount) > 0;
            pthread_mutex_unlock(&gStateMutex);
            PUT(UInt32, r);
        }
        case kAudioDevicePropertyDeviceCanBeDefaultDevice:
        case kAudioDevicePropertyDeviceCanBeDefaultSystemDevice: PUT(UInt32, mic ? 1 : 0);
        case kAudioDevicePropertyIsHidden: PUT(UInt32, mic ? 0 : 1);
        case kAudioDevicePropertyClockIsStable: PUT(UInt32, 1);
        case kAudioDevicePropertyLatency:
        case kAudioDevicePropertySafetyOffset: PUT(UInt32, 0);
        case kAudioDevicePropertyZeroTimeStampPeriod: PUT(UInt32, ZTS_PERIOD);
        case kAudioDevicePropertyNominalSampleRate: PUT(Float64, SAMPLE_RATE);
        case kAudioDevicePropertyAvailableNominalSampleRates: {
            AudioValueRange r = {SAMPLE_RATE, SAMPLE_RATE};
            PUT(AudioValueRange, r);
        }
        case kAudioObjectPropertyOwnedObjects: {
            AudioObjectID s = stream_of(obj);
            return put_ids(&s, scope_matches(obj, a->mScope) ? 1 : 0, inSize, outSize, outData);
        }
        case kAudioDevicePropertyStreams: {
            AudioObjectID s = stream_of(obj);
            return put_ids(&s, scope_matches(obj, a->mScope) ? 1 : 0, inSize, outSize, outData);
        }
        case kAudioObjectPropertyControlList: *outSize = 0; return noErr;
        case kAudioDevicePropertyPreferredChannelsForStereo: {
            if (inSize < 2 * sizeof(UInt32)) return kAudioHardwareBadPropertySizeError;
            ((UInt32*)outData)[0] = 1;
            ((UInt32*)outData)[1] = 2;
            *outSize = 2 * sizeof(UInt32);
            return noErr;
        }
        case kAudioDevicePropertyPreferredChannelLayout: {
            UInt32 size = (UInt32)(offsetof(AudioChannelLayout, mChannelDescriptions) +
                                   CHANNELS * sizeof(AudioChannelDescription));
            if (inSize < size) return kAudioHardwareBadPropertySizeError;
            AudioChannelLayout* l = outData;
            memset(l, 0, size);
            l->mChannelLayoutTag = kAudioChannelLayoutTag_UseChannelDescriptions;
            l->mNumberChannelDescriptions = CHANNELS;
            l->mChannelDescriptions[0].mChannelLabel = kAudioChannelLabel_Left;
            l->mChannelDescriptions[1].mChannelLabel = kAudioChannelLabel_Right;
            *outSize = size;
            return noErr;
        }
        case kAudioDevicePropertyIcon: return kAudioHardwareUnknownPropertyError;
        }
        return kAudioHardwareUnknownPropertyError;
    }

    // ---- streams
    if (is_stream(obj)) {
        bool micStream = obj == kObj_MicStream;
        switch (sel) {
        case kAudioObjectPropertyBaseClass: PUT(AudioClassID, kAudioObjectClassID);
        case kAudioObjectPropertyClass: PUT(AudioClassID, kAudioStreamClassID);
        case kAudioObjectPropertyOwner: PUT(AudioObjectID, device_of(obj));
        case kAudioObjectPropertyOwnedObjects: *outSize = 0; return noErr;
        // Always active and not settable: the IO path has no notion of a disabled stream.
        case kAudioStreamPropertyIsActive: PUT(UInt32, 1);
        case kAudioStreamPropertyDirection: PUT(UInt32, micStream ? 1 : 0); // 1 = input
        case kAudioStreamPropertyTerminalType:
            PUT(UInt32, micStream ? kAudioStreamTerminalTypeMicrophone : kAudioStreamTerminalTypeSpeaker);
        case kAudioStreamPropertyStartingChannel: PUT(UInt32, 1);
        case kAudioStreamPropertyLatency: PUT(UInt32, 0);
        case kAudioStreamPropertyVirtualFormat:
        case kAudioStreamPropertyPhysicalFormat: PUT(AudioStreamBasicDescription, stream_format());
        case kAudioStreamPropertyAvailableVirtualFormats:
        case kAudioStreamPropertyAvailablePhysicalFormats: {
            AudioStreamRangedDescription d = {stream_format(), {SAMPLE_RATE, SAMPLE_RATE}};
            PUT(AudioStreamRangedDescription, d);
        }
        }
        return kAudioHardwareUnknownPropertyError;
    }

    return kAudioHardwareBadObjectError;
}

// ---------------------------------------------------------------------------------------------
// Driver interface

static HRESULT Oto_QueryInterface(void* inDriver, REFIID inUUID, LPVOID* outInterface);
static ULONG Oto_AddRef(void* inDriver);
static ULONG Oto_Release(void* inDriver);

static OSStatus Oto_Initialize(AudioServerPlugInDriverRef d, AudioServerPlugInHostRef host) {
    (void)d;
    gHost = host;
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    gHostTicksPerFrame = (1.0e9 / SAMPLE_RATE) * (Float64)tb.denom / (Float64)tb.numer;
    ring_reset();
    os_log(gLog, "Oto driver initialized (%.3f host ticks/frame)", gHostTicksPerFrame);
    return noErr;
}

static OSStatus Oto_CreateDevice(AudioServerPlugInDriverRef d, CFDictionaryRef desc,
                                 const AudioServerPlugInClientInfo* ci, AudioObjectID* outID) {
    (void)d; (void)desc; (void)ci; (void)outID;
    return kAudioHardwareUnsupportedOperationError;
}
static OSStatus Oto_DestroyDevice(AudioServerPlugInDriverRef d, AudioObjectID dev) {
    (void)d; (void)dev;
    return kAudioHardwareUnsupportedOperationError;
}
static OSStatus Oto_AddDeviceClient(AudioServerPlugInDriverRef d, AudioObjectID dev,
                                    const AudioServerPlugInClientInfo* ci) {
    (void)d; (void)ci;
    return is_device(dev) ? noErr : kAudioHardwareBadObjectError;
}
static OSStatus Oto_RemoveDeviceClient(AudioServerPlugInDriverRef d, AudioObjectID dev,
                                       const AudioServerPlugInClientInfo* ci) {
    (void)d; (void)ci;
    return is_device(dev) ? noErr : kAudioHardwareBadObjectError;
}
static OSStatus Oto_PerformConfigChange(AudioServerPlugInDriverRef d, AudioObjectID dev, UInt64 action, void* info) {
    (void)d; (void)dev; (void)action; (void)info;
    return noErr;
}
static OSStatus Oto_AbortConfigChange(AudioServerPlugInDriverRef d, AudioObjectID dev, UInt64 action, void* info) {
    (void)d; (void)dev; (void)action; (void)info;
    return noErr;
}

static Boolean Oto_HasProperty(AudioServerPlugInDriverRef d, AudioObjectID obj, pid_t pid,
                               const AudioObjectPropertyAddress* a) {
    (void)d; (void)pid;
    if (a == NULL) return false;
    UInt64 scratch[128];
    UInt32 size = 0;
    return get_prop(obj, a, 0, NULL, sizeof scratch, &size, scratch, true) == noErr;
}

static OSStatus Oto_IsPropertySettable(AudioServerPlugInDriverRef d, AudioObjectID obj, pid_t pid,
                                       const AudioObjectPropertyAddress* a, Boolean* out) {
    if (a == NULL || out == NULL) return kAudioHardwareIllegalOperationError;
    if (!Oto_HasProperty(d, obj, pid, a)) return kAudioHardwareUnknownPropertyError;
    switch (a->mSelector) {
    // The sample rate is fixed, so kAudioDevicePropertyNominalSampleRate is read-only. The
    // stream formats stay settable because clients routinely "set" the one format we have.
    case kAudioStreamPropertyVirtualFormat:
    case kAudioStreamPropertyPhysicalFormat: *out = is_stream(obj); break;
    default: *out = false;
    }
    return noErr;
}

static OSStatus Oto_GetPropertyDataSize(AudioServerPlugInDriverRef d, AudioObjectID obj, pid_t pid,
                                        const AudioObjectPropertyAddress* a, UInt32 qSize,
                                        const void* qData, UInt32* outSize) {
    (void)d; (void)pid;
    if (a == NULL || outSize == NULL) return kAudioHardwareIllegalOperationError;
    UInt64 scratch[128];
    return get_prop(obj, a, qSize, qData, sizeof scratch, outSize, scratch, qData == NULL);
}

static OSStatus Oto_GetPropertyData(AudioServerPlugInDriverRef d, AudioObjectID obj, pid_t pid,
                                    const AudioObjectPropertyAddress* a, UInt32 qSize, const void* qData,
                                    UInt32 inSize, UInt32* outSize, void* outData) {
    (void)d; (void)pid;
    if (a == NULL || outSize == NULL || outData == NULL) return kAudioHardwareIllegalOperationError;
    return get_prop(obj, a, qSize, qData, inSize, outSize, outData, false);
}

static OSStatus Oto_SetPropertyData(AudioServerPlugInDriverRef d, AudioObjectID obj, pid_t pid,
                                    const AudioObjectPropertyAddress* a, UInt32 qSize, const void* qData,
                                    UInt32 inSize, const void* inData) {
    (void)d; (void)pid; (void)qSize; (void)qData;
    if (a == NULL || inData == NULL) return kAudioHardwareIllegalOperationError;
    switch (a->mSelector) {
    case kAudioDevicePropertyNominalSampleRate:
        if (!is_device(obj)) break;
        if (inSize < sizeof(Float64)) return kAudioHardwareBadPropertySizeError;
        // Reported as not settable; a set to the rate we already run at is still a harmless no-op.
        return *(const Float64*)inData == SAMPLE_RATE ? noErr : kAudioDeviceUnsupportedFormatError;
    case kAudioStreamPropertyVirtualFormat:
    case kAudioStreamPropertyPhysicalFormat: {
        if (!is_stream(obj)) break;
        if (inSize < sizeof(AudioStreamBasicDescription)) return kAudioHardwareBadPropertySizeError;
        const AudioStreamBasicDescription* f = inData;
        AudioStreamBasicDescription ours = stream_format();
        bool ok = f->mFormatID == ours.mFormatID && f->mSampleRate == ours.mSampleRate &&
                  f->mChannelsPerFrame == ours.mChannelsPerFrame && f->mBitsPerChannel == ours.mBitsPerChannel &&
                  (f->mFormatFlags & kAudioFormatFlagIsFloat);
        return ok ? noErr : kAudioDeviceUnsupportedFormatError;
    }
    }
    return Oto_HasProperty(d, obj, pid, a) ? kAudioHardwareUnsupportedOperationError
                                           : kAudioHardwareUnknownPropertyError;
}

static void notify_running(AudioObjectID dev) {
    if (!gHost) return;
    AudioObjectPropertyAddress ad = {kAudioDevicePropertyDeviceIsRunning, kAudioObjectPropertyScopeGlobal,
                                     kAudioObjectPropertyElementMain};
    gHost->PropertiesChanged(gHost, dev, 1, &ad);
}

static OSStatus Oto_StartIO(AudioServerPlugInDriverRef d, AudioObjectID dev, UInt32 client) {
    (void)d; (void)client;
    if (!is_device(dev)) return kAudioHardwareBadObjectError;
    pthread_mutex_lock(&gStateMutex);
    UInt32* count = dev == kObj_Mic ? &gMicIOCount : &gSinkIOCount;
    bool first = *count == 0;
    if (gMicIOCount == 0 && gSinkIOCount == 0) {
        // The shared clock restarts at sample time 0; forget every old stamp first.
        ring_reset();
        atomic_store(&gAnchorHostTime, mach_absolute_time());
    }
    (*count)++;
    pthread_mutex_unlock(&gStateMutex);
    if (first) notify_running(dev);
    return noErr;
}

static OSStatus Oto_StopIO(AudioServerPlugInDriverRef d, AudioObjectID dev, UInt32 client) {
    (void)d; (void)client;
    if (!is_device(dev)) return kAudioHardwareBadObjectError;
    pthread_mutex_lock(&gStateMutex);
    UInt32* count = dev == kObj_Mic ? &gMicIOCount : &gSinkIOCount;
    bool last = false;
    if (*count > 0) {
        (*count)--;
        last = *count == 0;
    }
    pthread_mutex_unlock(&gStateMutex);
    if (last) notify_running(dev);
    return noErr;
}

// Stateless and shared by both devices: the latest period boundary since the common anchor.
static OSStatus Oto_GetZeroTimeStamp(AudioServerPlugInDriverRef d, AudioObjectID dev, UInt32 client,
                                     Float64* outSampleTime, UInt64* outHostTime, UInt64* outSeed) {
    (void)d; (void)client;
    if (!is_device(dev)) return kAudioHardwareBadObjectError;
    UInt64 anchor = atomic_load(&gAnchorHostTime);
    UInt64 now = mach_absolute_time();
    Float64 ticksPerPeriod = gHostTicksPerFrame * ZTS_PERIOD;
    UInt64 periods = now > anchor ? (UInt64)((Float64)(now - anchor) / ticksPerPeriod) : 0;
    *outSampleTime = (Float64)periods * ZTS_PERIOD;
    *outHostTime = anchor + (UInt64)((Float64)periods * ticksPerPeriod);
    *outSeed = 1;
    return noErr;
}

static OSStatus Oto_WillDoIOOperation(AudioServerPlugInDriverRef d, AudioObjectID dev, UInt32 client,
                                      UInt32 op, Boolean* outWill, Boolean* outInPlace) {
    (void)d; (void)client;
    bool will = (dev == kObj_Mic && op == kAudioServerPlugInIOOperationReadInput) ||
                (dev == kObj_Sink && op == kAudioServerPlugInIOOperationWriteMix);
    if (outWill) *outWill = will;
    if (outInPlace) *outInPlace = true;
    return noErr;
}

static OSStatus Oto_BeginIOOperation(AudioServerPlugInDriverRef d, AudioObjectID dev, UInt32 client, UInt32 op,
                                     UInt32 frames, const AudioServerPlugInIOCycleInfo* info) {
    (void)d; (void)dev; (void)client; (void)op; (void)frames; (void)info;
    return noErr;
}

static OSStatus Oto_DoIOOperation(AudioServerPlugInDriverRef d, AudioObjectID dev, AudioObjectID stream,
                                  UInt32 client, UInt32 op, UInt32 frames,
                                  const AudioServerPlugInIOCycleInfo* info, void* main, void* secondary) {
    (void)d; (void)stream; (void)client; (void)secondary;
    if (main == NULL || info == NULL) return noErr;
    if (dev == kObj_Mic && op == kAudioServerPlugInIOOperationReadInput)
        ring_read(info->mInputTime.mSampleTime, main, frames);
    else if (dev == kObj_Sink && op == kAudioServerPlugInIOOperationWriteMix)
        ring_write(info->mOutputTime.mSampleTime, main, frames);
    return noErr;
}

static OSStatus Oto_EndIOOperation(AudioServerPlugInDriverRef d, AudioObjectID dev, UInt32 client, UInt32 op,
                                   UInt32 frames, const AudioServerPlugInIOCycleInfo* info) {
    (void)d; (void)dev; (void)client; (void)op; (void)frames; (void)info;
    return noErr;
}

static AudioServerPlugInDriverInterface gInterface = {
    NULL,
    Oto_QueryInterface,
    Oto_AddRef,
    Oto_Release,
    Oto_Initialize,
    Oto_CreateDevice,
    Oto_DestroyDevice,
    Oto_AddDeviceClient,
    Oto_RemoveDeviceClient,
    Oto_PerformConfigChange,
    Oto_AbortConfigChange,
    Oto_HasProperty,
    Oto_IsPropertySettable,
    Oto_GetPropertyDataSize,
    Oto_GetPropertyData,
    Oto_SetPropertyData,
    Oto_StartIO,
    Oto_StopIO,
    Oto_GetZeroTimeStamp,
    Oto_WillDoIOOperation,
    Oto_BeginIOOperation,
    Oto_DoIOOperation,
    Oto_EndIOOperation,
};
static AudioServerPlugInDriverInterface* gInterfacePtr = &gInterface;
static AudioServerPlugInDriverRef gDriverRef = &gInterfacePtr;

static HRESULT Oto_QueryInterface(void* inDriver, REFIID inUUID, LPVOID* outInterface) {
    if (inDriver != gDriverRef || outInterface == NULL) return kAudioHardwareBadObjectError;
    CFUUIDRef req = CFUUIDCreateFromUUIDBytes(NULL, inUUID);
    if (req == NULL) return kAudioHardwareIllegalOperationError;
    bool ok = CFEqual(req, IUnknownUUID) || CFEqual(req, kAudioServerPlugInDriverInterfaceUUID);
    CFRelease(req);
    if (!ok) return E_NOINTERFACE;
    atomic_fetch_add(&gRefCount, 1);
    *outInterface = gDriverRef;
    return S_OK;
}

static ULONG Oto_AddRef(void* inDriver) {
    if (inDriver != gDriverRef) return 0;
    return atomic_fetch_add(&gRefCount, 1) + 1;
}

static ULONG Oto_Release(void* inDriver) {
    if (inDriver != gDriverRef) return 0;
    UInt32 cur = atomic_load(&gRefCount);
    while (cur > 0 && !atomic_compare_exchange_weak(&gRefCount, &cur, cur - 1)) {}
    return cur > 0 ? cur - 1 : 0;
}

// Factory named in Info.plist (CFPlugInFactories).
__attribute__((visibility("default"))) void* Oto_Create(CFAllocatorRef alloc, CFUUIDRef requestedType) {
    (void)alloc;
    if (gLog == NULL) gLog = os_log_create("dev.oto.driver", "driver");
    if (!CFEqual(requestedType, kAudioServerPlugInTypeUUID)) return NULL;
    os_log(gLog, "Oto_Create");
    return gDriverRef;
}
