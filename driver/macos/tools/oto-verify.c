// SPDX-License-Identifier: MIT OR Apache-2.0
//
// oto-verify: checks an installed OtoMicrophone.driver.
//
//   1. "Oto Microphone" is in the system device list; "Oto Sink" is not, but resolves by UID
//      and reports IsHidden = 1.
//   2. Plays 2 s of seeded white noise into the sink (opened by UID) while recording the mic,
//      both through plain HAL IOProcs, and reports:
//        - normalized cross-correlation peak and its lag in the shared sample-time domain
//        - loopback delay: wall time from the sink IOProc rendering a frame to the mic IOProc
//          receiving it, which is what an app like the Oto engine sees
//        - how quickly the mic goes silent after the sink is stopped, and that it stays silent
//   Exits non-zero if any check fails.
//
// Usage: oto-verify [-b buffer_frames]   (default 512; applies to both devices)

#include <CoreAudio/CoreAudio.h>
#include <CoreFoundation/CoreFoundation.h>
#include <mach/mach_time.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MIC_UID "dev.oto.microphone"
#define SINK_UID "dev.oto.sink"
#define SR 48000.0
#define TOTAL_FRAMES (48000 * 8)
#define PLAY_FRAMES (48000 * 2)

static float gSent[TOTAL_FRAMES];       // what the sink rendered, by frame (relative to gBase)
static UInt64 gRenderHost[TOTAL_FRAMES]; // host time the sink IOProc rendered it
static float gRecv[TOTAL_FRAMES];       // what the mic delivered (left channel)
static float gRecvR[TOTAL_FRAMES];      // right channel
static UInt64 gDeliverHost[TOTAL_FRAMES];
static bool gHaveRecv[TOTAL_FRAMES];

static _Atomic double gBase = -1;     // first mic input sample time; index 0
static _Atomic int gPlayStarted = 0;  // sink frames rendered so far is tracked via gSinkStart
static double gSinkStart = -1;        // relative frame index where playback began
static _Atomic long gSinkRendered = 0;
static UInt32 gNoise = 12345;
static double gTicksToMs;

static float noise(void) {
    gNoise = gNoise * 1664525u + 1013904223u;
    return ((float)(gNoise >> 8) / 16777216.0f - 0.5f) * 0.6f;
}

static OSStatus mic_proc(AudioObjectID dev, const AudioTimeStamp* now, const AudioBufferList* in,
                         const AudioTimeStamp* inTime, AudioBufferList* out, const AudioTimeStamp* outTime,
                         void* ctx) {
    (void)dev; (void)out; (void)outTime; (void)ctx;
    if (!in || in->mNumberBuffers < 1) return noErr;
    const float* x = in->mBuffers[0].mData;
    UInt32 ch = in->mBuffers[0].mNumberChannels;
    UInt32 frames = in->mBuffers[0].mDataByteSize / (ch * sizeof(float));
    double base = atomic_load(&gBase);
    if (base < 0) {
        base = inTime->mSampleTime;
        atomic_store(&gBase, base);
    }
    for (UInt32 i = 0; i < frames; i++) {
        long idx = (long)(inTime->mSampleTime - base) + i;
        if (idx < 0 || idx >= TOTAL_FRAMES) continue;
        gRecv[idx] = x[i * ch];
        gRecvR[idx] = ch > 1 ? x[i * ch + 1] : x[i * ch];
        gDeliverHost[idx] = now->mHostTime;
        gHaveRecv[idx] = true;
    }
    return noErr;
}

static OSStatus sink_proc(AudioObjectID dev, const AudioTimeStamp* now, const AudioBufferList* in,
                          const AudioTimeStamp* inTime, AudioBufferList* out, const AudioTimeStamp* outTime,
                          void* ctx) {
    (void)dev; (void)in; (void)inTime; (void)ctx;
    if (!out || out->mNumberBuffers < 1) return noErr;
    float* y = out->mBuffers[0].mData;
    UInt32 ch = out->mBuffers[0].mNumberChannels;
    UInt32 frames = out->mBuffers[0].mDataByteSize / (ch * sizeof(float));
    double base = atomic_load(&gBase);
    for (UInt32 i = 0; i < frames; i++) {
        long idx = base < 0 ? -1 : (long)(outTime->mSampleTime - base) + i;
        float v = 0;
        if (idx >= 0 && idx < TOTAL_FRAMES && atomic_load(&gSinkRendered) < PLAY_FRAMES) {
            if (gSinkStart < 0) gSinkStart = idx;
            v = noise();
            gSent[idx] = v;
            gRenderHost[idx] = now->mHostTime;
            atomic_fetch_add(&gSinkRendered, 1);
        }
        for (UInt32 c = 0; c < ch; c++) y[i * ch + c] = v;
    }
    atomic_store(&gPlayStarted, 1);
    return noErr;
}

static AudioObjectID uid_to_device(const char* uid) {
    CFStringRef s = CFStringCreateWithCString(NULL, uid, kCFStringEncodingUTF8);
    AudioObjectID id = kAudioObjectUnknown;
    UInt32 size = sizeof id;
    AudioObjectPropertyAddress a = {kAudioHardwarePropertyTranslateUIDToDevice, kAudioObjectPropertyScopeGlobal,
                                    kAudioObjectPropertyElementMain};
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, sizeof s, &s, &size, &id);
    CFRelease(s);
    return id;
}

static UInt32 get_u32(AudioObjectID o, AudioObjectPropertySelector sel, AudioObjectPropertyScope sc) {
    UInt32 v = 0, size = sizeof v;
    AudioObjectPropertyAddress a = {sel, sc, kAudioObjectPropertyElementMain};
    if (AudioObjectGetPropertyData(o, &a, 0, NULL, &size, &v) != noErr) return 0xffffffff;
    return v;
}

static void get_str(AudioObjectID o, AudioObjectPropertySelector sel, char* buf, size_t n) {
    CFStringRef s = NULL;
    UInt32 size = sizeof s;
    AudioObjectPropertyAddress a = {sel, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    buf[0] = 0;
    if (AudioObjectGetPropertyData(o, &a, 0, NULL, &size, &s) == noErr && s) {
        CFStringGetCString(s, buf, (CFIndex)n, kCFStringEncodingUTF8);
        CFRelease(s);
    }
}

static int fails = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) fails++;
}

int main(int argc, char** argv) {
    UInt32 bufFrames = 512;
    int opt;
    while ((opt = getopt(argc, argv, "b:")) != -1) {
        if (opt == 'b') bufFrames = (UInt32)atoi(optarg);
        else {
            fprintf(stderr, "usage: %s [-b buffer_frames]\n", argv[0]);
            return 2;
        }
    }
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    gTicksToMs = (double)tb.numer / tb.denom / 1e6;

    // ---- 1. visibility
    printf("Devices (kAudioHardwarePropertyDevices):\n");
    AudioObjectPropertyAddress a = {kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal,
                                    kAudioObjectPropertyElementMain};
    UInt32 size = 0;
    AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &a, 0, NULL, &size);
    UInt32 n = size / sizeof(AudioObjectID);
    AudioObjectID* ids = calloc(n ? n : 1, sizeof *ids);
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, 0, NULL, &size, ids);
    bool micListed = false, sinkListed = false;
    for (UInt32 i = 0; i < n; i++) {
        char name[256], uid[256];
        get_str(ids[i], kAudioObjectPropertyName, name, sizeof name);
        get_str(ids[i], kAudioDevicePropertyDeviceUID, uid, sizeof uid);
        printf("  %3u  %-40s %s\n", ids[i], name, uid);
        if (!strcmp(uid, MIC_UID)) micListed = true;
        if (!strcmp(uid, SINK_UID)) sinkListed = true;
    }
    free(ids);

    AudioObjectID mic = uid_to_device(MIC_UID), sink = uid_to_device(SINK_UID);
    char micName[256] = "", sinkName[256] = "", manu[256] = "";
    if (mic) get_str(mic, kAudioObjectPropertyName, micName, sizeof micName);
    if (mic) get_str(mic, kAudioObjectPropertyManufacturer, manu, sizeof manu);
    if (sink) get_str(sink, kAudioObjectPropertyName, sinkName, sizeof sinkName);
    printf("\nVisibility:\n");
    check(mic != kAudioObjectUnknown && micListed, "\"Oto Microphone\" (dev.oto.microphone) is listed");
    printf("         id=%u name=\"%s\" manufacturer=\"%s\" hidden=%u\n", mic, micName, manu,
           get_u32(mic, kAudioDevicePropertyIsHidden, kAudioObjectPropertyScopeGlobal));
    check(!sinkListed, "\"Oto Sink\" is not in the device list");
    check(sink != kAudioObjectUnknown, "\"Oto Sink\" resolves by UID dev.oto.sink");
    check(get_u32(sink, kAudioDevicePropertyIsHidden, kAudioObjectPropertyScopeGlobal) == 1, "Oto Sink IsHidden = 1");
    printf("         id=%u name=\"%s\"\n", sink, sinkName);
    if (!mic || !sink) {
        printf("\nRESULT: FAIL (devices missing)\n");
        return 1;
    }

    // ---- 2. loopback
    AudioObjectPropertyAddress bs = {kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal,
                                     kAudioObjectPropertyElementMain};
    AudioObjectSetPropertyData(mic, &bs, 0, NULL, sizeof bufFrames, &bufFrames);
    AudioObjectSetPropertyData(sink, &bs, 0, NULL, sizeof bufFrames, &bufFrames);
    UInt32 micBuf = get_u32(mic, kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal);
    UInt32 sinkBuf = get_u32(sink, kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal);
    printf("\nLoopback (white noise, %d frames, IO buffers mic=%u sink=%u):\n", PLAY_FRAMES, micBuf, sinkBuf);

    AudioDeviceIOProcID micProc = NULL, sinkProc = NULL;
    if (AudioDeviceCreateIOProcID(mic, mic_proc, NULL, &micProc) ||
        AudioDeviceCreateIOProcID(sink, sink_proc, NULL, &sinkProc)) {
        printf("  cannot create IOProcs\n");
        return 1;
    }
    AudioDeviceStart(mic, micProc);
    usleep(500000); // mic alone first: must be silent
    AudioDeviceStart(sink, sinkProc);
    while (atomic_load(&gSinkRendered) < PLAY_FRAMES) usleep(10000);
    usleep(300000); // sink keeps running and writing zeros for a moment
    // Now stop the sink entirely: the next writes stop, so the ring holds the last audio it got.
    // To make that test meaningful, restart playback briefly so the ring's newest content is noise.
    atomic_store(&gSinkRendered, PLAY_FRAMES - 48000 / 10); // 100 ms more noise
    while (atomic_load(&gSinkRendered) < PLAY_FRAMES) usleep(1000);
    AudioDeviceStop(sink, sinkProc);
    UInt64 stopHost = mach_absolute_time();
    usleep(1500000);
    AudioDeviceStop(mic, micProc);
    AudioDeviceDestroyIOProcID(sink, sinkProc);
    AudioDeviceDestroyIOProcID(mic, micProc);

    long start = (long)gSinkStart;
    // a) silence before playback
    bool silentBefore = true;
    long recvBefore = 0;
    for (long i = 0; i < start; i++)
        if (gHaveRecv[i]) {
            recvBefore++;
            if (gRecv[i] != 0) silentBefore = false;
        }
    check(recvBefore > 0 && silentBefore, "mic is silent before the sink plays");

    // b) correlation over the main 2 s burst, lag search +-100 ms in the shared sample-time domain
    long len = PLAY_FRAMES - 48000 / 10;
    double bestR = 0;
    long bestLag = 0;
    for (long lag = -4800; lag <= 4800; lag++) {
        double sxy = 0, sxx = 0, syy = 0;
        for (long i = start; i < start + len; i += (lag == 0 ? 1 : 4)) {
            long j = i + lag;
            if (j < 0 || j >= TOTAL_FRAMES) continue;
            sxy += gSent[i] * gRecv[j];
            sxx += gSent[i] * gSent[i];
            syy += gRecv[j] * gRecv[j];
        }
        double r = (sxx > 0 && syy > 0) ? sxy / sqrt(sxx * syy) : 0;
        if (r > bestR) {
            bestR = r;
            bestLag = lag;
        }
    }
    printf("         correlation peak %.6f at lag %ld frames\n", bestR, bestLag);
    check(bestR > 0.999 && bestLag == 0, "mic reproduces the sink signal exactly (r > 0.999, lag 0)");

    long exact = 0, stereoOk = 0, delayN = 0;
    double dsum = 0, dmin = 1e9, dmax = 0;
    for (long i = start; i < start + len; i++) {
        if (gRecv[i] == gSent[i]) exact++;
        if (gRecvR[i] == gRecv[i]) stereoOk++;
        if (gHaveRecv[i] && gRenderHost[i] && gDeliverHost[i] >= gRenderHost[i]) {
            double d = (gDeliverHost[i] - gRenderHost[i]) * gTicksToMs;
            dsum += d;
            if (d < dmin) dmin = d;
            if (d > dmax) dmax = d;
            delayN++;
        }
    }
    printf("         bit-exact frames %ld / %ld, L==R %ld / %ld\n", exact, len, stereoOk, len);
    check(exact == len, "every played frame arrives bit-exact");
    if (delayN) {
        printf("         loopback delay (sink render -> mic delivery): mean %.2f ms, min %.2f, max %.2f\n",
               dsum / delayN, dmin, dmax);
        printf("         (= %.0f frames mean; IO buffers %u + %u frames = %.2f ms)\n",
               dsum / delayN * SR / 1000.0, sinkBuf, micBuf, (sinkBuf + micBuf) / SR * 1000.0);
    }
    check(delayN > 0 && dsum / delayN < 50.0, "loopback delay measured and < 50 ms");

    // c) silence after the sink stops
    long lastNonZero = -1, afterStop = 0;
    UInt64 lastNonZeroHost = 0;
    for (long i = 0; i < TOTAL_FRAMES; i++) {
        if (!gHaveRecv[i]) continue;
        if (gRecv[i] != 0) {
            lastNonZero = i;
            lastNonZeroHost = gDeliverHost[i];
        }
        if (gDeliverHost[i] > stopHost) afterStop++;
    }
    long silentTail = 0;
    for (long i = lastNonZero + 1; i < TOTAL_FRAMES; i++)
        if (gHaveRecv[i]) silentTail++;
    double lastAudioMs = lastNonZeroHost > stopHost ? (lastNonZeroHost - stopHost) * gTicksToMs
                                                    : -((double)(stopHost - lastNonZeroHost) * gTicksToMs);
    printf("         last non-zero mic frame delivered %.2f ms after AudioDeviceStop(sink) returned\n", lastAudioMs);
    printf("         then %ld consecutive silent frames (%.0f ms) until recording ended\n", silentTail,
           silentTail / SR * 1000.0);
    check(afterStop > 48000 && lastAudioMs < 50.0 && silentTail > 48000,
          "mic goes silent within 50 ms of the sink stopping and stays silent");

    printf("\nRESULT: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
