//! macOS backend: Core Audio HAL directly (no AudioUnit). Two IOProcs, one per device: the
//! physical mic (capture stage) and the hidden Oto Sink (output stage), joined by an `rtrb` ring.
//! The Oto HAL driver loops the sink to Oto Microphone.
//!
//! Notifications: the HAL is told to use its own notification thread (`RunLoop = NULL`), and
//! listener callbacks forward into the control thread's channel.

mod ffi;

use std::ffi::c_void;
use std::mem::size_of;
use std::ptr;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Once, OnceLock};

use ffi::*;
use oto_dsp::Chain;
use rtrb::RingBuffer;

use super::{HalEvent, Platform, Uids};
use crate::InputDevice;
use crate::control::Msg;
use crate::pipeline::{
    CaptureStage, DspSetup, DspThread, MAX_BLOCK, OutputStage, RING_CAPACITY, Shared,
    clamp_io_frames, downmix, estimate_latency_s, placement_for, scaled_io_frames, target_fill,
};

/// IO buffer we ask the mic for, in frames at 48 kHz (5.3 ms; leaves room for model inference).
/// Scaled to the device's rate, so the duration stays the same.
pub const IN_IO_FRAMES: u32 = 256;
/// IO buffer we ask the sink for, in frames at 48 kHz (2.7 ms; the output callback is cheap).
pub const OUT_IO_FRAMES: u32 = 128;

// ---------------------------------------------------------------------------------------------
// Property helpers

fn addr(sel: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: sel,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn get<T: Copy + Default>(obj: AudioObjectID, sel: u32, scope: u32) -> Result<T, OSStatus> {
    let a = addr(sel, scope);
    let mut v = T::default();
    let mut size = size_of::<T>() as u32;
    let st = unsafe {
        AudioObjectGetPropertyData(obj, &a, 0, ptr::null(), &mut size, (&raw mut v).cast())
    };
    if st == 0 { Ok(v) } else { Err(st) }
}

fn set<T>(obj: AudioObjectID, sel: u32, scope: u32, v: &T) -> Result<(), OSStatus> {
    let a = addr(sel, scope);
    let st = unsafe {
        AudioObjectSetPropertyData(
            obj,
            &a,
            0,
            ptr::null(),
            size_of::<T>() as u32,
            (v as *const T).cast(),
        )
    };
    if st == 0 { Ok(()) } else { Err(st) }
}

/// Raw property bytes, 8-byte aligned (enough for AudioBufferList).
fn get_raw(obj: AudioObjectID, sel: u32, scope: u32) -> Option<Vec<u64>> {
    let a = addr(sel, scope);
    let mut size = 0u32;
    if unsafe { AudioObjectGetPropertyDataSize(obj, &a, 0, ptr::null(), &mut size) } != 0 {
        return None;
    }
    let mut buf = vec![0u64; (size as usize).div_ceil(8).max(1)];
    let st = unsafe {
        AudioObjectGetPropertyData(obj, &a, 0, ptr::null(), &mut size, buf.as_mut_ptr().cast())
    };
    (st == 0).then_some(buf)
}

fn get_ids(obj: AudioObjectID, sel: u32, scope: u32) -> Vec<AudioObjectID> {
    let a = addr(sel, scope);
    let mut size = 0u32;
    if unsafe { AudioObjectGetPropertyDataSize(obj, &a, 0, ptr::null(), &mut size) } != 0 {
        return Vec::new();
    }
    let mut ids = vec![0u32; size as usize / 4];
    let st = unsafe {
        AudioObjectGetPropertyData(obj, &a, 0, ptr::null(), &mut size, ids.as_mut_ptr().cast())
    };
    if st != 0 {
        return Vec::new();
    }
    ids.truncate(size as usize / 4);
    ids
}

fn cf_to_string(s: CFStringRef) -> String {
    unsafe {
        let len = CFStringGetLength(s);
        let cap = CFStringGetMaximumSizeForEncoding(len, kCFStringEncodingUTF8) + 1;
        let mut buf = vec![0u8; cap as usize];
        if CFStringGetCString(s, buf.as_mut_ptr(), cap, kCFStringEncodingUTF8) == 0 {
            return String::new();
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    }
}

fn get_string(obj: AudioObjectID, sel: u32) -> Option<String> {
    let s: CFStringRef = get_ptr(obj, sel)?;
    let out = cf_to_string(s);
    unsafe { CFRelease(s) };
    Some(out)
}

fn get_ptr(obj: AudioObjectID, sel: u32) -> Option<CFStringRef> {
    let a = addr(sel, kAudioObjectPropertyScopeGlobal);
    let mut v: CFStringRef = ptr::null();
    let mut size = size_of::<CFStringRef>() as u32;
    let st = unsafe {
        AudioObjectGetPropertyData(obj, &a, 0, ptr::null(), &mut size, (&raw mut v).cast())
    };
    (st == 0 && !v.is_null()).then_some(v)
}

/// Finds a device by UID, including hidden ones such as Oto Sink.
pub fn device_for_uid(uid: &str) -> Option<AudioObjectID> {
    unsafe {
        let cf = CFStringCreateWithBytes(
            ptr::null(),
            uid.as_ptr(),
            uid.len() as isize,
            kCFStringEncodingUTF8,
            0,
        );
        if cf.is_null() {
            return None;
        }
        let a = addr(
            kAudioHardwarePropertyTranslateUIDToDevice,
            kAudioObjectPropertyScopeGlobal,
        );
        let mut id: AudioObjectID = kAudioObjectUnknown;
        let mut size = 4u32;
        let st = AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &a,
            size_of::<CFStringRef>() as u32,
            (&raw const cf).cast(),
            &mut size,
            (&raw mut id).cast(),
        );
        CFRelease(cf);
        (st == 0 && id != kAudioObjectUnknown).then_some(id)
    }
}

fn uid_of(dev: AudioObjectID) -> Option<String> {
    get_string(dev, kAudioDevicePropertyDeviceUID)
}

fn name_of(dev: AudioObjectID) -> String {
    get_string(dev, kAudioObjectPropertyName).unwrap_or_else(|| format!("device {dev}"))
}

fn channels(dev: AudioObjectID, scope: u32) -> u32 {
    let Some(buf) = get_raw(dev, kAudioDevicePropertyStreamConfiguration, scope) else {
        return 0;
    };
    let abl = buf.as_ptr() as *const AudioBufferList;
    unsafe { abl_slice(abl) }
        .iter()
        .map(|b| b.mNumberChannels)
        .sum()
}

/// The buffers of a (variable-length) AudioBufferList.
unsafe fn abl_slice<'a>(abl: *const AudioBufferList) -> &'a [AudioBuffer] {
    unsafe {
        let n = (*abl).mNumberBuffers as usize;
        std::slice::from_raw_parts(ptr::addr_of!((*abl).mBuffers).cast::<AudioBuffer>(), n)
    }
}

unsafe fn abl_slice_mut<'a>(abl: *mut AudioBufferList) -> &'a mut [AudioBuffer] {
    unsafe {
        let n = (*abl).mNumberBuffers as usize;
        std::slice::from_raw_parts_mut(ptr::addr_of_mut!((*abl).mBuffers).cast::<AudioBuffer>(), n)
    }
}

fn default_input() -> Option<AudioObjectID> {
    get::<u32>(
        kAudioObjectSystemObject,
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioObjectPropertyScopeGlobal,
    )
    .ok()
    .filter(|&d| d != kAudioObjectUnknown)
}

fn is_oto(uid: &str, uids: &Uids) -> bool {
    uid.starts_with("dev.oto.") || uid == uids.mic || uid == uids.sink
}

fn rate_of(dev: AudioObjectID) -> f64 {
    get::<f64>(
        dev,
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
    )
    .unwrap_or(0.0)
}

fn is_running_somewhere(dev: AudioObjectID) -> bool {
    get::<u32>(
        dev,
        kAudioDevicePropertyDeviceIsRunningSomewhere,
        kAudioObjectPropertyScopeGlobal,
    )
    .is_ok_and(|v| v != 0)
}

/// Device + stream latency plus safety offset, in device frames.
fn io_latency_frames(dev: AudioObjectID, scope: u32) -> u32 {
    let dev_lat = get::<u32>(dev, kAudioDevicePropertyLatency, scope).unwrap_or(0);
    let safety = get::<u32>(dev, kAudioDevicePropertySafetyOffset, scope).unwrap_or(0);
    let stream = get_ids(dev, kAudioDevicePropertyStreams, scope)
        .first()
        .and_then(|&s| {
            get::<u32>(
                s,
                kAudioStreamPropertyLatency,
                kAudioObjectPropertyScopeGlobal,
            )
            .ok()
        })
        .unwrap_or(0);
    dev_lat + safety + stream
}

fn check_float(dev: AudioObjectID, scope: u32, what: &str) -> Result<(), String> {
    let streams = get_ids(dev, kAudioDevicePropertyStreams, scope);
    let Some(&s) = streams.first() else {
        return Err(format!("{what} has no streams"));
    };
    let f = get::<AudioStreamBasicDescription>(
        s,
        kAudioStreamPropertyVirtualFormat,
        kAudioObjectPropertyScopeGlobal,
    )
    .map_err(|e| format!("{what}: cannot read format ({})", status_str(e)))?;
    if f.mFormatID != kAudioFormatLinearPCM
        || f.mFormatFlags & kAudioFormatFlagIsFloat == 0
        || f.mBitsPerChannel != 32
    {
        return Err(format!("{what}: unsupported stream format {f:?}"));
    }
    Ok(())
}

fn set_buffer_frames(dev: AudioObjectID, frames: u32) -> u32 {
    let g = kAudioObjectPropertyScopeGlobal;
    let range = get::<AudioValueRange>(dev, kAudioDevicePropertyBufferFrameSizeRange, g).unwrap_or(
        AudioValueRange {
            mMinimum: 0.0,
            mMaximum: 1e9,
        },
    );
    let want = clamp_io_frames(frames, range.mMinimum, range.mMaximum);
    let _ = set(dev, kAudioDevicePropertyBufferFrameSize, g, &want);
    get::<u32>(dev, kAudioDevicePropertyBufferFrameSize, g).unwrap_or(want)
}

fn timebase() -> (u64, u64) {
    static TB: OnceLock<(u64, u64)> = OnceLock::new();
    *TB.get_or_init(|| {
        let mut i = MachTimebaseInfo::default();
        unsafe { mach_timebase_info(&mut i) };
        if i.denom == 0 {
            (1, 1)
        } else {
            (i.numer as u64, i.denom as u64)
        }
    })
}

/// Host clock ticks (mach_absolute_time / AudioTimeStamp::mHostTime) to nanoseconds.
pub fn host_to_ns(t: u64) -> u64 {
    let (n, d) = timebase();
    (t as u128 * n as u128 / d as u128) as u64
}

/// Gives the calling thread (the DSP worker) realtime scheduling: Mach time-constraint policy
/// with a `period_ns` period, the whole period as its deadline and half of it as its expected
/// computation, like an audio IO thread. If the kernel refuses, falls back to the
/// user-interactive QoS class. Returns which one applied.
pub fn promote_dsp_thread(period_ns: u64) -> Result<&'static str, String> {
    let (n, d) = timebase();
    let ticks = |ns: u64| (ns as u128 * d as u128 / n as u128).min(u32::MAX as u128) as u32;
    // The kernel rejects computation below 50 µs or above 50 ms.
    let period_ns = period_ns.clamp(1_000_000, 50_000_000);
    let mut policy = ThreadTimeConstraintPolicy {
        period: ticks(period_ns),
        computation: ticks(period_ns / 2),
        constraint: ticks(period_ns),
        preemptible: 1,
    };
    let st = unsafe {
        thread_policy_set(
            pthread_mach_thread_np(pthread_self()),
            THREAD_TIME_CONSTRAINT_POLICY,
            (&raw mut policy).cast(),
            THREAD_TIME_CONSTRAINT_POLICY_COUNT,
        )
    };
    if st == 0 {
        return Ok("time-constraint");
    }
    if unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) } == 0 {
        return Ok("user-interactive QoS");
    }
    Err(format!("thread_policy_set failed ({st})"))
}

/// A retained `os_workgroup_t`.
struct Workgroup(*mut c_void);

// SAFETY: os_workgroup objects are thread-safe, reference-counted OS objects; joining from
// another thread is their purpose.
unsafe impl Send for Workgroup {}

impl Drop for Workgroup {
    fn drop(&mut self) {
        unsafe { os_release(self.0) };
    }
}

/// The audio workgroup of `dev`'s IO thread, if the system provides one.
fn io_workgroup(dev: AudioObjectID) -> Option<Workgroup> {
    let a = addr(
        kAudioDevicePropertyIOThreadOSWorkgroup,
        kAudioObjectPropertyScopeGlobal,
    );
    let mut wg: *mut c_void = ptr::null_mut();
    let mut size = size_of::<*mut c_void>() as u32;
    let st = unsafe {
        AudioObjectGetPropertyData(dev, &a, 0, ptr::null(), &mut size, (&raw mut wg).cast())
    };
    (st == 0 && !wg.is_null()).then_some(Workgroup(wg))
}

/// Membership of a workgroup; leaves it on drop, which must happen on the thread that joined.
struct Joined {
    wg: Workgroup,
    token: OsWorkgroupJoinToken,
    joined: bool,
}

impl Drop for Joined {
    fn drop(&mut self) {
        if self.joined {
            unsafe { os_workgroup_leave(self.wg.0, &mut self.token) };
        }
    }
}

/// The DSP worker's setup for a capture device: realtime priority, then membership of the
/// device's IO-thread workgroup. Without the workgroup, Apple Silicon runs the worker's short
/// bursts on efficiency cores or at low clocks when the machine is otherwise idle; in it, the
/// scheduler knows the worker shares the audio deadline. Measured on an M2 Pro (release,
/// `live_worker_on_a_real_mic`, 256-frame mic): DeepFilterNet3 worker time per block p99
/// 3.67 ms / max 3.94 ms without, 1.57 ms / 1.59 ms with, of a 5.33 ms period.
fn dsp_setup(dev: AudioObjectID) -> DspSetup {
    let wg = io_workgroup(dev);
    Box::new(move |period_ns| {
        let realtime = promote_dsp_thread(period_ns).is_ok();
        let guard = wg.and_then(|wg| {
            // Boxed first: the token must stay where it was when joining.
            let mut j = Box::new(Joined {
                wg,
                token: OsWorkgroupJoinToken {
                    sig: 0,
                    opaque: [0; 36],
                },
                joined: false,
            });
            j.joined = unsafe { os_workgroup_join(j.wg.0, &mut j.token) } == 0;
            j.joined.then_some(j as Box<dyn std::any::Any>)
        });
        DspThread {
            realtime,
            workgroup: guard.is_some(),
            guard,
        }
    })
}

fn use_hal_notification_thread() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let rl: *const c_void = ptr::null();
        let _ = set(
            kAudioObjectSystemObject,
            kAudioHardwarePropertyRunLoop,
            kAudioObjectPropertyScopeGlobal,
            &rl,
        );
    });
}

// ---------------------------------------------------------------------------------------------
// Public queries (used from any thread)

struct Candidate {
    dev: InputDevice,
    builtin: bool,
}

fn candidates(uids: &Uids) -> Vec<Candidate> {
    let default = default_input();
    get_ids(
        kAudioObjectSystemObject,
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
    )
    .into_iter()
    .filter(|&d| channels(d, kAudioObjectPropertyScopeInput) > 0)
    .filter_map(|d| {
        let uid = uid_of(d)?;
        if is_oto(&uid, uids) {
            return None;
        }
        let transport = get::<u32>(
            d,
            kAudioDevicePropertyTransportType,
            kAudioObjectPropertyScopeGlobal,
        )
        .unwrap_or(0);
        Some(Candidate {
            dev: InputDevice {
                id: uid,
                name: name_of(d),
                is_default: Some(d) == default,
            },
            builtin: transport == kAudioDeviceTransportTypeBuiltIn,
        })
    })
    .collect()
}

pub fn input_devices(uids: &Uids) -> Vec<InputDevice> {
    candidates(uids).into_iter().map(|c| c.dev).collect()
}

pub fn driver_installed(uids: &Uids) -> bool {
    device_for_uid(&uids.mic).is_some() && device_for_uid(&uids.sink).is_some()
}

pub fn default_input_uid() -> Option<String> {
    default_input().and_then(uid_of)
}

pub fn set_default_input(uid: &str) -> Result<(), String> {
    let dev = device_for_uid(uid).ok_or_else(|| format!("no device with UID {uid}"))?;
    set(
        kAudioObjectSystemObject,
        kAudioHardwarePropertyDefaultInputDevice,
        kAudioObjectPropertyScopeGlobal,
        &dev,
    )
    .map_err(|e| format!("cannot set default input: {}", status_str(e)))
}

/// A virtual device's state, for `oto status`.
pub fn device_status(uid: &str, scope_input: bool) -> Option<crate::DeviceStatus> {
    let dev = device_for_uid(uid)?;
    let scope = if scope_input {
        kAudioObjectPropertyScopeInput
    } else {
        kAudioObjectPropertyScopeOutput
    };
    Some(crate::DeviceStatus {
        name: name_of(dev),
        sample_rate: rate_of(dev),
        channels: channels(dev, scope),
        in_use: is_running_somewhere(dev),
    })
}

// ---------------------------------------------------------------------------------------------
// Listeners

struct ListenerCtx {
    tx: Sender<Msg>,
    ev: HalEvent,
}

unsafe extern "C" fn on_property(
    _obj: AudioObjectID,
    _n: u32,
    _addrs: *const AudioObjectPropertyAddress,
    data: *mut c_void,
) -> OSStatus {
    let ctx = unsafe { &*(data as *const ListenerCtx) };
    let _ = ctx.tx.send(Msg::Hal(ctx.ev));
    0
}

/// A registered property listener; removed on drop.
///
/// The context is intentionally leaked: the HAL may be mid-callback on its notification thread
/// while we remove the listener, and a few dozen bytes per device event is a fair price for
/// never touching freed memory.
struct Listener {
    obj: AudioObjectID,
    addr: AudioObjectPropertyAddress,
    ctx: *mut c_void,
}

impl Listener {
    fn new(obj: AudioObjectID, sel: u32, tx: &Sender<Msg>, ev: HalEvent) -> Option<Self> {
        let addr = addr(sel, kAudioObjectPropertyScopeGlobal);
        let ctx = Box::leak(Box::new(ListenerCtx { tx: tx.clone(), ev })) as *mut ListenerCtx;
        let st = unsafe { AudioObjectAddPropertyListener(obj, &addr, on_property, ctx.cast()) };
        (st == 0).then_some(Self {
            obj,
            addr,
            ctx: ctx.cast(),
        })
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        unsafe { AudioObjectRemovePropertyListener(self.obj, &self.addr, on_property, self.ctx) };
    }
}

// ---------------------------------------------------------------------------------------------
// Streams

struct InCtx {
    stage: CaptureStage,
    shared: Arc<Shared>,
    rate: f64,
}

struct OutCtx {
    stage: OutputStage,
    shared: Arc<Shared>,
    mono: Box<[f32]>,
}

fn first_sample_ns(ts: *const AudioTimeStamp) -> u64 {
    if ts.is_null() {
        return 0;
    }
    let ts = unsafe { &*ts };
    if ts.mFlags & kAudioTimeStampHostTimeValid == 0 {
        0
    } else {
        host_to_ns(ts.mHostTime)
    }
}

unsafe extern "C" fn input_proc(
    _dev: AudioObjectID,
    _now: *const AudioTimeStamp,
    input: *const AudioBufferList,
    input_time: *const AudioTimeStamp,
    _output: *mut AudioBufferList,
    _output_time: *const AudioTimeStamp,
    data: *mut c_void,
) -> OSStatus {
    let t0 = unsafe { mach_absolute_time() };
    let ctx = unsafe { &mut *(data as *mut InCtx) };
    if input.is_null() {
        return 0;
    }
    let bufs = unsafe { abl_slice(input) };
    let frames = bufs
        .iter()
        .filter(|b| b.mNumberChannels > 0)
        .map(|b| b.mDataByteSize as usize / 4 / b.mNumberChannels as usize)
        .min()
        .unwrap_or(0);
    if frames == 0 {
        return 0;
    }
    let first_ns = first_sample_ns(input_time);
    let mut at = 0;
    while at < frames {
        let n = (frames - at).min(MAX_BLOCK);
        let mono = &mut ctx.stage.mono_mut()[..n];
        // Each buffer's frames `at..at + n`; a buffer without data is passed as empty.
        let parts = bufs.iter().map(|b| {
            let ch = b.mNumberChannels as usize;
            if b.mData.is_null() || ch == 0 {
                return (&[][..], ch);
            }
            let data = unsafe { std::slice::from_raw_parts(b.mData as *const f32, frames * ch) };
            (&data[at * ch..(at + n) * ch], ch)
        });
        downmix(parts, mono);
        let ns = if first_ns == 0 {
            0
        } else {
            first_ns + (at as f64 * 1e9 / ctx.rate) as u64
        };
        ctx.stage.process(n, ns);
        at += n;
    }
    ctx.shared.in_cycles.fetch_add(1, Relaxed);
    let dt = host_to_ns(unsafe { mach_absolute_time() } - t0);
    ctx.shared.busy_ns.fetch_add(dt, Relaxed);
    ctx.shared.capture_times.record(dt);
    0
}

unsafe extern "C" fn output_proc(
    _dev: AudioObjectID,
    _now: *const AudioTimeStamp,
    _input: *const AudioBufferList,
    _input_time: *const AudioTimeStamp,
    output: *mut AudioBufferList,
    _output_time: *const AudioTimeStamp,
    data: *mut c_void,
) -> OSStatus {
    let t0 = unsafe { mach_absolute_time() };
    let ctx = unsafe { &mut *(data as *mut OutCtx) };
    if output.is_null() {
        return 0;
    }
    let bufs = unsafe { abl_slice_mut(output) };
    let frames = bufs
        .iter()
        .filter(|b| b.mNumberChannels > 0)
        .map(|b| b.mDataByteSize as usize / 4 / b.mNumberChannels as usize)
        .min()
        .unwrap_or(0);
    let mut at = 0;
    while at < frames {
        let n = (frames - at).min(MAX_BLOCK);
        let mono = &mut ctx.mono[..n];
        ctx.stage.render(mono);
        for b in bufs.iter_mut() {
            let ch = b.mNumberChannels as usize;
            if b.mData.is_null() || ch == 0 {
                continue;
            }
            let data = unsafe { std::slice::from_raw_parts_mut(b.mData as *mut f32, frames * ch) };
            for (frame, s) in data[at * ch..(at + n) * ch]
                .chunks_exact_mut(ch)
                .zip(mono.iter())
            {
                frame.fill(*s);
            }
        }
        at += n;
    }
    ctx.shared.out_cycles.fetch_add(1, Relaxed);
    let dt = host_to_ns(unsafe { mach_absolute_time() } - t0);
    ctx.shared.busy_ns.fetch_add(dt, Relaxed);
    ctx.shared.output_times.record(dt);
    0
}

/// One running IOProc. Dropping stops it (synchronously: the HAL guarantees the proc is not
/// running after `AudioDeviceStop` returns on a non-IO thread), then frees its context.
struct IoProc<C> {
    dev: AudioObjectID,
    id: AudioDeviceIOProcID,
    ctx: *mut C,
    started: bool,
}

impl<C> IoProc<C> {
    fn new(dev: AudioObjectID, f: AudioDeviceIOProc, ctx: C, what: &str) -> Result<Self, String> {
        let ctx = Box::into_raw(Box::new(ctx));
        let mut id: AudioDeviceIOProcID = None;
        let st = unsafe { AudioDeviceCreateIOProcID(dev, f, ctx.cast(), &mut id) };
        if st != 0 || id.is_none() {
            drop(unsafe { Box::from_raw(ctx) });
            return Err(format!(
                "{what}: cannot create IO proc ({})",
                status_str(st)
            ));
        }
        Ok(Self {
            dev,
            id,
            ctx,
            started: false,
        })
    }

    fn start(&mut self, what: &str) -> Result<(), String> {
        let st = unsafe { AudioDeviceStart(self.dev, self.id) };
        if st != 0 {
            return Err(format!("{what}: cannot start ({})", status_str(st)));
        }
        self.started = true;
        Ok(())
    }
}

impl<C> Drop for IoProc<C> {
    fn drop(&mut self) {
        unsafe {
            if self.started {
                AudioDeviceStop(self.dev, self.id);
            }
            AudioDeviceDestroyIOProcID(self.dev, self.id);
            drop(Box::from_raw(self.ctx));
        }
    }
}

struct Streams {
    // Field order is drop order: listeners first, then the mic (producer), then the sink.
    _listeners: Vec<Listener>,
    _input: IoProc<InCtx>,
    _output: IoProc<OutCtx>,
}

// ---------------------------------------------------------------------------------------------
// The platform

pub(crate) type Backend = Mac;

pub(crate) struct Mac {
    uids: Uids,
    tx: Sender<Msg>,
    streams: Option<Streams>,
    mic_watch: Option<(AudioObjectID, Listener)>,
    _global: Vec<Listener>,
    last_physical_default: std::cell::RefCell<Option<String>>,
}

impl Mac {
    pub fn new(uids: Uids, tx: Sender<Msg>) -> Result<Self, String> {
        use_hal_notification_thread();
        let sys = kAudioObjectSystemObject;
        let global = [
            (kAudioHardwarePropertyDevices, HalEvent::DeviceList),
            (
                kAudioHardwarePropertyDefaultInputDevice,
                HalEvent::DefaultInput,
            ),
        ]
        .into_iter()
        .filter_map(|(sel, ev)| Listener::new(sys, sel, &tx, ev))
        .collect::<Vec<_>>();
        if global.len() != 2 {
            return Err("cannot watch Core Audio devices".into());
        }
        let mut m = Self {
            uids,
            tx,
            streams: None,
            mic_watch: None,
            _global: global,
            last_physical_default: Default::default(),
        };
        m.refresh();
        // Remember the physical default now, so making Oto Microphone the default while idle
        // (`oto run --set-default`) still captures from it once an app starts recording.
        let _ = m.resolve_input(None);
        Ok(m)
    }
}

impl Platform for Mac {
    fn driver_installed(&self) -> bool {
        driver_installed(&self.uids)
    }

    fn clients_active(&self) -> bool {
        device_for_uid(&self.uids.mic).is_some_and(is_running_somewhere)
    }

    fn input_devices(&self) -> Vec<InputDevice> {
        input_devices(&self.uids)
    }

    fn resolve_input(&self, preferred: Option<&str>) -> Option<String> {
        let c = candidates(&self.uids);
        if let Some(p) = preferred.and_then(|p| c.iter().find(|c| c.dev.id == p)) {
            return Some(p.dev.id.clone());
        }
        if let Some(d) = c.iter().find(|c| c.dev.is_default) {
            *self.last_physical_default.borrow_mut() = Some(d.dev.id.clone());
            return Some(d.dev.id.clone());
        }
        // The default input is Oto Microphone (or none): use the last physical default we saw,
        // then the built-in mic, then anything.
        let last = self.last_physical_default.borrow().clone();
        last.and_then(|l| c.iter().find(|c| c.dev.id == l))
            .or_else(|| c.iter().find(|c| c.builtin))
            .or(c.first())
            .map(|c| c.dev.id.clone())
    }

    fn start(&mut self, input: &str, chain: Chain, shared: &Arc<Shared>) -> Result<(), String> {
        self.stop();
        if is_oto(input, &self.uids) {
            return Err("refusing to capture from an Oto device (feedback loop)".into());
        }
        let in_dev = device_for_uid(input).ok_or_else(|| format!("microphone {input} is gone"))?;
        let out_dev = device_for_uid(&self.uids.sink).ok_or("Oto Sink is not available")?;
        let (ins, outs) = (
            kAudioObjectPropertyScopeInput,
            kAudioObjectPropertyScopeOutput,
        );
        check_float(in_dev, ins, "microphone")?;
        check_float(out_dev, outs, "Oto Sink")?;
        if channels(in_dev, ins) == 0 || channels(out_dev, outs) == 0 {
            return Err("device has no channels in the needed direction".into());
        }
        let (in_rate, out_rate) = (rate_of(in_dev), rate_of(out_dev));
        if !(in_rate > 0.0 && out_rate > 0.0) {
            return Err("cannot read device sample rates".into());
        }
        // Same buffer duration whatever the device rate (a 16 kHz headset gets 85 frames, not 256).
        let in_frames = set_buffer_frames(in_dev, scaled_io_frames(IN_IO_FRAMES, in_rate));
        let out_frames = set_buffer_frames(out_dev, scaled_io_frames(OUT_IO_FRAMES, out_rate));
        let (prod, cons) = RingBuffer::new(RING_CAPACITY);
        let target = target_fill(in_frames as usize, in_rate, out_frames as usize, out_rate);
        let placement = placement_for(chain.model());
        let cap = CaptureStage::build(
            in_rate,
            in_frames as usize,
            chain,
            prod,
            shared.clone(),
            placement,
            dsp_setup(in_dev),
        )?;
        let out = OutputStage::new(cons, out_rate, target, shared.clone())?;

        let est_s = estimate_latency_s(
            in_frames as usize,
            io_latency_frames(in_dev, ins) as usize,
            in_rate,
            cap.latency_frames(),
            target,
            io_latency_frames(out_dev, outs) as usize + out.latency_frames(),
            out_rate,
        );
        shared.estimate_us.store((est_s * 1e6) as u32, Relaxed);
        shared.in_rate.store(in_rate as u32, Relaxed);
        shared.out_rate.store(out_rate as u32, Relaxed);
        // The callbacks are not running yet: start their timing statistics afresh.
        for (t, frames, rate) in [
            (&shared.capture_times, in_frames, in_rate),
            (&shared.output_times, out_frames, out_rate),
        ] {
            t.reset();
            t.period_ns
                .store((frames as f64 * 1e9 / rate) as u64, Relaxed);
        }

        let in_ctx = InCtx {
            stage: cap,
            shared: shared.clone(),
            rate: in_rate,
        };
        let out_ctx = OutCtx {
            stage: out,
            shared: shared.clone(),
            mono: vec![0.0; MAX_BLOCK].into(),
        };
        let mut input = IoProc::new(in_dev, input_proc, in_ctx, "microphone")?;
        let mut output = IoProc::new(out_dev, output_proc, out_ctx, "Oto Sink")?;
        output.start("Oto Sink")?;
        input.start("microphone")?;

        let mut listeners = Vec::new();
        for dev in [in_dev, out_dev] {
            for sel in [
                kAudioDevicePropertyDeviceIsAlive,
                kAudioDevicePropertyNominalSampleRate,
            ] {
                listeners.extend(Listener::new(dev, sel, &self.tx, HalEvent::StreamDevice));
            }
        }
        self.streams = Some(Streams {
            _listeners: listeners,
            _input: input,
            _output: output,
        });
        Ok(())
    }

    fn stop(&mut self) {
        self.streams = None;
    }

    fn refresh(&mut self) {
        let mic = device_for_uid(&self.uids.mic);
        if mic == self.mic_watch.as_ref().map(|(id, _)| *id) {
            return;
        }
        self.mic_watch = mic.and_then(|id| {
            Listener::new(
                id,
                kAudioDevicePropertyDeviceIsRunningSomewhere,
                &self.tx,
                HalEvent::MicClients,
            )
            .map(|l| (id, l))
        });
    }
}

// ---------------------------------------------------------------------------------------------
// Recorder (for `oto latency`)

type RecFn = Box<dyn FnMut(&[f32], usize, u64) + Send>;

struct RecCtx {
    f: RecFn,
}

unsafe extern "C" fn record_proc(
    _dev: AudioObjectID,
    _now: *const AudioTimeStamp,
    input: *const AudioBufferList,
    input_time: *const AudioTimeStamp,
    _output: *mut AudioBufferList,
    _output_time: *const AudioTimeStamp,
    data: *mut c_void,
) -> OSStatus {
    let ctx = unsafe { &mut *(data as *mut RecCtx) };
    if input.is_null() {
        return 0;
    }
    if let Some(b) = unsafe { abl_slice(input) }.first()
        && !b.mData.is_null()
        && b.mNumberChannels > 0
    {
        let s = unsafe {
            std::slice::from_raw_parts(b.mData as *const f32, b.mDataByteSize as usize / 4)
        };
        (ctx.f)(s, b.mNumberChannels as usize, first_sample_ns(input_time));
    }
    0
}

/// Records from an input device by UID. The callback gets the first buffer's interleaved
/// samples, its channel count and the host-clock time of its first frame (ns, 0 if unknown).
pub struct Recorder {
    _io: IoProc<RecCtx>,
}

impl Recorder {
    pub fn start(uid: &str, f: RecFn) -> Result<Self, String> {
        let dev = device_for_uid(uid).ok_or_else(|| format!("no device with UID {uid}"))?;
        check_float(dev, kAudioObjectPropertyScopeInput, uid)?;
        let mut io = IoProc::new(dev, record_proc, RecCtx { f }, uid)?;
        io.start(uid)?;
        Ok(Self { _io: io })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Levels;
    use crate::pipeline::{Placement, default_dsp_setup};
    use oto_dsp::Model;
    use std::time::Duration;

    /// Live check of the DSP worker on a real mic's IO thread, without Oto's devices (so it
    /// runs next to a running Oto app): the mic's callback drives `CaptureStage` exactly like
    /// the engine's and its output is discarded. Compares the worker with and without the mic's
    /// audio workgroup. Opens the mic for `OTO_LIVE_SECONDS` (default 15) per variant. The
    /// signal is synthetic noisy speech in place of the mic's (like `oto run --input-raw`).
    ///
    /// `cargo test --release -p oto-engine --lib live_worker -- --ignored --nocapture`
    /// (`OTO_LIVE_MIC=UID` picks the mic; default the built-in one).
    #[test]
    #[ignore = "live: opens a real microphone"]
    fn live_worker_on_a_real_mic() {
        let uid = std::env::var("OTO_LIVE_MIC").unwrap_or("BuiltInMicrophoneDevice".into());
        let secs: f64 = std::env::var("OTO_LIVE_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(15.0);
        let dev = device_for_uid(&uid).expect("no such microphone");
        let rate = rate_of(dev);
        let frames = set_buffer_frames(dev, scaled_io_frames(IN_IO_FRAMES, rate)) as usize;
        let model = if Model::DeepFilterNet3.available() {
            Model::DeepFilterNet3
        } else {
            Model::RNNoise
        };
        for workgroup in [false, true] {
            // The mic clocks the callbacks; the signal is synthetic noisy speech, so the model
            // really runs (a mic without permission delivers digital silence, which it skips).
            let mut shared = Shared::new(Arc::new(Levels::default()), None);
            shared.source = Some(oto_dsp::signal::noisy_speech(rate as usize * 4, 1).into());
            let shared = Arc::new(shared);
            let (prod, mut cons) = RingBuffer::new(RING_CAPACITY);
            let setup = if workgroup {
                dsp_setup(dev)
            } else {
                default_dsp_setup()
            };
            let chain = Chain::with_model(model).unwrap();
            let mut cap = CaptureStage::build(
                rate,
                frames,
                chain,
                prod,
                shared.clone(),
                Placement::Worker,
                setup,
            )
            .unwrap();
            let sh = shared.clone();
            let rec = Recorder::start(
                &uid,
                Box::new(move |data, ch, ns| {
                    let t0 = unsafe { mach_absolute_time() };
                    let n = (data.len() / ch).min(MAX_BLOCK);
                    downmix(
                        [(&data[..n * ch], ch)].into_iter(),
                        &mut cap.mono_mut()[..n],
                    );
                    cap.process(n, ns);
                    if let Ok(c) = cons.read_chunk(cons.slots()) {
                        c.commit_all();
                    }
                    let dt = host_to_ns(unsafe { mach_absolute_time() } - t0);
                    sh.capture_times.record(dt);
                }),
            )
            .unwrap();
            std::thread::sleep(Duration::from_secs_f64(secs));
            drop(rec);
            let s = crate::Stats::from_shared(&shared);
            let w = s.worker;
            eprintln!(
                "{} on {uid} ({rate} Hz, {frames} frames), workgroup {} (joined {}), realtime \
                 {}: worker p50 {:.0} µs, p99 {:.0} µs, max {:.0} µs of {:.0} µs over {} \
                 blocks; starvations {}, overflows {}, reserve {:.2} ms; callback p99 {:.0} \
                 µs, max {:.0} µs",
                model.name(),
                workgroup,
                s.worker_workgroup,
                s.worker_realtime,
                w.p50_us,
                w.p99_us,
                w.max_us,
                w.period_us,
                w.callbacks,
                s.starvations,
                s.worker_overflows,
                s.worker_reserve_ms,
                s.capture.p99_us,
                s.capture.max_us,
            );
        }
    }

    /// Live proof of plan 0001 step 5 on the real devices: a running engine (forced client,
    /// denoise bypassed so a steady tone passes whatever the model; the model's delay is kept)
    /// switches Standard → Light → Standard several times while Oto Microphone is recorded like
    /// an app would. For each switch: the longest silent run and the largest sample-to-sample
    /// step around it against the steady state's (a click is a step well above it). Needs the
    /// Oto driver and no other running engine.
    ///
    /// `cargo test --release -p oto-engine --lib live_tier_switch -- --ignored --nocapture`
    /// (`OTO_LIVE_MIC=UID` picks the mic that clocks the stream; default the built-in one;
    /// `OTO_LIVE_SWITCHES` the number of switches, default 6).
    #[test]
    #[ignore = "live: runs the engine on the real mic and Oto devices"]
    fn live_tier_switch_is_short_and_click_free() {
        use crate::{Command, Engine, EngineOptions, Settings, Tier, names};
        use std::sync::Mutex;
        const QUIET: f32 = 0.01;
        const AMP: f32 = 0.3;
        const HZ: f64 = 480.0;
        let uid = std::env::var("OTO_LIVE_MIC").unwrap_or("BuiltInMicrophoneDevice".into());
        let n_switches: usize = std::env::var("OTO_LIVE_SWITCHES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6);
        let spacing = Duration::from_secs(3);
        let in_rate = rate_of(device_for_uid(&uid).expect("no such microphone"));
        let mic = device_for_uid(names::MIC_UID).expect("Oto driver not installed");
        let rec_rate = rate_of(mic);
        // One second of a 480 Hz tone at the mic's rate: whole periods, so the loop is seamless.
        let n = in_rate as usize;
        let tone: Arc<[f32]> = (0..n)
            .map(|k| AMP * (2.0 * std::f64::consts::PI * HZ * k as f64 / in_rate).sin() as f32)
            .collect();
        let engine = Engine::start_with(
            Settings {
                enabled: false,
                input_device: Some(uid.clone()),
                tier: Tier::Standard,
                ..Default::default()
            },
            EngineOptions {
                force_clients_active: true,
                source: Some(tone),
                ..Default::default()
            },
        )
        .expect("engine start");

        // Channel 0 of Oto Microphone, plus (sample index, host ns) of each callback's start.
        type Rec = (Vec<f32>, Vec<(usize, u64)>);
        let rec: Arc<Mutex<Rec>> = Arc::new(Mutex::new((
            Vec::with_capacity(rec_rate as usize * 30),
            Vec::new(),
        )));
        let r = rec.clone();
        let recorder = Recorder::start(
            names::MIC_UID,
            Box::new(move |data, ch, ns| {
                let mut g = r.lock().unwrap();
                let at = g.0.len();
                g.1.push((at, ns));
                g.0.extend(data.chunks_exact(ch).map(|f| f[0]));
            }),
        )
        .expect("record Oto Microphone");
        let now_ns = || host_to_ns(unsafe { mach_absolute_time() });

        std::thread::sleep(spacing);
        let s0 = engine.stats();
        eprintln!(
            "start: model {:?}, mic {uid} at {in_rate} Hz, Oto Microphone at {rec_rate} Hz, \
             latency est {:.1} ms",
            s0.model, s0.estimated_latency_ms
        );
        let mut switches = Vec::new();
        let mut models = vec![s0.model];
        for i in 0..n_switches {
            let tier = if i % 2 == 0 {
                Tier::Light
            } else {
                Tier::Standard
            };
            switches.push((tier, now_ns()));
            engine.send(Command::SetTier(tier));
            std::thread::sleep(spacing);
            models.push(engine.stats().model);
        }
        let s1 = engine.stats();
        drop(recorder);
        drop(engine);

        let (samples, stamps) = std::mem::take(&mut *rec.lock().unwrap());
        // Sample index for a host time, from the nearest preceding callback stamp.
        let index_at = |t: u64| -> usize {
            let k = stamps.partition_point(|&(_, ns)| ns <= t).saturating_sub(1);
            let (i, ns) = stamps[k];
            i + (t.saturating_sub(ns) as f64 * rec_rate / 1e9) as usize
        };
        let ms = |samples: usize| samples as f64 * 1000.0 / rec_rate;
        let max_step = |a: usize, b: usize| -> (f32, usize) {
            (a.max(1)..b.min(samples.len()))
                .map(|i| ((samples[i] - samples[i - 1]).abs(), i))
                .fold((0.0, 0), |m, x| if x.0 > m.0 { x } else { m })
        };
        let at = |t: u64, off_ms: f64| -> usize {
            (index_at(t) as f64 + off_ms * rec_rate / 1000.0).max(0.0) as usize
        };

        eprintln!(
            "models after each step: {:?}",
            models
                .iter()
                .map(|m| m.map_or("none", |m| m.name()))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "underruns {} -> {}, resyncs {} -> {}, overflows {} -> {}, starvations {} -> {}",
            s0.underruns,
            s1.underruns,
            s0.resyncs,
            s1.resyncs,
            s0.overflows,
            s1.overflows,
            s0.starvations,
            s1.starvations
        );
        let mut failures = Vec::new();
        for (i, &(tier, t)) in switches.iter().enumerate() {
            // Steady state: the 2 s before the switch, ending 200 ms before it.
            let (steady, _) = max_step(at(t, -2200.0), at(t, -200.0));
            // Around the switch: from 100 ms before to 2.5 s after (load + fade + restart).
            let (a, b) = (at(t, -100.0), at(t, 2500.0).min(samples.len()));
            let (step, step_i) = max_step(a, b);
            let (mut gap, mut gap_at, mut run) = (0usize, 0usize, 0usize);
            for (k, x) in samples.iter().enumerate().take(b).skip(a) {
                if x.abs() < QUIET {
                    run += 1;
                    if run > gap {
                        gap = run;
                        gap_at = k + 1 - run;
                    }
                } else {
                    run = 0;
                }
            }
            let rel = |k: usize| ms(k) - ms(index_at(t));
            eprintln!(
                "switch {i} -> {tier:?}: silent gap {:.1} ms (from +{:.1} ms), largest step \
                 {step:.4} at +{:.1} ms, steady-state max step {steady:.4} (ratio {:.2})",
                ms(gap),
                rel(gap_at),
                rel(step_i),
                step / steady.max(1e-9),
            );
            if ms(gap) >= 500.0 {
                failures.push(format!("switch {i}: gap {:.1} ms", ms(gap)));
            }
            if ms(gap) < 1.0 {
                failures.push(format!("switch {i}: no gap at all: did it switch?"));
            }
            // The tone moves ≤ 2π·480/rate·0.3 per sample (0.019 at 48 kHz); fades only lower it.
            if step > steady * 1.5 + 1e-3 {
                failures.push(format!("switch {i}: click, step {step} vs steady {steady}"));
            }
        }
        let want = |tier: Tier| Some(crate::model_for(tier));
        assert_eq!(models[0], want(Tier::Standard), "starting model");
        for (i, &(tier, _)) in switches.iter().enumerate() {
            assert_eq!(models[i + 1], want(tier), "model after switch {i}");
        }
        assert_eq!(
            (s1.underruns, s1.resyncs, s1.overflows),
            (s0.underruns, s0.resyncs, s0.overflows),
            "glitches during the switches"
        );
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
