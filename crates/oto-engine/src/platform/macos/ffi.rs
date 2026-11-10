//! The small slice of Core Audio (AudioHardware.h) and CoreFoundation the backend needs.
//! Hand-written to keep dependencies light; layouts match the C headers on all Apple targets.

#![allow(non_upper_case_globals, non_snake_case, dead_code)]

use std::ffi::c_void;

pub type OSStatus = i32;
pub type AudioObjectID = u32;
pub type CFStringRef = *const c_void;

pub const fn fourcc(s: &[u8; 4]) -> u32 {
    ((s[0] as u32) << 24) | ((s[1] as u32) << 16) | ((s[2] as u32) << 8) | (s[3] as u32)
}

pub const kAudioObjectSystemObject: AudioObjectID = 1;
pub const kAudioObjectUnknown: AudioObjectID = 0;

pub const kAudioObjectPropertyScopeGlobal: u32 = fourcc(b"glob");
pub const kAudioObjectPropertyScopeInput: u32 = fourcc(b"inpt");
pub const kAudioObjectPropertyScopeOutput: u32 = fourcc(b"outp");
pub const kAudioObjectPropertyElementMain: u32 = 0;

pub const kAudioObjectPropertyName: u32 = fourcc(b"lnam");
pub const kAudioObjectPropertyManufacturer: u32 = fourcc(b"lmak");
pub const kAudioHardwarePropertyDevices: u32 = fourcc(b"dev#");
pub const kAudioHardwarePropertyDefaultInputDevice: u32 = fourcc(b"dIn ");
pub const kAudioHardwarePropertyTranslateUIDToDevice: u32 = fourcc(b"uidd");
pub const kAudioHardwarePropertyRunLoop: u32 = fourcc(b"rnlp");
pub const kAudioDevicePropertyDeviceUID: u32 = fourcc(b"uid ");
pub const kAudioDevicePropertyTransportType: u32 = fourcc(b"tran");
pub const kAudioDevicePropertyStreamConfiguration: u32 = fourcc(b"slay");
pub const kAudioDevicePropertyNominalSampleRate: u32 = fourcc(b"nsrt");
pub const kAudioDevicePropertyBufferFrameSize: u32 = fourcc(b"fsiz");
pub const kAudioDevicePropertyBufferFrameSizeRange: u32 = fourcc(b"fsz#");
pub const kAudioDevicePropertyDeviceIsRunningSomewhere: u32 = fourcc(b"gone");
pub const kAudioDevicePropertyDeviceIsAlive: u32 = fourcc(b"livn");
pub const kAudioDevicePropertyLatency: u32 = fourcc(b"ltnc");
pub const kAudioDevicePropertySafetyOffset: u32 = fourcc(b"saft");
pub const kAudioDevicePropertyStreams: u32 = fourcc(b"stm#");
pub const kAudioStreamPropertyVirtualFormat: u32 = fourcc(b"sfmt");
pub const kAudioStreamPropertyLatency: u32 = fourcc(b"ltnc");
/// The `os_workgroup_t` of the device's IO thread (macOS 11+), retained for the caller.
pub const kAudioDevicePropertyIOThreadOSWorkgroup: u32 = fourcc(b"oswg");

pub const kAudioDeviceTransportTypeBuiltIn: u32 = fourcc(b"bltn");
pub const kAudioDeviceTransportTypeAggregate: u32 = fourcc(b"grup");

pub const kAudioFormatLinearPCM: u32 = fourcc(b"lpcm");
pub const kAudioFormatFlagIsFloat: u32 = 1 << 0;

pub const kAudioTimeStampHostTimeValid: u32 = 1 << 1;

pub const kCFStringEncodingUTF8: u32 = 0x0800_0100;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AudioObjectPropertyAddress {
    pub mSelector: u32,
    pub mScope: u32,
    pub mElement: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioValueRange {
    pub mMinimum: f64,
    pub mMaximum: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioStreamBasicDescription {
    pub mSampleRate: f64,
    pub mFormatID: u32,
    pub mFormatFlags: u32,
    pub mBytesPerPacket: u32,
    pub mFramesPerPacket: u32,
    pub mBytesPerFrame: u32,
    pub mChannelsPerFrame: u32,
    pub mBitsPerChannel: u32,
    pub mReserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AudioBuffer {
    pub mNumberChannels: u32,
    pub mDataByteSize: u32,
    pub mData: *mut c_void,
}

#[repr(C)]
pub struct AudioBufferList {
    pub mNumberBuffers: u32,
    pub mBuffers: [AudioBuffer; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SMPTETime {
    pub mSubframes: i16,
    pub mSubframeDivisor: i16,
    pub mCounter: u32,
    pub mType: u32,
    pub mFlags: u32,
    pub mHours: i16,
    pub mMinutes: i16,
    pub mSeconds: i16,
    pub mFrames: i16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct AudioTimeStamp {
    pub mSampleTime: f64,
    pub mHostTime: u64,
    pub mRateScalar: f64,
    pub mWordClockTime: u64,
    pub mSMPTETime: SMPTETime,
    pub mFlags: u32,
    pub mReserved: u32,
}

pub type AudioObjectPropertyListenerProc = unsafe extern "C" fn(
    AudioObjectID,
    u32,
    *const AudioObjectPropertyAddress,
    *mut c_void,
) -> OSStatus;

pub type AudioDeviceIOProc = unsafe extern "C" fn(
    AudioObjectID,
    *const AudioTimeStamp,
    *const AudioBufferList,
    *const AudioTimeStamp,
    *mut AudioBufferList,
    *const AudioTimeStamp,
    *mut c_void,
) -> OSStatus;

pub type AudioDeviceIOProcID = Option<AudioDeviceIOProc>;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MachTimebaseInfo {
    pub numer: u32,
    pub denom: u32,
}

#[link(name = "CoreAudio", kind = "framework")]
unsafe extern "C" {
    pub fn AudioObjectHasProperty(id: AudioObjectID, addr: *const AudioObjectPropertyAddress)
    -> u8;
    pub fn AudioObjectGetPropertyDataSize(
        id: AudioObjectID,
        addr: *const AudioObjectPropertyAddress,
        qualifier_size: u32,
        qualifier: *const c_void,
        size: *mut u32,
    ) -> OSStatus;
    pub fn AudioObjectGetPropertyData(
        id: AudioObjectID,
        addr: *const AudioObjectPropertyAddress,
        qualifier_size: u32,
        qualifier: *const c_void,
        size: *mut u32,
        data: *mut c_void,
    ) -> OSStatus;
    pub fn AudioObjectSetPropertyData(
        id: AudioObjectID,
        addr: *const AudioObjectPropertyAddress,
        qualifier_size: u32,
        qualifier: *const c_void,
        size: u32,
        data: *const c_void,
    ) -> OSStatus;
    pub fn AudioObjectAddPropertyListener(
        id: AudioObjectID,
        addr: *const AudioObjectPropertyAddress,
        listener: AudioObjectPropertyListenerProc,
        client_data: *mut c_void,
    ) -> OSStatus;
    pub fn AudioObjectRemovePropertyListener(
        id: AudioObjectID,
        addr: *const AudioObjectPropertyAddress,
        listener: AudioObjectPropertyListenerProc,
        client_data: *mut c_void,
    ) -> OSStatus;
    pub fn AudioDeviceCreateIOProcID(
        dev: AudioObjectID,
        proc_: AudioDeviceIOProc,
        client_data: *mut c_void,
        out_id: *mut AudioDeviceIOProcID,
    ) -> OSStatus;
    pub fn AudioDeviceDestroyIOProcID(dev: AudioObjectID, id: AudioDeviceIOProcID) -> OSStatus;
    pub fn AudioDeviceStart(dev: AudioObjectID, id: AudioDeviceIOProcID) -> OSStatus;
    pub fn AudioDeviceStop(dev: AudioObjectID, id: AudioDeviceIOProcID) -> OSStatus;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub fn CFStringCreateWithBytes(
        alloc: *const c_void,
        bytes: *const u8,
        len: isize,
        encoding: u32,
        external: u8,
    ) -> CFStringRef;
    pub fn CFStringGetLength(s: CFStringRef) -> isize;
    pub fn CFStringGetMaximumSizeForEncoding(len: isize, encoding: u32) -> isize;
    pub fn CFStringGetCString(s: CFStringRef, buf: *mut u8, size: isize, encoding: u32) -> u8;
    pub fn CFRelease(obj: *const c_void);
}

unsafe extern "C" {
    pub fn mach_absolute_time() -> u64;
    pub fn mach_timebase_info(info: *mut MachTimebaseInfo) -> i32;
    pub fn pthread_self() -> *mut c_void;
    pub fn pthread_mach_thread_np(thread: *mut c_void) -> u32;
    pub fn thread_policy_set(thread: u32, flavor: u32, info: *mut i32, count: u32) -> i32;
    pub fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
    pub fn os_workgroup_join(wg: *mut c_void, token: *mut OsWorkgroupJoinToken) -> i32;
    pub fn os_workgroup_leave(wg: *mut c_void, token: *mut OsWorkgroupJoinToken);
    pub fn os_release(object: *mut c_void);
}

/// `os_workgroup_join_token_s` (`os/workgroup_object.h`).
#[repr(C)]
pub struct OsWorkgroupJoinToken {
    pub sig: u32,
    pub opaque: [u8; 36],
}

/// `thread_policy.h`: a thread with a periodic deadline (what Core Audio's IO threads use).
pub const THREAD_TIME_CONSTRAINT_POLICY: u32 = 2;
/// `sys/qos.h`.
pub const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

/// `thread_time_constraint_policy_data_t`. Times are in mach absolute time units.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ThreadTimeConstraintPolicy {
    pub period: u32,
    pub computation: u32,
    pub constraint: u32,
    pub preemptible: i32,
}

/// `THREAD_TIME_CONSTRAINT_POLICY_COUNT`, in `integer_t`s.
pub const THREAD_TIME_CONSTRAINT_POLICY_COUNT: u32 =
    (std::mem::size_of::<ThreadTimeConstraintPolicy>() / 4) as u32;

/// Renders an OSStatus as its four-char code when printable, else as a number.
pub fn status_str(s: OSStatus) -> String {
    let b = (s as u32).to_be_bytes();
    if b.iter().all(|c| c.is_ascii_graphic() || *c == b' ') {
        format!("'{}'", String::from_utf8_lossy(&b))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_c() {
        use std::mem::size_of;
        assert_eq!(size_of::<AudioTimeStamp>(), 64);
        assert_eq!(size_of::<AudioStreamBasicDescription>(), 40);
        assert_eq!(size_of::<AudioBuffer>(), 16);
        assert_eq!(size_of::<AudioBufferList>(), 24);
        assert_eq!(THREAD_TIME_CONSTRAINT_POLICY_COUNT, 4);
        assert_eq!(size_of::<OsWorkgroupJoinToken>(), 40);
        assert_eq!(fourcc(b"glob"), 0x676c_6f62);
        assert_eq!(status_str(fourcc(b"!dev") as i32), "'!dev'");
    }
}
