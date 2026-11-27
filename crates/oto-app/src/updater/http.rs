//! A tiny synchronous HTTPS client for the updater.
//!
//! On macOS this is an ephemeral `NSURLSession` (no cookies, no credentials, no cache), so it
//! honours the system's proxy settings, PAC files and trust roots, which apps launched from
//! Finder or a LaunchAgent don't get from `HTTPS_PROXY`. A delegate validates every redirect
//! hop (`https://` only, port 443, no credentials, at most 5 hops) before it is followed, and
//! enforces the byte cap on each chunk as it arrives, so an oversized or endless response is
//! cut off early. Requests carry `User-Agent: Oto/<version>` and nothing that identifies the
//! machine or user.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// Connect/request and whole-request timeout for `update.json`.
pub const MANIFEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Idle timeout while downloading the archive.
pub const ARCHIVE_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// Whole-download timeout for the archive.
pub const ARCHIVE_TIMEOUT: Duration = Duration::from_secs(300);

/// The error a cancelled transfer returns.
pub const CANCELLED: &str = "Cancelled.";

/// The `User-Agent` every request carries.
pub fn user_agent() -> String {
    format!("Oto/{}", env!("CARGO_PKG_VERSION"))
}

/// Synchronous HTTPS GETs. Implementations validate `url` and every redirect with
/// [`oto_sign::manifest::validate_redirect_url`].
pub trait Transport: Send + Sync {
    /// The body of `url`, at most `max_bytes`. Status other than 200 is an error.
    fn get_bytes(
        &self,
        url: &str,
        max_bytes: usize,
        cancel: &AtomicBool,
    ) -> Result<Vec<u8>, String>;

    /// Streams `url` into a new file `dest` (mode 0600, replaced if present). Fails, removing
    /// `dest`, unless exactly `expected_size` bytes arrive; never accepts more than `max_bytes`.
    /// `on_progress(received, expected_size)` is called from the calling thread.
    fn download_file(
        &self,
        url: &str,
        dest: &Path,
        expected_size: u64,
        max_bytes: u64,
        cancel: &AtomicBool,
        on_progress: &mut dyn FnMut(u64, u64),
    ) -> Result<(), String>;
}

/// The transport for this platform.
pub fn system_transport() -> Box<dyn Transport> {
    #[cfg(target_os = "macos")]
    {
        Box::new(UrlSessionTransport)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Box::new(UnsupportedTransport)
    }
}

/// Where the received bytes go.
enum Sink {
    Memory(Vec<u8>),
    File(std::fs::File),
}

/// Transfer bookkeeping shared by the delegate (on the session's queue) and the waiting caller.
struct Progress {
    sink: Sink,
    max_bytes: u64,
    received: u64,
    hops: usize,
    /// The first error; later ones are ignored.
    error: Option<String>,
    done: bool,
}

impl Progress {
    fn new(sink: Sink, max_bytes: u64) -> Self {
        Self {
            sink,
            max_bytes,
            received: 0,
            hops: 0,
            error: None,
            done: false,
        }
    }

    fn fail(&mut self, e: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(e.into());
        }
    }

    /// Redirect hop to `url`: true if it may be followed.
    fn redirect(&mut self, url: Option<&str>) -> bool {
        self.hops += 1;
        if self.hops > oto_sign::manifest::MAX_REDIRECTS {
            self.fail("Too many redirects.");
            return false;
        }
        match url.map(oto_sign::manifest::validate_redirect_url) {
            Some(Ok(())) => true,
            Some(Err(e)) => {
                self.fail(format!("Refused a redirect: {e}"));
                false
            }
            None => {
                self.fail("Refused a redirect without a URL.");
                false
            }
        }
    }

    /// Response headers: true to receive the body.
    fn response(&mut self, status: Option<i64>, expected_len: i64) -> bool {
        match status {
            Some(200) => {}
            Some(s) => {
                self.fail(format!("The server answered HTTP {s}."));
                return false;
            }
            None => {
                self.fail("Not an HTTP response.");
                return false;
            }
        }
        if expected_len > 0 && expected_len as u64 > self.max_bytes {
            self.fail("The download is larger than expected.");
            return false;
        }
        true
    }

    /// One chunk: false to cancel the transfer.
    fn data(&mut self, chunk: &[u8]) -> bool {
        if self.error.is_some() {
            return false;
        }
        let len = chunk.len() as u64;
        if self.received + len > self.max_bytes {
            self.fail("The download is larger than expected.");
            return false;
        }
        let res = match &mut self.sink {
            Sink::Memory(v) => {
                v.extend_from_slice(chunk);
                Ok(())
            }
            Sink::File(f) => std::io::Write::write_all(f, chunk),
        };
        if let Err(e) = res {
            self.fail(format!("Couldn't save the download: {e}"));
            return false;
        }
        self.received += len;
        true
    }

    fn complete(&mut self, error: Option<String>) {
        if let Some(e) = error {
            self.fail(e);
        }
        self.done = true;
    }
}

#[cfg(not(target_os = "macos"))]
pub struct UnsupportedTransport;

#[cfg(not(target_os = "macos"))]
impl Transport for UnsupportedTransport {
    fn get_bytes(&self, _: &str, _: usize, _: &AtomicBool) -> Result<Vec<u8>, String> {
        Err("Checking for updates isn't supported on this platform yet.".into())
    }
    fn download_file(
        &self,
        _: &str,
        _: &Path,
        _: u64,
        _: u64,
        _: &AtomicBool,
        _: &mut dyn FnMut(u64, u64),
    ) -> Result<(), String> {
        Err("Updating isn't supported on this platform yet.".into())
    }
}

#[cfg(target_os = "macos")]
pub use macos::UrlSessionTransport;

#[cfg(target_os = "macos")]
mod macos {
    use super::{CANCELLED, Progress, Sink, Transport, user_agent};
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    use block2::DynBlock;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2::{AnyThread, DefinedClass, define_class, msg_send};
    use objc2_foundation::{
        NSData, NSError, NSHTTPURLResponse, NSMutableURLRequest, NSObject, NSObjectProtocol,
        NSOperationQueue, NSString, NSURL, NSURLRequest, NSURLRequestCachePolicy, NSURLResponse,
        NSURLSession, NSURLSessionConfiguration, NSURLSessionDataDelegate, NSURLSessionDataTask,
        NSURLSessionDelegate, NSURLSessionResponseDisposition, NSURLSessionTask,
        NSURLSessionTaskDelegate,
    };

    struct Shared {
        state: Mutex<Progress>,
        cv: Condvar,
    }

    impl Shared {
        fn with<R>(&self, f: impl FnOnce(&mut Progress) -> R) -> R {
            let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let r = f(&mut g);
            self.cv.notify_all();
            r
        }
    }

    struct Ivars {
        shared: Arc<Shared>,
    }

    define_class!(
        // SAFETY: NSObject has no subclassing requirements and `Delegate` has no Drop impl.
        #[unsafe(super(NSObject))]
        #[name = "OtoUpdateURLSessionDelegate"]
        #[ivars = Ivars]
        struct Delegate;

        unsafe impl NSObjectProtocol for Delegate {}

        unsafe impl NSURLSessionDelegate for Delegate {}

        unsafe impl NSURLSessionTaskDelegate for Delegate {
            #[unsafe(method(URLSession:task:willPerformHTTPRedirection:newRequest:completionHandler:))]
            unsafe fn will_redirect(
                &self,
                _session: &NSURLSession,
                _task: &NSURLSessionTask,
                _response: &NSHTTPURLResponse,
                request: &NSURLRequest,
                completion: &DynBlock<dyn Fn(*mut NSURLRequest)>,
            ) {
                let url = request
                    .URL()
                    .and_then(|u| u.absoluteString())
                    .map(|s| s.to_string());
                let follow = self.ivars().shared.with(|p| p.redirect(url.as_deref()));
                let next: *mut NSURLRequest = if follow {
                    (request as *const NSURLRequest).cast_mut()
                } else {
                    std::ptr::null_mut()
                };
                completion.call((next,));
            }

            #[unsafe(method(URLSession:task:didCompleteWithError:))]
            fn did_complete(
                &self,
                _session: &NSURLSession,
                _task: &NSURLSessionTask,
                error: Option<&NSError>,
            ) {
                let e = error.map(|e| e.localizedDescription().to_string());
                self.ivars().shared.with(|p| p.complete(e));
            }
        }

        unsafe impl NSURLSessionDataDelegate for Delegate {
            #[unsafe(method(URLSession:dataTask:didReceiveResponse:completionHandler:))]
            unsafe fn did_receive_response(
                &self,
                _session: &NSURLSession,
                _task: &NSURLSessionDataTask,
                response: &NSURLResponse,
                completion: &DynBlock<dyn Fn(NSURLSessionResponseDisposition)>,
            ) {
                let status = response
                    .downcast_ref::<NSHTTPURLResponse>()
                    .map(|r| r.statusCode() as i64);
                let len = response.expectedContentLength();
                let ok = self.ivars().shared.with(|p| p.response(status, len));
                completion.call((if ok {
                    NSURLSessionResponseDisposition::Allow
                } else {
                    NSURLSessionResponseDisposition::Cancel
                },));
            }

            #[unsafe(method(URLSession:dataTask:didReceiveData:))]
            fn did_receive_data(
                &self,
                _session: &NSURLSession,
                task: &NSURLSessionDataTask,
                data: &NSData,
            ) {
                let chunk = data.to_vec();
                if !self.ivars().shared.with(|p| p.data(&chunk)) {
                    task.cancel();
                }
            }
        }
    );

    // SAFETY: the only state is `Arc<Shared>`, behind a mutex.
    unsafe impl Send for Delegate {}
    unsafe impl Sync for Delegate {}

    impl Delegate {
        fn new(shared: Arc<Shared>) -> Retained<Self> {
            let this = Self::alloc().set_ivars(Ivars { shared });
            // SAFETY: NSObject's designated initializer.
            unsafe { msg_send![super(this), init] }
        }
    }

    /// `NSURLSession` with an ephemeral configuration and no cookie, credential or cache
    /// storage. One session per transfer, invalidated when it ends.
    pub struct UrlSessionTransport;

    impl UrlSessionTransport {
        #[allow(clippy::too_many_arguments)]
        fn run(
            &self,
            url: &str,
            sink: Sink,
            max_bytes: u64,
            idle: Duration,
            total: Duration,
            cancel: &AtomicBool,
            on_progress: &mut dyn FnMut(u64),
        ) -> Result<Progress, String> {
            oto_sign::manifest::validate_redirect_url(url)?;
            let ns_url = NSURL::URLWithString(&NSString::from_str(url))
                .ok_or_else(|| format!("Invalid URL {url}"))?;

            let config = NSURLSessionConfiguration::ephemeralSessionConfiguration();
            config.setHTTPCookieStorage(None);
            config.setURLCredentialStorage(None);
            config.setURLCache(None);
            config.setHTTPShouldSetCookies(false);
            config.setRequestCachePolicy(NSURLRequestCachePolicy::ReloadIgnoringLocalCacheData);
            config.setTimeoutIntervalForRequest(idle.as_secs_f64());
            config.setTimeoutIntervalForResource(total.as_secs_f64());

            let request = NSMutableURLRequest::requestWithURL(&ns_url);
            request.setValue_forHTTPHeaderField(
                Some(&NSString::from_str(&user_agent())),
                &NSString::from_str("User-Agent"),
            );
            request.setHTTPShouldHandleCookies(false);
            request.setCachePolicy(NSURLRequestCachePolicy::ReloadIgnoringLocalCacheData);
            request.setTimeoutInterval(idle.as_secs_f64());

            let shared = Arc::new(Shared {
                state: Mutex::new(Progress::new(sink, max_bytes)),
                cv: Condvar::new(),
            });
            let delegate = Delegate::new(shared.clone());
            let queue = NSOperationQueue::new();
            queue.setMaxConcurrentOperationCount(1);
            // SAFETY: the queue is a private serial queue; the delegate is thread-safe.
            let session = unsafe {
                NSURLSession::sessionWithConfiguration_delegate_delegateQueue(
                    &config,
                    Some(ProtocolObject::from_ref(&*delegate)),
                    Some(&queue),
                )
            };
            let task = session.dataTaskWithRequest(&request);
            task.resume();

            // A hard stop a little after the session's own resource timeout.
            let deadline = Instant::now() + total + Duration::from_secs(10);
            let mut cancelled = false;
            let mut g = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if g.done {
                    break;
                }
                if !cancelled && (cancel.load(Ordering::Relaxed) || Instant::now() >= deadline) {
                    cancelled = true;
                    g.fail(if cancel.load(Ordering::Relaxed) {
                        CANCELLED
                    } else {
                        "The download timed out."
                    });
                    task.cancel();
                }
                let received = g.received;
                drop(g);
                on_progress(received);
                g = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                if g.done {
                    break;
                }
                g = shared
                    .cv
                    .wait_timeout(g, Duration::from_millis(100))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
            drop(g);
            // Releases the session's strong reference to the delegate.
            session.finishTasksAndInvalidate();
            let mut progress = {
                let mut g = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                std::mem::replace(&mut *g, Progress::new(Sink::Memory(Vec::new()), 0))
            };
            drop(delegate);
            on_progress(progress.received);
            if let Some(e) = progress.error.take() {
                return Err(e);
            }
            Ok(progress)
        }
    }

    impl Transport for UrlSessionTransport {
        fn get_bytes(
            &self,
            url: &str,
            max_bytes: usize,
            cancel: &AtomicBool,
        ) -> Result<Vec<u8>, String> {
            let p = self.run(
                url,
                Sink::Memory(Vec::new()),
                max_bytes as u64,
                super::MANIFEST_TIMEOUT,
                super::MANIFEST_TIMEOUT,
                cancel,
                &mut |_| {},
            )?;
            match p.sink {
                Sink::Memory(v) => Ok(v),
                Sink::File(_) => unreachable!(),
            }
        }

        fn download_file(
            &self,
            url: &str,
            dest: &Path,
            expected_size: u64,
            max_bytes: u64,
            cancel: &AtomicBool,
            on_progress: &mut dyn FnMut(u64, u64),
        ) -> Result<(), String> {
            let file = super::create_private_file(dest)?;
            let res = self
                .run(
                    url,
                    Sink::File(file),
                    max_bytes.min(expected_size),
                    super::ARCHIVE_IDLE_TIMEOUT,
                    super::ARCHIVE_TIMEOUT,
                    cancel,
                    &mut |n| on_progress(n, expected_size),
                )
                .and_then(|p| super::finish_file(p.sink, p.received, expected_size));
            if res.is_err() {
                let _ = std::fs::remove_file(dest);
            }
            res
        }
    }
}

/// A new file at `path` (any old one replaced), mode 0600.
fn create_private_file(path: &Path) -> Result<std::fs::File, String> {
    let _ = std::fs::remove_file(path);
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path)
        .map_err(|e| format!("Couldn't create {}: {e}", path.display()))
}

/// Checks the byte count and flushes the file to disk.
fn finish_file(sink: Sink, received: u64, expected: u64) -> Result<(), String> {
    if received != expected {
        return Err(format!(
            "The download is incomplete ({received} of {expected} bytes)."
        ));
    }
    if let Sink::File(f) = sink {
        f.sync_all()
            .map_err(|e| format!("Couldn't save the download: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;

    /// Serves fixed bodies by URL, enforcing the same caps as the real transport.
    #[derive(Default)]
    pub(crate) struct FakeTransport {
        pub bodies: Mutex<HashMap<String, Vec<u8>>>,
        pub requests: Mutex<Vec<String>>,
    }

    impl FakeTransport {
        pub fn serve(&self, url: &str, body: Vec<u8>) {
            self.bodies.lock().unwrap().insert(url.into(), body);
        }
        fn body(&self, url: &str) -> Result<Vec<u8>, String> {
            oto_sign::manifest::validate_redirect_url(url)?;
            self.requests.lock().unwrap().push(url.into());
            self.bodies
                .lock()
                .unwrap()
                .get(url)
                .cloned()
                .ok_or_else(|| "The server answered HTTP 404.".into())
        }
    }

    impl Transport for FakeTransport {
        fn get_bytes(&self, url: &str, max: usize, cancel: &AtomicBool) -> Result<Vec<u8>, String> {
            if cancel.load(Ordering::Relaxed) {
                return Err(CANCELLED.into());
            }
            let mut p = Progress::new(Sink::Memory(Vec::new()), max as u64);
            if !p.data(&self.body(url)?) {
                return Err(p.error.unwrap());
            }
            match p.sink {
                Sink::Memory(v) => Ok(v),
                Sink::File(_) => unreachable!(),
            }
        }
        fn download_file(
            &self,
            url: &str,
            dest: &Path,
            expected: u64,
            max: u64,
            cancel: &AtomicBool,
            on_progress: &mut dyn FnMut(u64, u64),
        ) -> Result<(), String> {
            let body = self.body(url)?;
            let mut p = Progress::new(Sink::File(create_private_file(dest)?), max.min(expected));
            let res = (|| {
                for chunk in body.chunks(1000) {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(CANCELLED.to_string());
                    }
                    if !p.data(chunk) {
                        return Err(p.error.clone().unwrap());
                    }
                    on_progress(p.received, expected);
                }
                let received = p.received;
                finish_file(
                    std::mem::replace(&mut p.sink, Sink::Memory(vec![])),
                    received,
                    expected,
                )
            })();
            if res.is_err() {
                let _ = std::fs::remove_file(dest);
            }
            res
        }
    }

    #[test]
    fn redirects_are_validated_and_bounded() {
        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        assert!(p.redirect(Some("https://release-assets.githubusercontent.com/x?sig=1")));
        assert!(!p.redirect(Some("http://example.com/x")));
        assert!(p.error.as_deref().unwrap().contains("Refused a redirect"));

        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        assert!(!p.redirect(Some("https://user:pw@example.com/")));
        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        assert!(!p.redirect(Some("https://example.com:8443/")));
        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        assert!(!p.redirect(None));

        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        for _ in 0..oto_sign::manifest::MAX_REDIRECTS {
            assert!(p.redirect(Some("https://example.com/")));
        }
        assert!(!p.redirect(Some("https://example.com/")));
        assert_eq!(p.error.as_deref(), Some("Too many redirects."));
    }

    #[test]
    fn the_byte_cap_applies_per_chunk_and_to_the_announced_length() {
        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        assert!(p.response(Some(200), 10));
        assert!(p.data(b"12345"));
        assert!(p.data(b"67890"));
        assert!(!p.data(b"x"));
        assert_eq!(p.received, 10);
        assert!(p.error.as_deref().unwrap().contains("larger"));
        // After an error nothing more is accepted.
        assert!(!p.data(b""));

        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        assert!(!p.response(Some(200), 11));
        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        assert!(
            p.response(Some(200), -1),
            "unknown length is capped while receiving"
        );
        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        assert!(!p.response(Some(404), 3));
        assert_eq!(p.error.as_deref(), Some("The server answered HTTP 404."));
        let mut p = Progress::new(Sink::Memory(vec![]), 10);
        assert!(!p.response(None, 3));
    }

    #[test]
    fn completion_keeps_the_first_error() {
        let mut p = Progress::new(Sink::Memory(vec![]), 1);
        p.fail("first");
        p.complete(Some("cancelled".into()));
        assert!(p.done);
        assert_eq!(p.error.as_deref(), Some("first"));
    }

    #[test]
    fn files_must_be_complete_and_private() {
        let dir = std::env::temp_dir().join(format!("oto-http-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.part");
        std::fs::write(&path, "old").unwrap();
        let f = create_private_file(&path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        assert!(finish_file(Sink::File(f), 3, 4).is_err());
        assert!(finish_file(Sink::Memory(vec![]), 4, 4).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_user_agent_says_only_the_version() {
        assert_eq!(user_agent(), format!("Oto/{}", env!("CARGO_PKG_VERSION")));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn the_real_transport_refuses_insecure_urls_before_connecting() {
        let t = UrlSessionTransport;
        let no = AtomicBool::new(false);
        for url in ["http://github.com/x", "https://a@github.com/x", "ftp://x/y"] {
            assert!(t.get_bytes(url, 10, &no).is_err(), "{url}");
        }
    }
}
