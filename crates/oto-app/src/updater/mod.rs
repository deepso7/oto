//! In-app updates (docs/plans/0003-app-updates.md, Decisions 1–3 and 6).
//!
//! - [`keys`]: the pinned manifest URL and the Ed25519 keys update envelopes must be signed
//!   with.
//! - [`http`]: a tiny synchronous [`http::Transport`]; on macOS an ephemeral `NSURLSession`
//!   that validates every redirect hop and enforces byte caps while receiving.
//! - [`verify`]: checking for an update and downloading, extracting and verifying one.
//! - [`apply`]: install-mode classification, the staged bundle swap through a detached helper,
//!   the candidate's two-stage acknowledgment, and recovery.
//!
//! The UI (Phase 4: `state.rs`, `view.rs`, `tray.rs`, `main.rs`) drives these from worker
//! threads; nothing here touches the UI thread.

pub mod apply;
pub mod http;
pub mod keys;
pub mod verify;
