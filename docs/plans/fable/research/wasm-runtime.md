Raw research notes, 2026-11-27; claims marked UNVERIFIED were not confirmed from a primary source

# WASM / AudioWorklet runtime research: hand-written Rust inference vs existing runtimes

Scope: is it sensible to run a small causal 48 kHz speech-enhancement net (~1-1.5M params, conv + GRU,
STFT 1024 / hop 512, claimed 50-100 MMAC/s) in hand-written pure Rust with SIMD kernels, natively and as
`wasm32-unknown-unknown +simd128` inside a single-threaded AudioWorkletProcessor (no SAB, no COOP/COEP),
within 3 ms per 10.67 ms hop and 3 MB download.

Method: web search and page fetches on 2026-11-27. Most pages were read through a summarising fetch tool, so
numbers are as that tool reported them; items I could not open directly are marked UNVERIFIED. Lines marked
"(my analysis)" are reasoning, not sourced facts.

---

## 0. Short answer

- Hand-written kernels are the norm, not the exception, for small-footprint browser denoisers. Every shipping
  example under ~3 MB is hand-written (RNNoise wasm, fastenhancer-web, DeepFilterNet3.c.wasm, Alkove's engine).
  Every generic-runtime example is 9-33 MB.
- The compute budget is not the hard part. ~1 M MAC per hop is roughly 3x less work than DeepFilterNet3 per
  frame, which tract-in-wasm runs in 0.33 ms (Apple Silicon) and 0.72 ms mean / 1.5 ms p95 (2020 i7-10750H, Chrome).
- The hard parts are the browser plumbing (128-frame quantum, deadline per quantum not per hop, sample rate on
  Safari/iOS, Android capture bugs), tail latency under paced execution, denormals, and int8 quality on the GRU.
- The "~21 ms in browser" figure is only the STFT algorithmic delay. Real added delay is higher (see 1.5).
- "Bit-for-bit to <= -60 dB RMS" is self-contradictory and, for f32, far too loose; for int8 it is unreachable
  (published int8 ports sit at about -38 to -53 dB against their fp32 reference). See section 7.

---

## 1. Browser AudioWorklet realities (late 2026)

### 1.1 Render quantum

- Fixed at 128 frames everywhere by default: 2.67 ms at 48 kHz.
- Chrome: `renderSizeHint` constructor option and `renderQuantumSize` attribute ship in Chrome 153 according to
  an MDN browser-compat-data PR dated 2026-10-22. Intent to Ship on blink-dev was 2026-07-23, with three LGTMs
  by 2026-07-29, after an origin trial that was extended (to M150, then a request to M152) because of
  implementation bugs and fingerprinting concerns.
  - Values: an integer, `"default"` (128), or `"hardware"`.
  - Caveat from the Intent to Ship: Chromium's `"hardware"` mode currently still gives 128. The spec says it is
    "a hint that might not be honored".
  - With `"hardware"`, `renderQuantumSize` reads 0 until the context is running (fingerprinting safeguard).
  - https://github.com/mdn/browser-compat-data/pull/30947
  - http://www.mail-archive.com/blink-dev@chromium.org/msg17051.html
  - https://chromestatus.com/feature/5078190552907776
  - https://developer.mozilla.org/en-US/docs/Web/API/AudioContext/AudioContext
- Firefox: standards position "positive" with stated intent to implement (padenot, 2026-11-05). Tracking bug
  2034508. Not shipped.
  - https://github.com/mozilla/standards-positions/issues/1407
  - https://bugzilla.mozilla.org/show_bug.cgi?id=2034508
- Safari/WebKit: "No signal". Standards-position issue 662 has no linked implementation bug.
  - https://github.com/WebKit/standards-positions/issues/662
- Consequence: design for 128 with a ring buffer. Hop 512 = exactly 4 quanta at 48 kHz, so there is no drift
  and no fractional reframing (unlike RNNoise's 480, which costs Jitsi 640 frames = 13.3 ms).
  - A `renderSizeHint: 512` on Chrome would be a nice optional optimisation, never a requirement.

### 1.2 What the device actually does (bursts)

- A live context renders each device callback as a burst of quanta back to back. Observed in the anira web
  runtime work: headless Chromium 2048-frame callback = 16 quanta; a 44.1 kHz context on a 48 kHz device with
  2048-frame callbacks gives runs of 14 and 15 quanta.
- `baseLatency` is not portable: Chromium reports max(callback, quantum)/rate; Safari reports 128/rate;
  Firefox reports 0. `renderQuantumSize` is not exposed by Firefox or Safari.
  - https://github.com/anira-project/anira/pull/213
- Android Chrome: device bursts of 96 or 192 frames clash with the 128 quantum and cause glitches unrelated to
  DSP cost; one Samsung A10 test saw Chrome choose a 3828-sample (80 ms) buffer.
  - https://issues.chromium.org/issues/40133762
  - https://issues.chromium.org/issues/40103372
  - https://kanejaku.org/posts/2021/02/reducing-audio-glitches-on-chrome-for-android/ (process() under 1 ms yet still glitching on a Pixel 4a)
- (my analysis) The real deadline is "all quanta of this device callback finish before the callback period
  ends". With a 10 ms device buffer, one 3 ms quantum plus three cheap ones is fine. With a 128-frame device
  buffer, a 3 ms hop frame overruns a 2.67 ms slot. So "3 ms per hop" is not safe as stated; the budget that
  matters is "well under 2.67 ms at p99 on the slowest supported machine", or the inference has to be split
  across the 4 quanta of a hop.

### 1.3 Overruns, GC, priority

- Overrun: there is no exception. Late output means an underrun (click or silence), and callbacks pile up.
  A ring buffer bridges block sizes but buys no time.
  - https://developer.chrome.com/blog/audio-worklet-design-pattern
  - https://github.com/GoogleChrome/developer.chrome.com/issues/3355
- GC: the worklet thread has its own JS heap. Allocating in `process()` eventually triggers a collection that
  blocks the thread. Chrome's guidance is to diagnose with `about://tracing`.
  - https://web.dev/profiling-web-audio-apps-in-chrome/
- Chrome priority: originally DISPLAY priority, later an opt-in flag (~Chrome 87), and by build 96.0.4652.0 an
  engineer verified `REALTIME_AUDIO` for the worklet thread.
  - Exception: worklets in iframes are denied the realtime thread, with sizable delays reported.
  - https://issues.chromium.org/issues/40195691
  - https://github.com/WebAudio/web-audio-api/issues/2269
  - The web.dev profiling article still says "no real-time priority"; that text is stale.
- Firefox: worklet runs on a dedicated real-time system thread; code that allocates nothing never triggers GC there.
  - https://hacks.mozilla.org/2020/05/high-performance-web-audio-with-audioworklet-in-firefox/
- Safari: no primary source found on thread priority. UNVERIFIED.
- V8 wasm tiering: functions are compiled lazily by Liftoff on first call (blocking that call), then promoted
  to the optimising tier in the background; there is no on-stack replacement for wasm. Liftoff code runs
  roughly 1.5x slower. Opening DevTools drops code back to Liftoff.
  - https://v8.dev/docs/wasm-compilation-pipeline
  - A project reported first-play dropouts from untiered code and added a warm-up per AudioWorkletGlobalScope:
    https://github.com/tacowars/windsor/issues/837
  - (my analysis) Run a few hundred dummy frames at init, before the node is connected.
- `performance.now()` is missing in AudioWorkletGlobalScope (Chromium, WebKit); only `Date.now()` with 1 ms
  steps. In-worklet per-frame timing is therefore coarse.
  - https://github.com/workadventure/noise-suppression/pull/19

### 1.4 Sample rate

- `new AudioContext({ sampleRate: 48000 })` makes the browser resample device I/O. It is the usual approach
  (RNNoise-based packages require a 48 kHz context; one PoC simply disables RNNoise at other rates).
- Safari/iOS is the problem area:
  - WebKit bug 274507 (iPadOS 17.5 regression): a 44100 Hz context plus any AudioWorklet distorts, whatever the
    worklet computes. https://bugs.webkit.org/show_bug.cgi?id=274507
  - The hardware rate can change under a live context when the audio session flips to play-and-record. One
    project's fix: call getUserMedia first, then construct the context, and log the resulting rate.
    https://github.com/0xMaxMa/claude-gateway/issues/606 (also cites WebKit bug 221334, getUserMedia + WebAudio glitching, status NEW)
  - Safari's reported `sampleRate` is not guaranteed stable or truthful:
    https://github.com/chrisguttandin/standardized-audio-context/issues/489 and https://bugs.webkit.org/show_bug.cgi?id=154538
- (my analysis) Read `sampleRate` after the mic is live. If it is not 48000, either bypass or resample inside
  the wasm. Bluetooth headset mics (HFP) deliver 8/16 kHz content upsampled, which the model will see as
  band-limited speech.
- Android Chrome capture bug: with echoCancellation / noiseSuppression / autoGainControl enabled, the worklet
  intermittently receives all-zero input on Android 14+. Workaround is to disable all three, which also
  removes echo cancellation. https://issues.chromium.org/issues/502356103

### 1.5 End-to-end latency (the "~21 ms" claim)

- (my analysis) STFT 1024 / hop 512 at 48 kHz has an algorithmic delay of one window: 1024 samples = 21.33 ms
  (18.7 ms if the frame's output is emitted in the same quantum that completes it). That is all the "~21 ms"
  covers.
- Not included:
  - Capture path (ADC, OS buffer, browser capture buffering).
  - The extra AudioContext hop itself. WorkAdventure measured roughly 21 ms for the extra AudioContext on top
    of model + reframing delay (their DFN3 chain: 30 ms model + 10.7 ms reframing + 30 ms gate = 71 ms, about
    91 ms with the context). https://github.com/workadventure/noise-suppression/pull/19
  - Another project measured RNNoise in a worklet at 21 ms added delay (algorithmic 13.3 ms) and DeepFilterNet3
    at 39 ms, on an M4 Pro. https://github.com/nickolaiposs/friendspeak/pull/77
  - `MediaStreamAudioDestinationNode`: the spec allows internal buffering; a WebKit bug comment warns of
    "additional buffering/latency". No per-browser millisecond figure found. UNVERIFIED magnitude.
    https://bugs.webkit.org/show_bug.cgi?format=multiple&id=231105
  - WebRTC itself (10 ms framing, Opus packetisation, jitter buffer) is there with or without the denoiser.
- Round-trip mic to speaker through a worklet, for scale: about 67 ms Chrome / 55 ms Firefox with default
  getUserMedia settings on a 2016 MacBook Pro, 19 / 14 ms with processing off (old measurement).
  https://www.jefftk.com/p/browser-audio-latency
  - Crowd-sourced table shows wide spreads, e.g. Safari 26.2 at 48 kHz / 128 frames: 69-75 ms.
    https://superpowered.com/webbrowserlatency
- (my analysis) A realistic "added by the denoiser path" figure on desktop is 30-45 ms, not 21 ms. Mobile and
  Bluetooth are worse. UNVERIFIED as a number; it needs a loopback measurement per browser.

### 1.6 Worklet or Worker + SharedArrayBuffer?

- In-worklet inference is what the open implementations do: Jitsi RNNoise, `@sapphi-red/web-noise-suppressor`,
  WorkAdventure (DTLN via LiteRT.js, DFN3 via tract), DeepFilterNet3.c.wasm, fastenhancer-web, Datadog dtln-rs.
- Krisp does the opposite: `createNoiseFilter` returns an AudioWorkletNode but inference runs in a dedicated
  Worker; `useSharedArrayBuffer` is optional (off by default) and lowers CPU; frames are dropped once the
  buffer reaches 500 ms. https://developers-docs.krisp.ai/docs/getting-started-js
  - Stream's Krisp upgrade notes a failure mode: with SAB on, a worker failure leaves the filter emitting
    silence. https://github.com/GetStream/stream-video-js/pull/2537
- WorkAdventure considered worklet + Worker + SAB "probably the best long-term architecture" for apps that can
  enable COOP/COEP, and rejected it because COEP is recursive across iframes.
  https://workadventu.re/tech/building-an-easy-to-use-browser-noise-suppression-library-in-an-audio-worklet/
- Zoom's web SDK only needs cross-origin isolation for its WebAssembly audio path; the WebRTC audio path does
  not. https://github.com/zoom/zoom-developers-plugin-codex/blob/main/skills/meeting-sdk/web/concepts/sharedarraybuffer.md
- `MediaStreamTrackProcessor` for audio is Chromium-only in late 2026 (Safari 18+ has video only, in workers;
  Firefox has not shipped). Shiguredo's `@shiguredo/noise-suppression` uses it and is therefore Chromium-only.
  https://blog.mozilla.org/webrtc/unbundling-mediastreamtrackprocessor-and-videotrackgenerator/
  https://bugzilla.mozilla.org/show_bug.cgi?id=1749532
- (my analysis) For a model that costs well under 1 ms per hop, in-worklet is correct. A Worker only pays off
  for models that cannot meet the quantum deadline, and then it needs SAB to avoid postMessage jitter.

### 1.7 Shipping products: size and CPU

| Product | Where inference runs | Download | CPU / timing | Source |
|---|---|---|---|---|
| Jitsi RNNoise (`@jitsi/rnnoise-wasm`) | worklet, sync-compiled inlined wasm | 1.8 MB Emscripten build (as reported by a downstream issue) | not stated | https://github.com/jitsi/rnnoise-wasm , https://github.com/SoaresDavidson/Vcall/issues/5 |
| `@sapphi-red/web-noise-suppressor` 0.4.x | worklet | rnnoise.wasm ~150 kB, simd ~155 kB, worklet ~64 kB (113 kB / 12 kB compressed) | ~0.95 % of a core with SIMD, ~1.6 % without (M1 Max) | https://github.com/enzolaOps/vortex/pull/250 , https://github.com/rafaelcg/pqp/pull/517 |
| Krisp Browser SDK | Worker (+ optional SAB) | 12 MB package, ~100 MB memory | 1.5-2 ms per 10 ms frame; CPU tables are images only | https://developers-docs.krisp.ai/docs/introduction |
| LiveKit `@livekit/krisp-noise-filter` | Krisp SDK | Krisp | Safari >= 17.4; LiveKit Cloud only | https://docs.livekit.io/transport/media/noise-cancellation/ |
| Daily | Krisp | Krisp | launch (2023) excluded desktop Safari and mobile web | https://www.daily.co/blog/daily-partners-with-krisp-to-provide-world-class-noise-cancellation-capabilities/ |
| WorkAdventure DTLN (LiteRT.js, 16 kHz) | worklet | ~4 MB | 4.3-6.0 ms per 32 ms frame, p95 9 ms (i7-10750H) | https://workadventu.re/tech/building-an-easy-to-use-browser-noise-suppression-library-in-an-audio-worklet/ |
| WorkAdventure DFN3 (libDF + tract 0.23.8) | worklet | wasm 14.6 MB + 8 MB model | 0.33 ms mean per 10 ms frame (Apple Silicon); 0.72 ms mean / 1.5 ms p95 (i7-10750H, Chrome 154) | https://github.com/workadventure/noise-suppression/pull/19 |
| `@lofcz/deepfilternet-web` | worklet | 32.6 MB (weights inside wasm) | not stated | https://npm.io/package/@lofcz/deepfilternet-web |
| DeepFilterNet3.c.wasm (hand-written C) | worklet, single-file base64 | weights 8.2 MB (2.1 M params, f32); wasm size not stated | ~1 ms M2, ~4 ms Galaxy S23, ~6 ms Note10+ per 10 ms frame | https://github.com/kdrkdrkdr/DeepFilterNet3.c.wasm |
| fastenhancer-web (hand-written C) | worklet, base64 bundle, no COOP/COEP | not found | not found | https://github.com/ryyr-ry/fastenhancer-web/ |
| Alkove (Rust engine, 16 kHz) | wasm, single thread | engine 185 KB brotli; Focus model 2.3 MB | not found | https://github.com/alkove-ai/alkove |
| ai-coustics `aic-sdk-wasm` 0.26 | wasm, needs SIMD | models 5.3-36.7 MB | no browser CPU figures | https://docs.ai-coustics.com/reference/sdk/models |
| Picovoice Koala web | Worker (`KoalaWorker`) | not found | RTF 0.0126 (platform unstated) | https://picovoice.ai/products/voice/noise-suppression/ |
| Datadog dtln-rs (TFLite via Rust + Emscripten) | worklet | not stated | 33 ms per 1 s of audio, native, M1 | https://www.datadoghq.com/blog/engineering/noise-suppression-library/ |
| Google Meet | cloud denoiser on desktop web (2020 reporting; current behaviour UNVERIFIED) | n/a | n/a | https://venturebeat.com/2020/06/08/google-meet-noise-cancellation-ai-cloud-denoiser-g-suite/ |
| Discord web | Krisp, on-device; that the web client uses the wasm SDK is an inference, UNVERIFIED | | | |
| Twilio | Krisp audio processor integration; no size/CPU figures collected | | | https://www.twilio.com/en-us/blog/introducing-audio-processor-apis |

The fastenhancer-web README could not be fetched (404 / rate limit); what is known comes from search snippets:
C implementation of FastEnhancer, relaxed-simd FMA build plus scalar fallback, weights and wasm as base64
inside the JS bundle, a single up-front allocation, Playwright tests on Chromium/Firefox/WebKit, npm 0.1.1, MIT.

---

## 2. Loading WASM in an AudioWorklet from Rust

### 2.1 Constraints

- AudioWorkletGlobalScope has no `fetch`, no `importScripts`, no dynamic `import()`, no `document`, and (the
  point that bites wasm-bindgen) no `TextDecoder` / `TextEncoder`.
  - The wasm-bindgen glue constructs a `TextDecoder` (or `TextEncoder`) at module evaluation, so the worklet
    script dies before `registerProcessor` runs. Upstream tracking issue: wasm-bindgen #2367.
  - Still true in 2026: a September 2026 PR on Chromium 151 fixed it with a UTF-8 decoder prelude, and
    WorkAdventure's October 2026 PR ships a `TextEncoder` shim. "Without it the processor never registers."
    https://github.com/AnthonE/Gates/pull/152 , https://github.com/workadventure/noise-suppression/pull/19
  - Testing trap: Node has these globals, so a Node-based worklet test passes falsely.
- `getrandom` is pulled in by some dependencies and needs `crypto.getRandomValues`, also absent (libDF hit this).
- The official wasm-bindgen `wasm-audio-worklet` example is built around threads: nightly, `-Zbuild-std`,
  `+atomics,+bulk-memory,+mutable-globals`, and COOP/COEP headers for SharedArrayBuffer. It is not a template
  for the no-SAB case.
  - https://wasm-bindgen.github.io/wasm-bindgen/examples/wasm-audio-worklet.html
  - https://github.com/wasm-bindgen/wasm-bindgen/discussions/3928

### 2.2 Getting the module into the worklet

- Fetch on the main thread, then either:
  - compile on the main thread and pass the `WebAssembly.Module` through `processorOptions`, then
    `new WebAssembly.Instance(module, imports)` synchronously in the processor constructor; or
  - transfer the raw bytes and compile synchronously in the worklet.
- Chromium issue 40855462: a Module arrived via `processorOptions` but not via `port.postMessage()`; Firefox
  handled all paths. Current status not checked. UNVERIFIED whether fixed.
  https://issues.chromium.org/issues/40855462
- Projects doing Module-through-`processorOptions`: https://github.com/security-union/videocall-rs/pull/935
  - With a bytes fallback when the Module cannot be cloned: https://github.com/liminal-hq/undertone/pull/124
- Projects sending bytes (a Module sent by postMessage surfaced as `messageerror`): https://github.com/AnthonE/Gates/pull/152
- Jitsi compiles synchronously from an inlined binary (`-s SINGLE_FILE=1 -s WASM_ASYNC_COMPILATION=0`) because
  a worklet will not wait on promises.
- Base64-in-JS (fastenhancer-web, DeepFilterNet3.c.wasm, WorkAdventure) also sidesteps CSP and asset hosting,
  at +33 % size before compression. One project noted a strict CSP needs `'wasm-unsafe-eval'`.

### 2.3 Dropping wasm-bindgen

- Hand-written instantiation with two imports (panic hook and an externref table initialiser) worked for one
  author; `Vec<f32>` returns come back as pointer/length pairs. https://davidshimel.substack.com/p/theyre-claudes-not-gods
- A planned nnnoiseless port targets `wasm32-unknown-unknown` with no wasm-bindgen at all.
  https://github.com/SoaresDavidson/Vcall/issues/5
- DeepFilterNet3.c.wasm's API is the shape to copy: `create(weights_ptr, weights_len)`, write N floats at the
  input pointer, `process()`, read N floats at the output pointer.

### 2.4 Simplest robust pattern (my analysis)

1. Rust `cdylib`, no wasm-bindgen, `panic = "abort"`, zero imports. Exports: `alloc(len) -> ptr` (init only),
   `init(weights_ptr, len) -> status`, `input_ptr()`, `output_ptr()`, `process_hop()`, `reset()`.
2. Main thread: `fetch` wasm and weights, `WebAssembly.validate` a tiny SIMD probe to choose the build,
   `WebAssembly.compile`, pass the Module via `processorOptions` with a bytes fallback.
3. Processor constructor: `new WebAssembly.Instance`, copy weights into linear memory, run warm-up frames, build
   `Float32Array` views once (re-create only if `memory.buffer` changes; never grow memory after init).
4. `process()`: copy 128 in, every 4th call run one hop, copy 128 out. No allocation, no closures, no port
   traffic on the hot path.
5. Bypass (pass input through with matched delay) if init failed or the load monitor trips.
6. Test in real browsers (Playwright / Vitest browser mode), not Node.

---

## 3. WASM SIMD performance and the Rust SIMD toolchain

### 3.1 simd128 against native

- No clean same-kernel GEMM table was found. Data points:
  - Hand-written Rust transformer decoder, instructions per step: native 56 M, wasm simd128 182 M (3.25x),
    wasm simd128 + relaxed-simd FMA 120 M (2.1x). onnxruntime-web 1.30 wasm: 481 M.
    https://github.com/pzfreo/homr-rs/pull/1
  - Same spectral kernel on an M4 Pro: native NEON took 0.74-0.93x the wasm time (wasm 1.08-1.35x slower).
    https://github.com/steph4n-gh/chronos-spectral/pull/3
  - Scalar wasm vs native on PolyBench: about 1.3x. https://arxiv.org/pdf/2206.12888
  - Browser vs native DL inference, worst case on CPU: up to 16.9x. https://arxiv.org/pdf/2402.05981
- (my analysis) Against NEON expect 1.1-1.5x; against AVX2+FMA expect 2-3.5x (half the vector width, no FMA).
  The "1.5-3x" guess in the question is about right; plan on 3x against AVX2.
- Subnormals: wasm cannot set FTZ/DAZ. tract had padding lanes holding leftover bytes, giving ~57k subnormal
  multiplies per frame, each a ~100-cycle microcode assist on Intel; that build was 1.6x slower on Intel than
  the non-SIMD one, while Apple cores were unaffected. A GRU state decaying in silence does the same thing.
  https://github.com/workadventure/noise-suppression/pull/19
  - (my analysis) Clamp tiny state values to zero, or add a negligible bias, and test with digital silence.

### 3.2 relaxed-simd (FMA, int8 dot)

- Chrome 114+. Firefox 146+ (per a caniuse check dated 2026-09-25 quoted in a library issue).
- Safari: not in stable. Safari Technology Preview 250 (August 2026) added relaxed SIMD; STP 257 (mid-November
  2026) fixed a wrong result on ARM64 in `i32x4.relaxed_dot_i8x16_i7x16_add_s`. A single relaxed instruction
  makes iOS Safari reject the whole module.
  - https://webkit.org/blog/18191/release-notes-for-safari-technology-preview-250/
  - https://webkit.org/blog/18652/release-notes-for-safari-technology-preview-257/
  - https://github.com/google-ai-edge/LiteRT/issues/10903
  - https://github.com/gradium-ai/xn/pull/74
  - https://github.com/linebender/fearless_simd/issues/401
  - SEO pages claiming "Safari 18.4+" are wrong.
- `relaxed_madd` may or may not fuse; libraries probe at startup. Results therefore differ across engines.
- Gain where it works: tract reports 8-23 % on low-accumulator matmul paths; the decoder above saw ~34 % fewer
  instructions.
- (my analysis) Treat relaxed-simd as an optional third build chosen with `WebAssembly.validate`. Do not
  depend on it, and do not expect bit-identical output from it.

### 3.3 Rust SIMD options on a stable toolchain

- `std::simd` / `portable_simd`: still nightly-only as of the September 2026 survey, with no timeline; it can
  fall back to scalar silently where LLVM lacks an op. Not usable on Rust 1.99 stable.
  https://shnatsel.github.io/state-of-simd-rust-2026/
- `core::arch::wasm32` intrinsics are stable. Wasm has no runtime dispatch: compile one binary with
  `-C target-feature=+simd128` and one without, and pick in JS.
- `+simd128` is off by default on wasm32. Two projects found they had never enabled it.
  https://github.com/jl1nie/mfsk-core/issues/208 , https://github.com/clarkedb/sonoglyph/issues/65
- Crates, all with wasm routines per the survey:
  - `fearless_simd` 1.0: multiversioning that works on small functions, little `unsafe`, no trig.
  - `wide` 1.0: broad coverage, no multiversioning ("fine if you're not targeting x86"), so AVX2 needs a fixed
    `-C target-cpu`.
  - `pulp`: built for `faer`, verbose, native-width oriented.
  - `macerator`: burn's CPU backend, sparse docs.
- Auto-vectorisation: "reliably(ish)" for careful code over `as_chunks()`, but float reductions are not
  vectorised because that would change results. Rust 1.98 (2026-08-20) stabilised algebraic float ops
  (`algebraic_add` etc.), which permit reassociation; code must be rewritten to use them. Source: same survey.
  - (my analysis) Not a basis for a parity-gated kernel. Explicit lane-structured reductions are both faster and
    reproducible.
- `f16`: not stable. Rust 1.94 stabilised fp16 intrinsics except those that need the `f16` type; a 2026 project
  goal targets stabilisation; Rust 1.100 (2026-11-12) does not include it.
  - https://github.com/rust-lang/goals/issues/655
  - https://blog.rust-lang.org/2026/03/05/Rust-1.94.0/
  - Irrelevant here: f16 to f32 at load time is about twenty lines of bit manipulation, or the `half` crate.
    Wasm has no f16 SIMD anyway.

### 3.4 FFT

- `rustfft` 6.2 added wasm SIMD; current release 6.4.1. Opt-in `wasm_simd` feature, no runtime detection (a
  non-SIMD engine traps). `realfft` forwards the feature. Use `FftPlanner`; constructing algorithms directly
  skips the SIMD kernels.
  - https://users.rust-lang.org/t/release-rustfft-6-2-now-with-support-for-wasm-simd-acceleration/105675
  - https://lib.rs/crates/realfft/features
- Gain: about 1.5-2x for f32 FFTs. One project went from 2.70 to 1.55 ms on its FFT stage while the wasm grew
  from 907 to 1304 KB, mostly SIMD butterflies. https://github.com/miniscope/CaLab/pull/306
  - That is ~400 KB of a 3 MB budget for the generic planner. (my analysis) A fixed-size 1024 real FFT written
    or trimmed for this one size would be far smaller.
- Open issue on a development branch: `SimdRadixN` 10x+ slower than the old wasm Radix4 for power-of-two sizes,
  because of missed inlining across `#[target_feature]`. https://github.com/ejmahler/RustFFT/issues/186
- Cost of one 1024-point real FFT: no published wasm figure found. (my analysis) Single-digit to low tens of
  microseconds; UNVERIFIED. In the DTLN worklet profile, FFT + magnitude + inverse FFT together were 4 % of the
  frame. In faster-enhancer.c the gap between the GEMM floor (RTF 0.033) and the measured 0.069 is "STFT,
  nonlinearities and per-frame quantization", so the non-GEMM part is not negligible once the kernels are fast.

---

## 4. Alternatives to hand-written inference

Criteria: (a) wasm32 in a worklet without threads, (b) size, (c) per-frame allocation, (d) speed on small
conv+GRU, (e) maintenance.

| Option | (a) worklet, no threads | (b) size | (c) allocation | (d) speed | (e) maintenance |
|---|---|---|---|---|---|
| tract 0.23.10 (2026-11-05) | Proven (WorkAdventure DFN3) with TextEncoder + getRandomValues shims | 9.0 MB (2.3 MB gzip) on 0.21.4, 14.6 MB on 0.23.8, plus model | Not guaranteed; recent releases removed many per-turn copies. Model build blocks the audio thread 200-550 ms | Good since 0.23: DFN3 0.33 ms (Apple Silicon), 0.72 ms mean (i7-10750H) | Very active; heavy API churn 0.21 to 0.23 |
| ONNX Runtime Web | Not supported; loader needs `document`. Issue 13072 open since 2022 per third parties (status UNVERIFIED) | `ort-wasm-simd-threaded.wasm` 11-14 MB (about 3 MB gzip) | n/a | 2.6-4x more instructions than hand-written Rust in one comparison | Active, large |
| `ort` + `ort-web` 0.4.0 | Async init, runtime fetched at run time, separate wasm contexts with manual data sync | ORT web | n/a | n/a | Young |
| `ort` + `ort-tract` | Same as tract | tract | tract | tract | Pins tract 0.23 |
| burn-onnx codegen (Flex backend) | Plausible; no worklet report found | No published wasm size | Tensor framework, allocates per op (UNVERIFIED) | No GRU numbers published | Active; GRU import since 0.21; NdArray backend being phased out |
| candle | wasm SIMD incomplete (scalar fallbacks reported Oct 2026) | Large | Allocates | Not tuned for this | Active, LLM-focused |
| wonnx | WebGPU only | - | - | - | Archived 2025-05-07 |
| rten 0.27 | wasm SIMD build; no worklet report found | No published size | Unknown | GRU fused in 0.27; native ~20 % behind ORT (2024) | One maintainer, active |
| LiteRT.js (XNNPACK) | Needs a fork: stock loader fails with "document is not defined"; relies on an internal sync path | Multi-MB (all variants 37 MB; one is downloaded) | Emscripten heap | Fast: DTLN 4.3 ms vs 32 ms for a Rust/WASM backend | Google; upstream ESM PR open; threads need relaxed SIMD |
| ExecuTorch wasm | Experimental | 6.6 MB (WebGPU build) | - | - | Experimental |
| IREE | Experimental web targets (community wiki only, UNVERIFIED) | - | - | - | - |
| onnx2c | Emits one C file, no heap | Small | None | Plain C loops | GRU PR dated 2026-10-08, merge state unclear; four bugs reported on FastEnhancer 48 kHz graphs |
| microflow / edge-infer / nanoglia | No GRU | Tiny | None | - | MCU-focused |

Sources for the table:
- tract: https://github.com/sonos/tract/blob/main/CHANGELOG.md , https://github.com/workadventure/noise-suppression/pull/19
- ORT web: https://github.com/microsoft/onnxruntime/issues/13072 , https://github.com/microsoft/onnxruntime/issues/14529 , https://github.com/microsoft/onnxruntime/issues/33516 , https://gist.github.com/CosteGieF/1f98738b3eab2592593ecb2018a98211
- ort backends: https://ort.pyke.io/backends , https://ort.pyke.io/backends/web , https://ort.pyke.io/backends/tract
- burn: https://burn.dev/blog/release-burn-onnx-0.21.0/ , https://github.com/tracel-ai/burn-onnx
- candle: https://github.com/huggingface/candle/issues/4043
- wonnx: https://github.com/webonnx/wonnx
- rten: https://github.com/robertknight/rten , https://github.com/dvjn/ears/pull/11
- LiteRT.js: https://workadventu.re/tech/building-an-easy-to-use-browser-noise-suppression-library-in-an-audio-worklet/ , https://github.com/thisisqubika/litert-poc
- ExecuTorch: https://github.com/pytorch/executorch/blob/main/extension/wasm/README.md , https://huggingface.co/executorch-community/ExecuTorch-WebGPU-Runtime
- onnx2c: https://github.com/kraiskil/onnx2c/pull/137 , https://github.com/kraiskil/onnx2c/issues/133
- microflow: https://github.com/matteocarnelos/microflow-rs ; edge-infer: https://github.com/idan-ben-ami/edge-infer

### 4.1 tract in detail (the only serious runtime contender)

- Wasm kernels arrived late. tract-linalg at 0.21.4 has none, so `+simd128` alone gained nothing. 0.21.18 has a
  4x4 f32 matmul kernel. 0.23.0 added relaxed-SIMD FMA in all matmul kernels, a 32x1 GEMV, vectorised
  sigmoid/tanh and `rustfft wasm_simd`. 0.23.6 fused the ONNX GRU cell (`GruEpilogue`) and added more simd128
  kernels. 0.23.8 runs a GRU "as one op instead of a per-timestep scan" with packed state no longer reallocated
  each timestep. 0.23.9 picks an 8x8 GEMM at every N and fuses GRUs of any hidden size.
- Measured effect on DFN3 per 10 ms frame, real noisy speech:
  - Apple Silicon, Node/V8: 1.05 ms (0.21.4), 0.99 ms (0.21.18 + SIMD), 0.33 ms (0.23.8 + SIMD), p95 0.45 ms.
  - i7-10750H, Chrome 154: 2.22 ms mean / 4.9 ms p95 (0.21.4), 3.62 / 8.2 ms (0.21.18 + SIMD, the subnormal
    bug), 0.72 / 1.5 ms (0.23.8 + SIMD).
- A project on tract 0.21 today gets no wasm SIMD from it, and the upgrade to 0.23 is a real port (facade crate
  renamed, `Value` to `Tensor`, output selection API changed, state API changed, MSRV 1.91).
- 0.21.5 and 0.21.6 failed to load DFN3 at all ("Patch created duplicate name").
- Size alone rules it out for a 3 MB budget: 9-14.6 MB of wasm before the model.
- tract-pulse: STFT pulsifier added in 0.23.0-dev.4; pulse changes in every release since, several labelled
  experimental or WIP. I would not build a product on it yet.

### 4.2 Existing hand-written inference for speech enhancement

- RNNoise (C), the reference point for "tiny, hand-written, runs everywhere". wasm ~150 kB.
- `nnnoiseless` (Rust port of RNNoise 0.1): tuned port about 15 % faster than C natively; first safe translation
  was about 50 % slower. No hand-written simd128; no published wasm size. Model is the 2018 ~215 K-param net.
  https://github.com/jneem/nnnoiseless , https://jneem.github.io/nnnoiseless/
- `rnnoise-rs` (Rust port of the 2024 RNNoise, 2x Conv1D + 3x GRU, ~2.9 M params), dependency-free:
  - ~316 us float / ~105 us int8 per 10 ms frame on an M4 Pro; ~159 us mean, 1.04 ms max on a Zen 3 core.
  - Float model 11.3 MiB, int8 3.0 MiB; loader-only binary 361 KB.
  - Runtime-dispatched int8 dot kernels (NEON sdot; SSE2 to AVX-512 VNNI); every tier computes the same exact
    i32, so output is tier-independent. No wasm tier mentioned.
  - No-allocation gate enforced by an allocation-counting test.
  - https://github.com/Mumcord/rnnoise-rs
- `faster-enhancer.c` (C, int8, FastEnhancer-Medium 48 kHz, 511,754 params), arXiv 2607.25350:
  - q8 weights 565,108 bytes; static library 162 KiB; state 432 KB allocated once.
  - 46.3 MMAC per 6.67 ms frame with Winograd (about 6.9 GMAC/s).
  - M2: 0.458 ms per frame (RTF 0.069) vs ONNX Runtime fp32 1.531 ms (RTF 0.230), 3.3x. NEON-only tier
    1.085 ms. Galaxy S23+: 0.640 ms (I8MM), 2.293 ms (NEON only).
  - Kernel throughput about 186 GMAC/s (int8 SDOT) on M2.
  - Paced to the real 6.67 ms deadline instead of racing through a file, per-frame cost rises 4.2x (RTF 0.286)
    while energy drops 49 %; on efficiency cores 96 % of deadlines are missed. 30-minute paced run on M2: p99
    RTF 0.56, 0.124 % misses; S23+: p99 1.01, 1.77 % misses.
  - No wasm tier.
  - https://arxiv.org/abs/2607.25350 , https://github.com/ishine/faster-enhancer.c
- `faster_enhancer_rs`: std-only Rust translation of the above. x86-64 only (AVX2 + FMA + F16C, VNNI when
  present), no NEON, no wasm, no scalar fallback. 320-sample frames, 704-sample (14.67 ms) delay. Weights
  embedded as `weights/fe.q8`, checked and prepared in `const` evaluation. 3 commits, 0 stars, no benchmarks, no
  tests directory visible, no crates.io release found. MIT. It demonstrates the approach; it is not something to
  depend on. https://github.com/TanixLu/faster_enhancer_rs
- DeepFilterNet3.c.wasm and fastenhancer-web: see 1.7.
- libDF (DeepFilterNet's own Rust loop) uses tract; RTF 0.19 on an i5-8250U for DFN3 natively.
  https://arxiv.org/pdf/2305.08227
  - Its defaults skip stages by local SNR; WorkAdventure found PESQ-WB 2.12 with defaults against 3.22 for the
    PyTorch reference, fixed by always running both stages. A white-noise benchmark was invalid for the same
    reason. https://github.com/workadventure/noise-suppression/pull/21
- Alkove: "the engine is Alkove's own, in Rust", single thread native or wasm, custom `.alk` format, 185 KB
  brotli engine, 16 kHz models of 1-2.8 MB. Internals not inspected.
  https://github.com/alkove-ai/alkove , https://huggingface.co/alkove-ai/alkove-focus-1
- `sonora`: pure-Rust port of WebRTC audio processing (AEC3, NS, AGC2). Its noise suppressor is a classical
  Wiener filter, not neural. 0.2.0 on 2026-07-29. No wasm information found.
  https://github.com/dignifiedquire/sonora
- A caution from WorkAdventure: their first "Rust/WASM browser backend" for DTLN took 32.1 ms per 32 ms frame
  against 4.3 ms for LiteRT.js/XNNPACK. Which Rust runtime that was is not stated (UNVERIFIED). Writing it in
  Rust does not make it fast; the kernels do.

---

## 5. Weight format and quantisation

### 5.1 safetensors or custom

- safetensors layout: 8-byte little-endian u64 header length, UTF-8 JSON header (per tensor: `dtype`, `shape`,
  `data_offsets`), then raw little-endian row-major data. Dtypes include F16, F32, I8. Metadata values must be
  strings. https://github.com/safetensors/safetensors
- The Rust crate (0.7.0) has a no_std mode but parses the header with serde + serde_json. Alignment of tensor
  data is a writer convention, so a zero-copy `&[f32]` cast needs a runtime alignment check. (The second half
  is my inference.)
- What the hand-written ports do: all use a flat blob. `fe.q8` (faster-enhancer), `dfn3_weights.bin`, RNNoise's
  weight blob, Alkove's `.alk`.
- (my analysis) A custom format is unnecessary but harmless. The lowest-risk split:
  - Export from Python as safetensors (inspectable, standard tooling, the artefact parity tests load).
  - Convert at build time to a flat, 64-byte-aligned blob with a version, a hash of the architecture config and
    a checksum. The runtime then needs no JSON parser, and int8 scales live in a fixed table, not in string
    metadata.
  - Or parse safetensors directly with a ~100-line parser if a build step is unwanted. Both are fine. What
    matters is a hash that makes a weights/architecture mismatch fail at load.

### 5.2 Is f16 / int8 storage worth it

- Lossless compression does little for float weights: brotli quality 11 left ONNX f32 files at 82-93 % of raw
  size, while wasm binaries shrank to 14-22 %.
  https://dev.to/sharksevenro/brotli-for-wasm-and-onnx-models-cutting-first-run-downloads-from-72mb-to-13mb-and-the-2ik7
- Already-quantised int8/f16 files gain 12-21 % from gzip/brotli. Real reductions come from precision: 169 MB
  f32, 86.5 MB f16, 44.7 MB int8 for one model. https://github.com/audiojs/neural/tree/main/packages/neural-separate
- (my analysis) Sizes for this model:
  - 1.0 M params: 4.0 MB f32 (about 3.5 MB brotli), 2.0 MB f16, 1.0 MB int8.
  - 1.5 M params: 6.0 MB f32 (about 5.2 MB brotli), 3.0 MB f16, 1.5 MB int8.
  - Against a 3 MB budget including the wasm: f32 never fits; f16 fits up to roughly 1.3 M params; int8 fits
    easily. So f16 storage is the minimum and is essentially free in quality.
- f16 unpacked to f32 at load keeps a 4-6 MB working set, larger than L2 on a 2020 laptop. Each hop then
  streams the GRU weights from L3 or memory, usually cold after 10 ms of idle. int8 executed as int8 would cut
  that to 1-1.5 MB, but wasm simd128 has no good int8 dot product without relaxed-simd, which Safari lacks.

### 5.3 int8 quality on GRU-based enhancers

- Uniform int8 PTQ on recurrent enhancers costs about 0.08-0.3 PESQ:
  - TinyDenoiser: mixed FP16 / int8-RNN loses ~0.06 PESQ; uniform int8 ~0.3. https://arxiv.org/pdf/2210.07692
  - LiSenNet: recurrent original loses 0.08 PESQ under int8 PTQ, the convolutional redesign 0.01.
    https://arxiv.org/pdf/2609.29866
- Careful schemes get close to lossless:
  - faster-enhancer.c: -0.006 PESQ, -0.08 dB SNR over 824 VoiceBank-DEMAND utterances, using per-frame
    activation ranges (no calibration set), fp16 recurrent state ("int8 state would lose about 40 dB SNR
    through the decoder concat"), and fused GRU epilogues.
- Waveform error is still large: q8 vs fp32 output is 37.8 dB SNR in that paper; rnnoise-rs int8 vs float is
  4.8e-6 relative energy (about -53 dB).
- (my analysis) Ship f16-stored, f32-executed first. Treat int8 as a later optimisation that needs its own
  quality gate and probably QAT or mixed precision for the GRU.

---

## 6. Compute budget sanity check

### 6.1 Arithmetic

- Hop = 512 / 48000 = 10.667 ms, 93.75 hops/s.
- 100 MMAC/s x 0.010667 s = 1.067 M MAC per hop. 50 MMAC/s gives 0.533 M. The arithmetic in the question is right.
- (my analysis) Consistency check on the claim itself: a layer applied once per frame costs one MAC per
  weight, so a fully dense/GRU network of 1.0-1.5 M params costs 94-141 MMAC/s at 93.75 fps. Any convolution
  that slides over frequency bins reuses its weights and costs more. "1-1.5 M params at 50-100 MMAC/s"
  therefore implies almost no weight reuse, i.e. nearly all parameters in GRU/linear layers on a heavily
  reduced feature vector. Worth re-counting with a profiler (e.g. ptflops/thop on the real model), because
  published conv-heavy models of this size are 10-100x higher (see below).

### 6.2 Published reference points

| Model | Params | MACs | Timing | Source |
|---|---|---|---|---|
| GTCRN (16 kHz) | 23.7 K (48.2 K counting ERB) | 39.6 MMAC/s (33.0 revised) | no RTF found | https://github.com/Xiaobin-Rong/gtcrn |
| RNNoise 2018 | ~215 K | - | ~26-28 us per 10 ms frame native; ~0.95 % of a core in a worklet | https://github.com/Mumcord/rnnoise-rs |
| RNNoise 2024 | ~2.9 M | - | 316 us float / 105 us int8 per frame (M4 Pro) | same |
| DeepFilterNet3 | 2.13 M | 0.344 GMAC/s (third-party table; UNVERIFIED in the paper) | wasm via tract 0.23.8: 0.33 ms (Apple Silicon), 0.72 ms mean / 1.5 ms p95 (i7-10750H) | https://arxiv.org/html/2409.03377v2 , https://github.com/workadventure/noise-suppression/pull/19 |
| FastEnhancer-Tiny 48 kHz | 28 K | 177 MMAC/s | RTF 0.0122 (M5, ORT) | https://github.com/aask1357/fastenhancer |
| FastEnhancer-Base 48 kHz | 101 K | 750 MMAC/s | RTF 0.0261 | same |
| FastEnhancer-Small 48 kHz | 207 K | 1.82 GMAC/s | RTF 0.0421 | same |
| FastEnhancer-Medium 48 kHz | 512 K | 8.0 GMAC/s | RTF 0.1347 (M5, ORT); 0.069 int8 hand-written (M2) | same, and arXiv 2607.25350 |
| FastEnhancer-Large 48 kHz | 1132 K | 32.4 GMAC/s | RTF 0.4239 | same |

Note how far the plan's figure sits from these: FastEnhancer-Tiny at 28 K params already costs 177 MMAC/s at 48 kHz.

### 6.3 Effective throughput in wasm (my analysis)

- DFN3 is about 3.44 M MAC per 10 ms frame. Through a generic runtime in single-thread wasm that is:
  - 0.33 ms on Apple Silicon: about 10 GMAC/s effective.
  - 0.72 ms mean on an i7-10750H in Chrome: about 4.8 GMAC/s; at p95 (1.5 ms) about 2.3 GMAC/s.
- So on a 2020-era laptop, expect 2-5 GMAC/s effective for small-matrix conv/GRU work in simd128, including
  runtime overhead, FFT and nonlinearities. Scalar wasm would be perhaps 0.5-1 GMAC/s (UNVERIFIED).
- 1.07 M MAC per hop at 2-5 GMAC/s is 0.2-0.5 ms. Add STFT/iSTFT and feature code: perhaps 0.3-0.8 ms total in
  wasm, 0.1-0.3 ms native. Both the 3 ms wasm and 0.8 ms native budgets hold with 3-10x headroom, if the MAC
  count is right.
- If the true count is 10x higher (1 GMAC/s, i.e. FastEnhancer-Base/Small territory), wasm lands at 2-5 ms on
  that laptop and the plan fails on the quantum deadline. The MAC count is the number everything hinges on.
- Apply the pacing penalty: racing benchmarks understate real-time cost. The measured factor was 4.2x on an M2
  (frequency scaling and cold caches between frames). 0.5 ms racing can be 2 ms paced.
- Phones: the hand-written DFN3 wasm is 4x slower on a Galaxy S23 and 6x slower on a Note10+ than on an M2.
- Budget suggestion: racing mean <= 0.5 ms and p99 <= 1 ms in wasm on the reference 2020 laptop; paced p99 in a
  real worklet <= 2 ms; hard ceiling 2.67 ms.

---

## 7. Parity methodology

### 7.1 What is wrong with "bit-for-bit to <= -60 dB RMS error"

- The phrase mixes two different gates. Bit-for-bit means identical. -60 dB relative RMS is an error of 1e-3.
- For an f32 port, -60 dB is far too loose. It would pass a wrong epsilon, a slightly different window, or a
  one-off state-update bug.
- For an int8 port, -60 dB is unreachable (-38 to -53 dB in the ports above).

### 7.2 What ports actually achieve

- f32 against f32, different runtimes or builds:
  - tract 0.23.8 vs 0.21.4: max abs diff 1.9e-7 over 240,000 samples; 0.21.18 vs 0.21.4 bit-identical.
  - Rust/WASM vs LiteRT.js DTLN: max abs diff 9.69e-8.
  - Hand-decomposed GRU in onnx2c vs ONNX Runtime: about 1e-6.
  - rnnoise-rs vs C: 3 of 47,520 samples differ by 1 LSB (relative energy 2e-10), "within the spread between two
    C builds". The cause of an earlier mismatch was one expression the C code evaluates in double.
    https://github.com/Mumcord/rnnoise-rs/pull/23 , https://github.com/Mumcord/rnnoise-rs/pull/20
- Bit-identity across SIMD tiers is achievable when designed for:
  - rnnoise-rs: every int8 tier computes the same exact i32.
  - faster-enhancer.c: all tiers in one architecture family are byte-identical on 824 clips, checked by output
    hash as a gate; identical across M2 and S23+. Not identical across ARM and x86.
- What broke bit-identity there:
  - `cosf` for the Hann window: two compilers agreed on only 22 % of samples until the window was computed in
    double and narrowed once.
  - No FMA (SSE4.1): about 1 ULP drift per operation, so that tier was dropped.
  - Approximate reciprocal instructions (`rcp14` vs 12-bit `rcp`): 44 dB SNR between tiers.
  - Device flush-to-zero behaviour (noted in an unrelated GEMM verification paper, arXiv 2611.01045).
- STFT convention: the reference must be generated on the engine's causal STFT grid; the default centred STFT
  was offset by 192 samples in that port.
- Tolerances used elsewhere: f32 atol 1e-5 / rtol 1.3e-6 (Ambiq test suite); for accumulations, a tolerance that
  scales with the number of terms (atol = 1e-4 + 2e-7 K).
  https://github.com/AmbiqAI/helia-core-tester/pull/456 , https://arxiv.org/pdf/2611.01045
- Quality parity matters as much as numeric parity: libDF "worked" numerically yet scored PESQ 2.12 against 3.22
  because of default thresholds. WorkAdventure's check was 50 VoiceBank+DEMAND pairs through the wasm, frame by
  frame, against PyTorch on PESQ-WB, STOI, SI-SDR, DNSMOS.
- Relaxed-simd FMA is engine-dependent; onnxruntime-web without FMA differed from native at some decoding step on
  29 of 627 sequences in one comparison. https://github.com/pzfreo/homr-rs/pull/1

### 7.3 A layered gate (my analysis)

1. Scalar Rust reference vs PyTorch, f32: per-layer activations on fixed inputs within atol 1e-5 / rtol 1e-5;
   end-to-end waveform error <= -90 dB over 30 s clips including silence, onsets and clipping. A recurrent state
   will accumulate drift, so also check that the error does not grow over a 10-minute run.
2. SIMD tiers vs the scalar reference: bit-identical, by giving the scalar path the same lane-structured
   summation order and using FMA either everywhere or nowhere. Realistic targets: NEON == simd128 == scalar
   (no FMA); AVX2+FMA as its own family with a -100 dB gate against the others.
3. Quantised variants: no waveform gate tighter than about 35 dB SNR; gate on metric deltas (PESQ, SI-SDR,
   DNSMOS) over a fixed corpus, plus an output hash to detect unintended change.
4. Browser: run the same vectors through the actual worklet build in Chromium, Firefox and WebKit.
5. Pin the oracle (exported reference vectors from a tagged PyTorch commit), not the port's own output.
6. Compute the window in f64 at build time and embed it. Use the same FFT conventions on both sides and test the
   STFT/iSTFT round trip separately from the network.

---

## 8. Recommendation

### 8.1 Hand-written kernel or existing runtime

- Hand-written is the right call for the browser target. The 3 MB budget and the worklet environment exclude
  every general runtime: tract is 9-14.6 MB, ORT web 11-14 MB and unsupported in worklets, LiteRT.js needs a
  fork. All shipping sub-3 MB denoisers are hand-written.
- Keep tract or ORT on the native side as the cross-check oracle while the hand-written path matures. Do not
  delete them until parity gates have been green for a while.
- Keep the kernel set small: GEMV/GEMM f32, the conv inner loop, GRU cell with fused gates, sigmoid/tanh,
  elementwise ops, and a fixed-size real FFT. A scalar reference for each is mandatory.
- SIMD on stable Rust 1.99: `core::arch` intrinsics per target behind one small internal trait, or
  `fearless_simd` 1.0. Not `std::simd` (nightly). Not auto-vectorisation for anything parity-gated.
- "100 % pure Rust, zero dependencies" is a preference, not a requirement. `realfft` is fine natively; in wasm
  check its size cost (about 400 KB observed for the SIMD planner) before deciding on a fixed 1024 FFT.

### 8.2 Simplest robust worklet architecture

- Inference inside the AudioWorkletProcessor. No Worker, no SAB, no COOP/COEP.
- No wasm-bindgen: `extern "C"` exports, zero imports, static state, `panic = "abort"`.
- Two builds (simd128, scalar), chosen on the main thread with `WebAssembly.validate`. Optional relaxed-simd
  build later, for Chrome and Firefox only.
- Module via `processorOptions` with a bytes fallback; weights as a transferred ArrayBuffer; f16 on the wire,
  f32 in memory.
- 128-in / 128-out ring; one hop every 4th quantum; warm-up frames at init; no allocation or port traffic in
  `process()`.
- `AudioContext({ sampleRate: 48000 })` created after getUserMedia resolves; verify the rate; bypass or resample
  otherwise.
- Load monitor counting slow hops (as WorkAdventure does: an `overload` message when two 2 s windows exceed
  70 % load), with automatic bypass.
- Real-browser CI on Chromium, Firefox and WebKit.

### 8.3 Realistic budgets

- Compute: racing mean <= 0.5 ms, p99 <= 1 ms in wasm on a 2020 laptop; paced p99 <= 2 ms; hard limit 2.67 ms
  per quantum. Native <= 0.3 ms. Replace "3 ms per hop" with these.
- Download: wasm 100-300 KB brotli plus f16 weights. 3 MB total holds up to about 1.3 M params. Above that,
  int8 or a larger budget.
- Latency: 21.3 ms algorithmic, plus the AudioContext hop. State 30-45 ms added on desktop until measured.

### 8.4 Top risks

1. The MAC count. 50-100 MMAC/s for a 1-1.5 M-param conv+GRU model at 48 kHz is at the theoretical floor;
   comparable published models are 10-300x higher. If it is really around 1 GMAC/s the wasm budget fails.
2. Tail latency, not mean: pacing (4.2x measured), efficiency cores, thermal throttling, Liftoff before
   tier-up, subnormals on Intel, DevTools open.
3. The quantum deadline is 2.67 ms and the hop's work lands in one quantum.
4. Safari and iOS: sample-rate behaviour, no relaxed-simd, unknown worklet thread priority, WebKit audio bugs.
5. Android Chrome: zero-input bug with 3A enabled on Android 14+, burst/quantum mismatch glitches.
6. int8: GRU quality loss without QAT or mixed precision, and no fast int8 dot in baseline simd128.
7. Parity gate as written would not catch real bugs (f32) or cannot be met (int8).
8. Maintenance: a private inference engine means every architecture change in training needs a matching
   kernel change and re-validation. Freeze the operator set early.
9. Worklets inside iframes do not get the realtime thread in Chrome.

### 8.5 Items I could not confirm

- fastenhancer-web's sizes and timings (README unreachable).
- Whether ONNX Runtime issue 13072 is still open (page returned 404 to the fetcher; third parties say unresolved).
- Whether Chromium issue 40855462 (Module via `port.postMessage`) is fixed.
- Safari's AudioWorklet thread priority.
- Millisecond cost of `MediaStreamAudioDestinationNode` into WebRTC per browser.
- Google Meet's and Discord's current web implementations.
- A 1024-point real FFT timing in wasm.
- DeepFilterNet3's 0.344 GMAC/s figure from the paper itself.
- Stable Safari version that will carry relaxed-simd.
- Which runtime WorkAdventure's slow "Rust/WASM backend" for DTLN used.
- Allocation behaviour of burn, rten and candle per inference call.
