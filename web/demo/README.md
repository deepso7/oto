# Oto-1 browser demo

This is a static page that runs Oto-1 (`crates/oto-wasm`) inside an `AudioWorkletProcessor`. It doesn't need a bundler, `wasm-bindgen`, `SharedArrayBuffer` or COOP/COEP headers.

```sh
OTO_WASM_TOOLCHAIN=stable crates/oto-wasm/build-wasm.sh   # → web/demo/wasm/oto-{simd,scalar}.wasm
python3 -m http.server -d web/demo 8000                   # then open http://localhost:8000
node web/demo/bench-node.mjs 5                            # V8 timing + parity without a browser
node web/demo/bench-node.mjs 5 --weights b24-f16.otow --label B24 --build simd   # time any .otow
```

`OTO_WASM_TOOLCHAIN` is only needed if the pinned toolchain lacks the `wasm32-unknown-unknown` target.

Files:

- **`main.js`** (main thread):
  - Probes for SIMD with `WebAssembly.validate` and picks `oto-simd.wasm` or `oto-scalar.wasm`.
  - Compiles the build and passes the `WebAssembly.Module` through `processorOptions`, with the raw bytes alongside (cloned, not transferred). The worklet uses the bytes if the Module can't be cloned or won't instantiate there.
  - Benchmarks the build, and runs it live on the mic.
- **`worklet.js`** (the processor):
  - Instantiates the wasm synchronously and loads either a `.otow` file or random weights.
  - Runs 300 warm-up hops in the constructor so V8 tiers the code up to TurboFan.
  - Collects render quanta in a ring and runs one 512-sample hop as soon as 512 samples are in (every 4th quantum at the usual 128 samples; other quantum sizes work too, with the same timing).
  - Keeps a matched 1024-sample dry delay for bypass.
  - Overload monitor: if more than 2% of hops in a second take over 2 ms, it switches to bypass.
  - Supports only 48 kHz. At any other context rate it stays in matched-delay bypass and reports why (the demo doesn't resample).
- **`bench-worker.js`, `bench-core.js`**: timing in a Worker. "Racing" runs hops back to back. "Paced" runs one hop per 10.67 ms of wall time.
- **`bench-node.mjs`**: the same timing in Node, plus a parity check against the native reference vectors. `--weights <path.otow>` times a trained or custom-width model instead of the random A/B models (no parity check), `--label <name>` names the row, and `--build simd|scalar|both` picks the builds (default both).

Caveats:

- The live `AudioWorklet` path needs no `SharedArrayBuffer` and no COOP/COEP headers, so any static server works.
- `performance.now()` is coarse without cross-origin isolation. The racing mean comes from total wall time, so it stays exact. For high-resolution (5–20 µs) per-hop timings in the Worker benchmark, serve the page with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.
- Inside the worklet only `Date.now()` exists on Chromium and WebKit, so live per-hop timings are in 1 ms steps (shown as `Date.now (1 ms steps)` in the stats line). A hop still counts as slow only when the reading is over 2 ms: with a 1 ms clock, a hop that really takes t ms in [2, 3) reads 3 with probability t − 2, so the overload trip point stays near 2 ms.
- Real added latency in a browser is the 21.33 ms chain delay plus the context and device latency, which the live view prints.
