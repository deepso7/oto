# Oto-1 browser demo

This is a static page that runs Oto-1 (`crates/oto-wasm`) inside an `AudioWorkletProcessor`. It doesn't need a bundler, `wasm-bindgen`, `SharedArrayBuffer` or COOP/COEP headers.

```sh
OTO_WASM_TOOLCHAIN=stable crates/oto-wasm/build-wasm.sh   # → web/demo/wasm/oto-{simd,scalar}.wasm
python3 -m http.server -d web/demo 8000                   # then open http://localhost:8000
node web/demo/bench-node.mjs 5                            # V8 timing + parity without a browser
```

`OTO_WASM_TOOLCHAIN` is only needed if the pinned toolchain lacks the `wasm32-unknown-unknown` target.

Files:

- **`main.js`** (main thread):
  - Probes for SIMD with `WebAssembly.validate` and picks `oto-simd.wasm` or `oto-scalar.wasm`.
  - Compiles the build and passes the `WebAssembly.Module` through `processorOptions`. If the Module can't be cloned, it passes the raw bytes instead.
  - Benchmarks the build, and runs it live on the mic.
- **`worklet.js`** (the processor):
  - Instantiates the wasm synchronously and loads either a `.otow` file or random weights.
  - Runs 300 warm-up hops in the constructor so V8 tiers the code up to TurboFan.
  - Collects 128-sample quanta in a ring and runs one 512-sample hop every 4th quantum.
  - Keeps a matched 1024-sample dry delay for bypass.
  - Overload monitor: if more than 2% of hops in a second take over 2 ms, it switches to bypass.
- **`bench-worker.js`, `bench-core.js`**: timing in a Worker. "Racing" runs hops back to back. "Paced" runs one hop per 10.67 ms of wall time.
- **`bench-node.mjs`**: the same timing in Node, plus a parity check against the native reference vectors.

Caveats:

- `performance.now()` is coarse without cross-origin isolation. The racing mean comes from total wall time, so it stays exact.
- Inside the worklet only `Date.now()` exists on Chromium and WebKit, so live per-hop timings are in 1 ms steps.
- Real added latency in a browser is the 21.33 ms chain delay plus the context and device latency, which the live view prints.
