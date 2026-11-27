// Oto-1 AudioWorkletProcessor: runs one 512-sample hop of the wasm model every 4th 128-sample
// render quantum, entirely inside the audio thread (no Worker, no SharedArrayBuffer).
//
// processorOptions:
//   module   WebAssembly.Module (preferred; compiled on the main thread)
//   bytes    ArrayBuffer of the .wasm (fallback when a Module cannot be cloned into the worklet)
//   weights  optional ArrayBuffer of a .otow file; otherwise a random model (arch, seed)
//   arch     1 = Candidate A, 2 = Candidate B          seed  random-weight seed
//   warmup   hops to run in the constructor so V8 tiers the hot functions up to TurboFan before
//            audio flows (wasm has no on-stack replacement; Liftoff code is ~1.5x slower)
//
// Latency: 512 (quantum ring) + 512 (model overlap) = 1024 samples = 21.33 ms, also in bypass.
// No allocation in process() except the once-a-second stats message.

const HOP = 512;
const QUANTUM = 128;
const DELAY = 2 * HOP;
// A hop must finish inside one 2.67 ms quantum; slower than this counts as an overrun.
const SLOW_MS = 2.0;
const now = globalThis.performance && performance.now ? () => performance.now() : () => Date.now();

class OtoProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    this.bypass = true;
    this.forcedBypass = !!o.forcedBypass;
    this.error = null;
    this.pos = 0;
    // Dry path, always running, so bypass keeps the same 1024-sample delay and switching is clean.
    this.dry = new Float32Array(DELAY);
    this.dryPos = 0;
    this.hops = 0;
    this.slow = 0;
    this.windowHops = 0;
    this.windowSlow = 0;
    this.maxMs = 0;
    this.sumMs = 0;
    // Fixed-size ring for live paced percentile reporting (no per-hop allocation).
    this.hist = new Float32Array(1024);
    this.histCount = 0;
    this.histPos = 0;
    this.lastReport = currentTime;
    this.overloaded = false;
    this.disableOverload = !!o.disableOverload;
    try {
      const mod = o.module instanceof WebAssembly.Module ? o.module : new WebAssembly.Module(o.bytes);
      this.wasm = new WebAssembly.Instance(mod, {}).exports;
      const w = this.wasm;
      let rc;
      if (o.weights) {
        const bytes = new Uint8Array(o.weights);
        const p = w.oto_alloc(bytes.length);
        new Uint8Array(w.memory.buffer, p, bytes.length).set(bytes);
        rc = w.oto_init(p, bytes.length);
        w.oto_free(p, bytes.length);
      } else {
        rc = w.oto_init_random(o.arch || 1, o.seed || 1);
      }
      if (rc !== 0) {
        const m = new Uint8Array(w.memory.buffer, w.oto_error_ptr(), w.oto_error_len());
        let s = "";
        for (let i = 0; i < m.length; i++) s += String.fromCharCode(m[i]);
        throw new Error("oto_init failed: " + s);
      }
      // Views only after init: init grows linear memory, processing never does.
      this.inView = new Float32Array(w.memory.buffer, w.oto_input_ptr(), HOP);
      this.outView = new Float32Array(w.memory.buffer, w.oto_output_ptr(), HOP);
      const warm = o.warmup === undefined ? 300 : o.warmup;
      let seed = 1;
      const t0 = now();
      for (let h = 0; h < warm; h++) {
        for (let i = 0; i < HOP; i++) {
          seed = (seed * 1664525 + 1013904223) >>> 0;
          this.inView[i] = (seed / 4294967296 - 0.5) * 0.1;
        }
        w.oto_process_hop();
      }
      const warmMs = now() - t0;
      w.oto_reset();
      this.inView.fill(0);
      this.bypass = false;
      this.port.postMessage({
        type: "ready",
        simd: w.oto_simd(),
        params: w.oto_params(),
        macs: w.oto_macs_per_hop(),
        warmup: warm,
        warmupMs: warmMs,
        latencySamples: DELAY,
      });
    } catch (e) {
      this.error = String(e && e.message ? e.message : e);
      this.port.postMessage({ type: "error", message: this.error });
    }
    this.port.onmessage = (ev) => {
      const d = ev.data;
      if (d.type === "bypass") this.forcedBypass = !!d.value;
      if (d.type === "reset" && this.wasm) this.wasm.oto_reset();
      if (d.type === "unoverload" && this.overloaded) {
        if (this.wasm) {
          this.wasm.oto_reset();
          this.inView.fill(0);
          this.outView.fill(0);
        }
        this.windowHops = 0;
        this.windowSlow = 0;
        this.overloaded = false;
      }
    };
  }

  _pct(q) {
    const n = this.histCount;
    if (!n) return 0;
    const copy = this.hist.slice(0, n).sort();
    return copy[Math.min(n - 1, Math.floor((n - 1) * q))];
  }

  process(inputs, outputs) {
    const input = inputs[0] && inputs[0][0];
    const output = outputs[0];
    const out = output[0];
    if (!out) return true;
    const pos = this.pos;
    const live = !this.bypass && !this.forcedBypass && !this.overloaded;
    for (let i = 0; i < QUANTUM; i++) {
      const x = input ? input[i] : 0;
      // Dry path: DELAY-sample ring.
      const d = this.dry[this.dryPos];
      this.dry[this.dryPos] = x;
      this.dryPos = this.dryPos + 1 === DELAY ? 0 : this.dryPos + 1;
      if (this.inView) this.inView[pos + i] = x;
      out[i] = live ? this.outView[pos + i] : d;
    }
    for (let c = 1; c < output.length; c++) output[c].set(out);
    this.pos = pos + QUANTUM;
    if (this.pos === HOP) {
      this.pos = 0;
      // Suspend inference while overloaded so the render thread actually recovers.
      if (!this.bypass && !this.overloaded) {
        const t0 = now();
        this.wasm.oto_process_hop();
        const ms = now() - t0;
        this.hops++;
        this.windowHops++;
        this.sumMs += ms;
        this.hist[this.histPos] = ms;
        this.histPos = (this.histPos + 1) & 1023;
        if (this.histCount < 1024) this.histCount++;
        if (ms > this.maxMs) this.maxMs = ms;
        if (ms > SLOW_MS) {
          this.slow++;
          this.windowSlow++;
        }
      }
    }
    // Once a second: report, and trip the overload bypass if more than 2% of hops in the last
    // second overran (WorkAdventure-style load monitor).
    if (currentTime - this.lastReport >= 1) {
      if (!this.disableOverload && this.windowHops > 0 && this.windowSlow / this.windowHops > 0.02 && !this.overloaded) {
        this.overloaded = true;
        if (this.wasm) {
          this.wasm.oto_reset();
          this.outView.fill(0);
        }
        this.port.postMessage({ type: "overload", slow: this.windowSlow, hops: this.windowHops });
      }
      this.port.postMessage({
        type: "stats",
        hops: this.hops,
        slow: this.slow,
        meanMs: this.hops ? this.sumMs / this.hops : 0,
        p50Ms: this._pct(0.5),
        p95Ms: this._pct(0.95),
        p99Ms: this._pct(0.99),
        maxMs: this.maxMs,
        pTarget: this.wasm ? this.wasm.oto_p_target() : 0,
        bypass: !live,
        overloaded: this.overloaded,
        timer: globalThis.performance && performance.now ? "performance.now" : "Date.now (1 ms steps)",
      });
      this.windowHops = 0;
      this.windowSlow = 0;
      this.maxMs = 0;
      this.lastReport = currentTime;
    }
    return true;
  }
}

registerProcessor("oto-processor", OtoProcessor);
