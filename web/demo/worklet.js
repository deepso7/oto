// Oto-1 AudioWorkletProcessor: runs one 512-sample hop of the wasm model every 512 input samples
// (every 4th render quantum at the usual 128 samples), entirely inside the audio thread (no Worker,
// no SharedArrayBuffer). Any render quantum length works: the hop fires the moment 512 samples
// have been collected, even mid-quantum.
//
// processorOptions:
//   module   WebAssembly.Module (preferred; compiled on the main thread)
//   bytes    ArrayBuffer of the .wasm (used when module is missing or fails to instantiate)
//   weights  optional ArrayBuffer of a .otow file; otherwise a random model (arch, seed)
//   arch     1 = Candidate A, 2 = Candidate B          seed  random-weight seed
//   warmup   hops to run in the constructor so V8 tiers the hot functions up to TurboFan before
//            audio flows (wasm has no on-stack replacement; Liftoff code is ~1.5x slower)
//   forcedBypass  start in (matched-delay) bypass
//
// Only 48 kHz is supported: at any other context rate the processor stays in bypass and posts an
// error (code "sampleRate"); the demo does not resample.
//
// Latency: 512 (quantum ring) + 512 (model overlap) = 1024 samples = 21.33 ms, also in bypass.
// No allocation in process() except the once-a-second stats messages.

const HOP = 512;
const DELAY = 2 * HOP;
const HIST = 1024;
const RATE = 48000;
// A hop must finish inside one 2.67 ms quantum; slower than this counts as an overrun.
const SLOW_MS = 2.0;
// AudioWorkletGlobalScope has no performance.now() on Chromium and WebKit, so timings fall back
// to Date.now() in 1 ms integer steps. The slow test stays `ms > SLOW_MS` either way: a hop that
// really takes t ms in [2, 3) reads 3 with probability t - 2 (and 2 otherwise), so the fraction
// of "slow" readings still tracks the real overrun rate and the 2% trip point lands at ~2.02 ms.
// `>= 2` would instead count a 1.2 ms hop as slow 20% of the time and trip the overload bypass
// on hops that fit the 2.67 ms quantum comfortably.
const HAS_PERF = !!(globalThis.performance && typeof globalThis.performance.now === "function");
const now = HAS_PERF ? () => performance.now() : () => Date.now();

class OtoProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    this.bypass = true;
    this.forcedBypass = !!o.forcedBypass;
    this.error = null;
    this.errorCode = null;
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
    this.hist = new Float32Array(HIST);
    // Sorted copy for percentiles, preallocated so stats reports don't allocate typed arrays.
    this.histScratch = new Float32Array(HIST);
    this.histCount = 0;
    this.histPos = 0;
    this.lastReport = currentTime;
    this.overloaded = false;
    this.refillHops = 0;
    this.disableOverload = !!o.disableOverload;
    try {
      if (typeof sampleRate === "number" && sampleRate !== RATE) {
        this.errorCode = "sampleRate";
        throw new Error(`unsupported sampleRate ${sampleRate} Hz (Oto-1 requires ${RATE} Hz); matched-delay bypass only`);
      }
      const w = (this.wasm = OtoProcessor._instantiate(o));
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
      this.port.postMessage({
        type: "ready",
        simd: w.oto_simd(),
        params: w.oto_params(),
        macs: w.oto_macs_per_hop(),
        warmup: warm,
        warmupMs: warmMs,
        latencySamples: DELAY,
      });
      this.bypass = false;
    } catch (e) {
      this.bypass = true;
      this.wasm = null;
      this.inView = null;
      this.outView = null;
      this.error = String(e && e.message ? e.message : e);
      this.port.postMessage({ type: "error", code: this.errorCode || "init", message: this.error });
    }
    this.port.onmessage = (ev) => {
      const d = ev.data;
      if (d.type === "bypass") this.forcedBypass = !!d.value;
      if (d.type === "reset" && this.wasm) this.wasm.oto_reset();
      if (d.type === "unoverload" && this.overloaded) {
        if (this.wasm) {
          // Keep inView[0..pos] intact so samples already buffered for the current hop are not lost.
          this.wasm.oto_reset();
        }
        this.windowHops = 0;
        this.windowSlow = 0;
        // Keep playing matched-delay dry audio while the reset model refills its 2-hop pipeline.
        this.refillHops = 2;
        this.overloaded = false;
      }
    };
  }

  // Module first; if it is missing or will not instantiate here, compile the bytes instead.
  static _instantiate(o) {
    let moduleError = null;
    if (o.module instanceof WebAssembly.Module) {
      try {
        return new WebAssembly.Instance(o.module, {}).exports;
      } catch (e) {
        moduleError = e;
      }
    } else if (o.module !== undefined && o.module !== null) {
      moduleError = new TypeError("processorOptions.module is not a WebAssembly.Module");
    }
    if (o.bytes) {
      try {
        return new WebAssembly.Instance(new WebAssembly.Module(o.bytes), {}).exports;
      } catch (e) {
        const m = (x) => (x && x.message ? x.message : String(x));
        throw new Error(moduleError ? `module: ${m(moduleError)}; bytes: ${m(e)}` : `bytes: ${m(e)}`);
      }
    }
    if (moduleError) throw new Error(`module: ${moduleError.message || moduleError} (no bytes fallback provided)`);
    throw new Error("no wasm in processorOptions (need module or bytes)");
  }

  // In-place heapsort of this.histScratch[0..n) with zero heap allocations (no subarray views,
  // closures, or result objects). Leaves this.hist untouched.
  static _siftDown(a, start, end) {
    let root = start;
    while (true) {
      let child = root * 2 + 1;
      if (child >= end) break;
      if (child + 1 < end && a[child + 1] > a[child]) child++;
      if (a[root] >= a[child]) break;
      const t = a[root];
      a[root] = a[child];
      a[child] = t;
      root = child;
    }
  }

  _sortHistScratch(n) {
    const a = this.histScratch;
    const h = this.hist;
    for (let i = 0; i < n; i++) a[i] = h[i];
    for (let i = (n >> 1) - 1; i >= 0; i--) OtoProcessor._siftDown(a, i, n);
    for (let end = n - 1; end > 0; end--) {
      const t = a[0];
      a[0] = a[end];
      a[end] = t;
      OtoProcessor._siftDown(a, 0, end);
    }
  }

  _hop() {
    // Suspend inference while overloaded so the render thread actually recovers.
    if (this.bypass || this.overloaded) return;
    const t0 = now();
    this.wasm.oto_process_hop();
    const ms = now() - t0;
    this.hops++;
    this.windowHops++;
    this.sumMs += ms;
    this.hist[this.histPos] = ms;
    this.histPos = (this.histPos + 1) % HIST;
    if (this.histCount < HIST) this.histCount++;
    if (ms > this.maxMs) this.maxMs = ms;
    if (ms > SLOW_MS) {
      this.slow++;
      this.windowSlow++;
    }
    if (this.refillHops > 0) this.refillHops--;
  }

  process(inputs, outputs) {
    const input = inputs[0] && inputs[0][0];
    const output = outputs[0];
    const out = output && output[0];
    if (!out) return true;
    // Render quantum length: 128 today, but don't assume it (renderSizeHint may change it).
    const n = out.length;
    if (n === 0) return true;
    const inLen = input ? input.length : 0;
    const dry = this.dry;
    const inView = this.inView;
    const outView = this.outView;
    let dryPos = this.dryPos;
    let pos = this.pos;
    let i = 0;
    while (i < n) {
      const live = !this.bypass && !this.forcedBypass && !this.overloaded && this.refillHops === 0;
      // Run up to the end of the quantum or of the hop, whichever comes first, so pos stays in
      // 0..HOP-1 and the hop fires exactly when 512 samples are in (same timing as 4 x 128).
      const end = i + Math.min(n - i, HOP - pos);
      for (; i < end; i++, pos++) {
        const x = i < inLen ? input[i] : 0;
        // Dry path: DELAY-sample ring.
        const d = dry[dryPos];
        dry[dryPos] = x;
        dryPos = dryPos + 1 === DELAY ? 0 : dryPos + 1;
        if (inView) inView[pos] = x;
        out[i] = live ? outView[pos] : d;
      }
      if (pos === HOP) {
        pos = 0;
        this._hop();
      }
    }
    this.dryPos = dryPos;
    this.pos = pos;
    for (let c = 1; c < output.length; c++) output[c].set(out);
    // Once a second: report, and trip the overload bypass if more than 2% of hops in the last
    // second overran (WorkAdventure-style load monitor).
    if (currentTime - this.lastReport >= 1) {
      if (!this.disableOverload && this.windowHops > 0 && this.windowSlow / this.windowHops > 0.02 && !this.overloaded) {
        this.overloaded = true;
        this.refillHops = 0;
        if (this.wasm) {
          this.wasm.oto_reset();
          this.outView.fill(0);
        }
        this.port.postMessage({ type: "overload", slow: this.windowSlow, hops: this.windowHops, slowMs: SLOW_MS, coarseTimer: !HAS_PERF });
      }
      const hn = this.histCount;
      let p50Ms = 0;
      let p95Ms = 0;
      let p99Ms = 0;
      if (hn > 0) {
        this._sortHistScratch(hn);
        const s = this.histScratch;
        p50Ms = s[Math.min(hn - 1, Math.floor((hn - 1) * 0.5))];
        p95Ms = s[Math.min(hn - 1, Math.floor((hn - 1) * 0.95))];
        p99Ms = s[Math.min(hn - 1, Math.floor((hn - 1) * 0.99))];
      }
      const liveNow = !this.bypass && !this.forcedBypass && !this.overloaded && this.refillHops === 0;
      this.port.postMessage({
        type: "stats",
        hops: this.hops,
        slow: this.slow,
        meanMs: this.hops ? this.sumMs / this.hops : 0,
        p50Ms,
        p95Ms,
        p99Ms,
        maxMs: this.maxMs,
        pTarget: this.wasm ? this.wasm.oto_p_target() : 0,
        bypass: !liveNow,
        overloaded: this.overloaded,
        slowMs: SLOW_MS,
        coarseTimer: !HAS_PERF,
        timer: HAS_PERF ? "performance.now" : "Date.now (1 ms steps)",
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
