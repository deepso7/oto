// Shared by bench-worker.js (browser) and bench-node.mjs (Node/V8): times oto_process_hop on a
// wasm instance. Works as a classic worker script (importScripts) and as an ES module import.

const HOP = 512;
const HOP_MS = (HOP / 48000) * 1000;

function otoInstance(module, { arch = 1, seed = 1, weights = null } = {}) {
  const w = new WebAssembly.Instance(module, {}).exports;
  let rc;
  if (weights) {
    const b = new Uint8Array(weights);
    const p = w.oto_alloc(b.length);
    new Uint8Array(w.memory.buffer, p, b.length).set(b);
    rc = w.oto_init(p, b.length);
    w.oto_free(p, b.length);
  } else {
    rc = w.oto_init_random(arch, seed);
  }
  if (rc !== 0) {
    const m = new Uint8Array(w.memory.buffer, w.oto_error_ptr(), w.oto_error_len());
    throw new Error("oto_init failed: " + new TextDecoder().decode(m));
  }
  return {
    w,
    input: new Float32Array(w.memory.buffer, w.oto_input_ptr(), HOP),
    output: new Float32Array(w.memory.buffer, w.oto_output_ptr(), HOP),
  };
}

function testSignal(hops) {
  const x = new Float32Array(hops * HOP);
  let s = 0x12345678;
  for (let i = 0; i < x.length; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    x[i] = 0.1 * (s / 4294967296 - 0.5) + 0.2 * Math.sin((2 * Math.PI * 220 * i) / 48000);
  }
  return x;
}

function pct(sorted, p) {
  return sorted[Math.round((sorted.length - 1) * p)];
}

function summary(times) {
  const s = Float64Array.from(times).sort();
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  return { n: s.length, mean, p50: pct(s, 0.5), p95: pct(s, 0.95), p99: pct(s, 0.99), max: s[s.length - 1] };
}

// Racing: hops back to back. Mean from total wall time (exact even with a coarse timer).
function racing(inst, signal, hops) {
  const { w, input } = inst;
  const n = signal.length / HOP;
  const times = new Array(hops);
  const t0 = performance.now();
  for (let h = 0; h < hops; h++) {
    input.set(signal.subarray((h % n) * HOP, (h % n + 1) * HOP));
    const a = performance.now();
    w.oto_process_hop();
    times[h] = performance.now() - a;
  }
  const total = performance.now() - t0;
  return { ...summary(times), mean: total / hops };
}

// Paced: one hop per 10.67 ms of wall time, like the audio thread (cold caches, clock scaling).
function paced(inst, signal, seconds, sleep) {
  const { w, input } = inst;
  const n = signal.length / HOP;
  const hops = Math.round((seconds * 1000) / HOP_MS);
  const times = [];
  let next = performance.now();
  return (async () => {
    for (let h = 0; h < hops; h++) {
      next += HOP_MS;
      input.set(signal.subarray((h % n) * HOP, (h % n + 1) * HOP));
      const a = performance.now();
      w.oto_process_hop();
      times.push(performance.now() - a);
      const wait = next - performance.now();
      if (wait > 0) await sleep(wait);
    }
    return summary(times);
  })();
}

async function benchAll(module, opts, log) {
  const signal = testSignal(200);
  const inst = otoInstance(module, opts);
  const warm = opts.warmup ?? 300;
  const tw = performance.now();
  racing(inst, signal, warm);
  const warmMs = performance.now() - tw;
  inst.w.oto_reset();
  const r = racing(inst, signal, opts.racingHops ?? 3000);
  log && log({ phase: "racing", r });
  const sleep = opts.sleep || ((ms) => new Promise((res) => setTimeout(res, ms)));
  const p = await paced(inst, signal, opts.pacedSeconds ?? 5, sleep);
  return {
    simd: inst.w.oto_simd(),
    params: inst.w.oto_params(),
    macs: inst.w.oto_macs_per_hop(),
    warmup: { hops: warm, ms: warmMs },
    racing: r,
    paced: p,
  };
}

if (typeof module !== "undefined") module.exports = { HOP, HOP_MS, otoInstance, testSignal, racing, paced, benchAll, summary };
if (typeof globalThis !== "undefined") Object.assign(globalThis, { OtoBench: { HOP, HOP_MS, otoInstance, testSignal, racing, paced, benchAll, summary } });
