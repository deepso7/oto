// Unit tests for web/demo/worklet.js (zero dependencies; run with `node --test web/demo/worklet.test.mjs`).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const WORKLET_SRC = readFileSync(join(here, "worklet.js"), "utf8");
const HOP = 512;
const DELAY = 2 * HOP;

/**
 * Loads worklet.js in an isolated AudioWorkletGlobalScope-like VM with a controllable clock,
 * sampleRate, and a mock WebAssembly.Instance implementing a 1-hop (512-sample) identity delay
 * (matching Oto-1's 512-sample STFT overlap delay on top of the worklet's 512-sample hop ring,
 * for 1024 samples total).
 */
function loadWorklet({ sampleRate = 48000, hasPerf = true, hopCostMs = 0.5, wasmOverrides = null } = {}) {
  let clockMs = 0;
  let currentTime = 0;
  let registered = null;
  const mem = new WebAssembly.Memory({ initial: 2 });
  const IN_PTR = 1024;
  const OUT_PTR = IN_PTR + HOP * 4;
  const prevHop = new Float32Array(HOP);
  let processCalls = 0;
  let resetCalls = 0;
  const seenHops = [];

  class FakeModule {
    constructor(bytes) {
      this.bytes = bytes;
    }
  }
  class FakeInstance {
    constructor(mod) {
      if (wasmOverrides && wasmOverrides.onInstantiate) {
        wasmOverrides.onInstantiate(mod);
      }
      const inView = () => new Float32Array(mem.buffer, IN_PTR, HOP);
      const outView = () => new Float32Array(mem.buffer, OUT_PTR, HOP);
      const exports = {
        memory: mem,
        oto_alloc: () => 0,
        oto_free: () => {},
        oto_init: () => 0,
        oto_init_random: () => 0,
        oto_input_ptr: () => IN_PTR,
        oto_output_ptr: () => OUT_PTR,
        oto_process_hop: () => {
          processCalls++;
          clockMs += typeof hopCostMs === "function" ? hopCostMs(processCalls) : hopCostMs;
          const inp = inView();
          const out = outView();
          seenHops.push(Float32Array.from(inp));
          // Tag wet samples as 1000 + delayed sample so tests distinguish wet from dry unambiguously.
          for (let i = 0; i < HOP; i++) {
            out[i] = prevHop[i] === 0 ? 0 : prevHop[i] + 1000;
            prevHop[i] = inp[i];
          }
          return 0;
        },
        oto_reset: () => {
          resetCalls++;
          prevHop.fill(0);
          outView().fill(0);
        },
        oto_simd: () => 1,
        oto_params: () => 373817,
        oto_macs_per_hop: () => 1508480,
        oto_p_target: () => 0.75,
        oto_error_ptr: () => 0,
        oto_error_len: () => 0,
        ...(wasmOverrides && wasmOverrides.exports ? wasmOverrides.exports : {}),
      };
      this.exports = exports;
    }
  }

  class AudioWorkletProcessor {
    constructor() {
      const msgs = [];
      this.port = {
        onmessage: null,
        postMessage: (m) => msgs.push(m),
        _messages: msgs,
      };
    }
  }

  const sandbox = {
    AudioWorkletProcessor,
    registerProcessor: (name, cls) => {
      registered = { name, cls };
    },
    WebAssembly: { Module: FakeModule, Instance: FakeInstance },
    Float32Array,
    Uint8Array,
    Math,
    Error,
    TypeError,
    String,
    Date: { now: () => Math.floor(clockMs) },
    sampleRate,
  };
  if (hasPerf) {
    sandbox.performance = { now: () => clockMs };
  }
  Object.defineProperty(sandbox, "currentTime", {
    get: () => currentTime,
    enumerable: true,
  });
  sandbox.globalThis = sandbox;

  vm.runInNewContext(WORKLET_SRC, sandbox, { filename: "worklet.js" });
  assert.ok(registered && registered.name === "oto-processor");

  return {
    Processor: registered.cls,
    FakeModule,
    advanceTime: (s) => {
      currentTime += s;
    },
    stats: () => ({ processCalls, resetCalls, seenHops }),
  };
}

function runStream(proc, input, quantumSizes) {
  const out = new Float32Array(input.length);
  let i = 0;
  let qIdx = 0;
  while (i < input.length) {
    const q = Math.min(quantumSizes[qIdx++ % quantumSizes.length], input.length - i);
    const inBuf = input.subarray(i, i + q);
    const outBuf = new Float32Array(q);
    proc.process([[inBuf]], [[outBuf]]);
    out.set(outBuf, i);
    i += q;
  }
  return out;
}

test("arbitrary render quantum sizes crossing 512-sample boundaries match 128-sample output and 1024-sample delay", () => {
  const N = 4096;
  const input = new Float32Array(N);
  for (let i = 0; i < N; i++) input[i] = i + 1;

  // Reference: standard 128-sample quantum (live wet path).
  const refEnv = loadWorklet();
  const refProc = new refEnv.Processor({ processorOptions: { bytes: new ArrayBuffer(8), warmup: 0 } });
  const refOut = runStream(refProc, input, [128]);

  // First 1024 samples are pipeline pre-roll (zeros); from 1024 onward wet output is input[i - 1024] + 1000.
  for (let i = 0; i < DELAY; i++) assert.equal(refOut[i], 0, `pre-roll sample ${i}`);
  for (let i = DELAY; i < N; i++) assert.equal(refOut[i], input[i - DELAY] + 1000, `wet sample ${i}`);

  // Non-128 and boundary-crossing quantum patterns must produce bit-identical output.
  for (const pattern of [[256], [512], [100], [384], [1000], [1], [127, 389, 73, 513]]) {
    const env = loadWorklet();
    const proc = new env.Processor({ processorOptions: { bytes: new ArrayBuffer(8), warmup: 0 } });
    const got = runStream(proc, input, pattern);
    assert.deepEqual(got, refOut, `quantum pattern ${JSON.stringify(pattern)} diverged from 128-sample reference`);
  }

  // Bypass path keeps the exact same 1024-sample dry delay across odd quanta.
  const dryEnv = loadWorklet();
  const dryProc = new dryEnv.Processor({ processorOptions: { bytes: new ArrayBuffer(8), warmup: 0, forcedBypass: true } });
  const dryOut = runStream(dryProc, input, [100, 384, 73]);
  for (let i = 0; i < DELAY; i++) assert.equal(dryOut[i], 0);
  for (let i = DELAY; i < N; i++) assert.equal(dryOut[i], input[i - DELAY]);
});

test("mid-hop unoverload preserves buffered input, holds dry audio for 2 refill hops, then resumes wet output", () => {
  // First 10 hops are slow (3 ms > 2 ms) to trip overload; recovery hops are fast (0.3 ms).
  const env = loadWorklet({ hopCostMs: (call) => (call <= 10 ? 3.0 : 0.3) });
  const proc = new env.Processor({ processorOptions: { bytes: new ArrayBuffer(8), warmup: 0 } });

  // Run 4 hops (2048 samples) with 128-sample quanta and advance currentTime >= 1s to trip overload.
  let sampleVal = 1;
  const push = (n) => {
    const inBuf = new Float32Array(n);
    for (let i = 0; i < n; i++) inBuf[i] = sampleVal++;
    const outBuf = new Float32Array(n);
    proc.process([[inBuf]], [[outBuf]]);
    return { inBuf, outBuf };
  };

  for (let q = 0; q < 15; q++) push(128);
  env.advanceTime(1.1);
  push(128); // completes hop 4 and triggers the 1-second stats + overload check

  const msgs = proc.port._messages;
  const overloadMsg = msgs.find((m) => m.type === "overload");
  const statsAfterOverload = msgs.filter((m) => m.type === "stats").at(-1);
  assert.ok(overloadMsg, "expected overload message");
  assert.equal(statsAfterOverload.overloaded, true);
  assert.equal(statsAfterOverload.bypass, true, "stats must report bypass=true immediately after overload");

  // Buffer 128 samples mid-hop while overloaded (distinct values in sampleVal..sampleVal+127).
  const midHopStart = sampleVal;
  const mid = push(128);
  // While overloaded, output is matched-delay dry audio (no +1000 wet tag).
  assert.equal(mid.outBuf[0], midHopStart - DELAY);

  // Trigger unoverload mid-hop (pos === 128).
  proc.port.onmessage({ data: { type: "unoverload" } });
  assert.equal(proc.overloaded, false);
  assert.equal(proc.refillHops, 2);

  // Finish the remaining 384 samples of the first recovery hop across a boundary-crossing quantum (200 + 184).
  const r1a = push(200);
  const r1b = push(184);
  // Still dry during refill hop 1 (refillHops drops 2 -> 1 at end of r1b).
  assert.equal(r1a.outBuf[0], midHopStart + 128 - DELAY);
  assert.equal(r1b.outBuf[0], midHopStart + 328 - DELAY);
  assert.equal(proc.refillHops, 1);

  // Verify the first recovery hop actually received the 128 pre-unoverload samples intact (not zeroed).
  const firstRecoveryHop = env.stats().seenHops.at(-1);
  for (let i = 0; i < 128; i++) {
    assert.equal(firstRecoveryHop[i], midHopStart + i, `mid-hop sample ${i} dropped on unoverload`);
  }

  // Second recovery hop (512 samples) must ALSO stay dry until the hop finishes (refillHops 1 -> 0).
  const r2Start = sampleVal;
  const r2 = push(512);
  for (let i = 0; i < 512; i++) {
    assert.equal(r2.outBuf[i], r2Start + i - DELAY, `refill hop 2 switched to wet too early at sample ${i}`);
  }
  assert.equal(proc.refillHops, 0);

  // Next quantum immediately switches to wet output carrying the preserved mid-hop samples!
  const wetStart = sampleVal;
  const wet = push(256);
  for (let i = 0; i < 256; i++) {
    assert.equal(wet.outBuf[i], wetStart + i - DELAY + 1000, `expected wet output at sample ${i} after 2 refill hops`);
  }
});

test("non-48 kHz sampleRate stays in matched-delay bypass and runs zero model hops", () => {
  const env = loadWorklet({ sampleRate: 44100 });
  const proc = new env.Processor({ processorOptions: { bytes: new ArrayBuffer(8), warmup: 0 } });
  const err = proc.port._messages.find((m) => m.type === "error");
  assert.ok(err && err.code === "sampleRate");
  assert.equal(proc.bypass, true);

  const input = new Float32Array(2048);
  for (let i = 0; i < input.length; i++) input[i] = i + 1;
  const out = runStream(proc, input, [128]);
  assert.equal(env.stats().processCalls, 0);
  for (let i = 0; i < DELAY; i++) assert.equal(out[i], 0);
  for (let i = DELAY; i < input.length; i++) assert.equal(out[i], input[i - DELAY]);
});

test("module instantiation failure falls back to bytes, and late init error stays in bypass", () => {
  let attempts = 0;
  const env = loadWorklet({
    wasmOverrides: {
      onInstantiate: (mod) => {
        attempts++;
        if (attempts === 1 && mod.tag === "bad-module") throw new Error("clone failed");
      },
    },
  });
  const badMod = new env.FakeModule(null);
  badMod.tag = "bad-module";
  const proc = new env.Processor({ processorOptions: { module: badMod, bytes: new ArrayBuffer(8), warmup: 0 } });
  assert.equal(proc.bypass, false);
  assert.equal(attempts, 2);
  assert.ok(proc.port._messages.some((m) => m.type === "ready"));

  // Late failure during ready export read leaves bypass=true and process() safe.
  const badExportEnv = loadWorklet({
    wasmOverrides: {
      exports: {
        oto_macs_per_hop: () => {
          throw new Error("missing export");
        },
      },
    },
  });
  const badProc = new badExportEnv.Processor({ processorOptions: { bytes: new ArrayBuffer(8), warmup: 0 } });
  assert.equal(badProc.bypass, true);
  assert.ok(badProc.port._messages.some((m) => m.type === "error"));
  const out = runStream(badProc, new Float32Array(1024).fill(0.5), [128]);
  assert.equal(out.length, 1024);
});

test("zero-alloc heapsort percentiles match reference sort and leave hist untouched", () => {
  const env = loadWorklet({ hopCostMs: (call) => ((call * 37) % 101) / 100 });
  const proc = new env.Processor({ processorOptions: { bytes: new ArrayBuffer(8), warmup: 0, disableOverload: true } });

  const input = new Float32Array(128);
  const out = new Float32Array(128);
  // 25 hops -> 25 timings in hist[0..25).
  for (let q = 0; q < 100; q++) {
    if (q === 99) env.advanceTime(1.1);
    proc.process([[input]], [[out]]);
  }
  const histBefore = Float32Array.from(proc.hist.subarray(0, proc.histCount));
  const sorted = Float32Array.from(histBefore).sort();
  const n = sorted.length;
  const stats = proc.port._messages.filter((m) => m.type === "stats").at(-1);
  assert.equal(n, 25);
  assert.equal(stats.p50Ms, sorted[Math.floor((n - 1) * 0.5)]);
  assert.equal(stats.p95Ms, sorted[Math.floor((n - 1) * 0.95)]);
  assert.equal(stats.p99Ms, sorted[Math.floor((n - 1) * 0.99)]);
  assert.deepEqual(proc.hist.subarray(0, n), histBefore);
});
