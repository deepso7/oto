// Oto-1 browser demo: picks the SIMD or scalar build, benchmarks it (Worker + OfflineAudioContext)
// and runs it live on the microphone through worklet.js.

// Smallest module using a v128 instruction (wasm-feature-detect's SIMD probe).
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
const HOP_MS = (512 / 48000) * 1000;

const $ = (id) => document.getElementById(id);
const out = $("log");
function log(...a) {
  out.textContent += a.map((x) => (typeof x === "string" ? x : JSON.stringify(x, null, 1))).join(" ") + "\n";
  out.scrollTop = out.scrollHeight;
}
const f = (x) => x.toFixed(3);

function simdSupported() {
  return WebAssembly.validate(SIMD_PROBE);
}

function buildName() {
  const want = $("build").value;
  if (want === "auto") return simdSupported() ? "simd" : "scalar";
  return want;
}

const cache = {};
async function loadWasm(name) {
  if (!cache[name]) {
    const bytes = await (await fetch(`wasm/oto-${name}.wasm`)).arrayBuffer();
    cache[name] = { bytes, module: await WebAssembly.compile(bytes) };
  }
  return cache[name];
}

async function weights() {
  const file = $("weights").files[0];
  return file ? await file.arrayBuffer() : null;
}

function opts() {
  return { arch: Number($("arch").value), seed: 1 };
}

function fmt(name, s) {
  return `${name}: mean ${f(s.mean)} ms, p50 ${f(s.p50)}, p95 ${f(s.p95)}, p99 ${f(s.p99)}, max ${f(s.max)} (n=${s.n})`;
}

async function benchWorker() {
  const name = buildName();
  const { bytes } = await loadWasm(name);
  const w = await weights();
  const pacedSeconds = Number($("paced").value);
  log(`\n== Worker kernel benchmark (not AudioWorklet): ${name} build, arch ${$("arch").value}${w ? " (.otow file)" : " (random weights)"}, ${pacedSeconds} s paced`);
  log(`   hop = 512 samples = ${f(HOP_MS)} ms; quantum ceiling 2.667 ms; Worker kernel reference: racing mean <= 0.5 ms, paced p99 <= 2.0 ms`);
  log(`   performance.now() resolution here is coarsened unless cross-origin isolated (Chrome ~0.1 ms, Firefox 1 ms); racing mean uses total wall time`);
  const worker = new Worker("bench-worker.js");
  const result = await new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      if (e.data.type === "progress") log("   " + fmt("racing", e.data.r) + " ... pacing");
      if (e.data.type === "result") resolve(e.data.result);
      if (e.data.type === "error") reject(new Error(e.data.message));
    };
    worker.postMessage({ bytes, weights: w, ...opts(), racingHops: 3000, pacedSeconds });
  });
  worker.terminate();
  const r = result;
  log(`   simd=${r.simd} params=${r.params} MAC/hop=${r.macs} (${f((r.macs / HOP_MS) / 1e3)} GMAC/s needed); warm-up ${r.warmup.hops} hops in ${f(r.warmup.ms)} ms`);
  log("   " + fmt("kernel racing", r.racing));
  log("   " + fmt("kernel paced ", r.paced));
  log(`   kernel timings (Worker): racing mean ${r.racing.mean <= 0.5 ? "PASS" : "FAIL"}, paced p99 ${r.paced.p99 <= 2.0 ? "PASS" : "FAIL"} (run Live Mic for the AudioWorklet render-thread gate)`);
}

// Runs the real worklet (same processor as live) in an OfflineAudioContext: per-hop cost including
// the JS ring buffer and render-quantum plumbing.
async function benchOffline() {
  const name = buildName();
  const wasm = await loadWasm(name);
  const seconds = 20;
  const ctx = new OfflineAudioContext(1, 48000 * seconds, 48000);
  await ctx.audioWorklet.addModule("worklet.js");
  const buf = ctx.createBuffer(1, 48000 * seconds, 48000);
  const d = buf.getChannelData(0);
  for (let i = 0; i < d.length; i++) d[i] = 0.1 * (Math.random() - 0.5) + 0.2 * Math.sin((2 * Math.PI * 220 * i) / 48000);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  const node = makeNode(ctx, wasm, await weights(), { disableOverload: true });
  const ready = new Promise((res) => (node.port.onmessage = (e) => { if (e.data.type === "ready" || e.data.type === "error") res(e.data); }));
  src.connect(node).connect(ctx.destination);
  src.start();
  const t0 = performance.now();
  const rendered = await ctx.startRendering();
  const ms = performance.now() - t0;
  const r = await ready;
  if (r.type === "error") return log("worklet error: " + r.message);
  const hops = (48000 * seconds) / 512;
  log(`\n== OfflineAudioContext throughput (unpaced) through worklet.js: ${name} build, arch ${$("arch").value}`);
  log(`   ${seconds} s of audio in ${f(ms)} ms (incl. ${r.warmup} warm-up hops = ${f(r.warmupMs)} ms): ${f((ms - r.warmupMs) / hops)} ms per hop all-in, ${f((48000 * seconds) / 48 / (ms - r.warmupMs))}x realtime`);
  let e = 0;
  for (const v of rendered.getChannelData(0)) e += v * v;
  log(`   output energy ${e.toExponential(3)} (non-zero = model ran)`);
}

function makeNode(ctx, wasm, w, extra = {}) {
  const base = { ...opts(), weights: w, warmup: 300, ...extra };
  const prefer = $("transfer").value;
  if (prefer === "module") {
    try {
      // bytes ride along (structured-cloned, not transferred, so the cached buffer stays usable): the
      // worklet falls back to them if the Module arrives but won't instantiate in its scope.
      return new AudioWorkletNode(ctx, "oto-processor", { processorOptions: { ...base, module: wasm.module, bytes: wasm.bytes } });
    } catch (e) {
      log(`   Module not cloneable into the worklet (${e.name}); falling back to bytes`);
    }
  }
  return new AudioWorkletNode(ctx, "oto-processor", { processorOptions: { ...base, bytes: wasm.bytes } });
}

let live = null;
let pending = null;
let liveGen = 0;

function cleanupSession(s) {
  if (!s) return;
  if (s.cancel) {
    const cancel = s.cancel;
    s.cancel = null;
    cancel();
  }
  if (s.node) {
    s.node.port.onmessage = null;
    s.node = null;
  }
  if (s.stream) {
    s.stream.getTracks().forEach((t) => t.stop());
    s.stream = null;
  }
  if (s.ctx) {
    const c = s.ctx;
    s.ctx = null;
    c.close().catch(() => {});
  }
}

async function startLive() {
  if (live || pending) return;
  const gen = ++liveGen;
  const session = { gen, stream: null, ctx: null, node: null, cancel: null };
  pending = session;
  try {
    const name = buildName();
    const wasm = await loadWasm(name);
    if (gen !== liveGen) return;
    // getUserMedia before the context: Safari may change the hardware rate when capture starts.
    session.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    if (gen !== liveGen) return;
    session.ctx = new AudioContext({ sampleRate: 48000, latencyHint: "interactive" });
    const rateBypass = session.ctx.sampleRate !== 48000;
    if (rateBypass) log(`!! context runs at ${session.ctx.sampleRate} Hz, not 48000: Oto-1 stays bypassed (matched-delay dry signal only; resampling is not implemented in the demo)`);
    await session.ctx.audioWorklet.addModule("worklet.js");
    if (gen !== liveGen) return;
    const w = await weights();
    if (gen !== liveGen) return;
    const node = makeNode(session.ctx, wasm, w, { forcedBypass: rateBypass || $("bypass").checked });
    session.node = node;
    let initReady = false;
    let initError = false;
    node.port.onmessage = (e) => {
      if (gen !== liveGen) return;
      const d = e.data;
      if (d.type === "ready") {
        initReady = true;
        log(`live: ${name} build ready, simd=${d.simd}, ${d.params} params, warm-up ${d.warmup} hops in ${f(d.warmupMs)} ms, chain delay ${d.latencySamples} samples (21.33 ms) + context/device latency`);
      }
      if (d.type === "error") {
        initError = true;
        log(d.code === "sampleRate" ? "live: Oto-1 bypassed: " + d.message : "live: worklet error, bypassing: " + d.message);
      }
      if (d.type === "overload") log(`live: OVERLOAD (${d.slow}/${d.hops} hops > ${d.slowMs} ms${d.coarseTimer ? " on the 1 ms Date.now() clock" : ""} in the last second) -> bypass`);
      if (d.type === "stats") {
        const gate = rateBypass ? "N/A (rate)" : initError || d.overloaded || (d.hops > 0 && d.p99Ms > 2.0) ? "FAIL" : (!initReady || d.hops === 0 ? "N/A" : "PASS");
        $("stats").textContent = `AudioWorklet paced: hops ${d.hops}, slow ${d.slow}, mean ${f(d.meanMs)} ms, p50 ${f(d.p50Ms)}, p95 ${f(d.p95Ms)}, p99 ${f(d.p99Ms)}, max(1 s) ${f(d.maxMs)} ms [${d.timer}] | gate(p99<=2.0 ms): ${gate} | p_target ${d.pTarget.toFixed(2)}, ${d.bypass ? "BYPASS" : "model on"}`;
      }
    };
    const src = session.ctx.createMediaStreamSource(session.stream);
    src.connect(node).connect(session.ctx.destination);
    // The context may have been created suspended (no user activation left after the awaits above).
    // Race resume() against both stopLive() and a timeout: a broken audio daemon can leave it pending forever.
    let timer = null;
    try {
      const resumeP = session.ctx.resume();
      resumeP.catch(() => {});
      await Promise.race([
        resumeP,
        new Promise((res) => { session.cancel = res; }),
        new Promise((_, rej) => { timer = setTimeout(() => rej(new Error("AudioContext.resume() timed out (no audio output device?)")), 5000); }),
      ]);
    } finally {
      clearTimeout(timer);
      session.cancel = null;
    }
    if (gen !== liveGen) return;
    // Re-sync bypass in case the checkbox was toggled while resume() was pending.
    node.port.postMessage({ type: "bypass", value: rateBypass || $("bypass").checked });
    log(`live: baseLatency ${f(session.ctx.baseLatency * 1000)} ms, outputLatency ${f((session.ctx.outputLatency || 0) * 1000)} ms, state ${session.ctx.state}`);
    live = { ctx: session.ctx, stream: session.stream, node, rateBypass };
    session.stream = null;
    session.ctx = null;
    session.node = null;
  } catch (e) {
    if (gen === liveGen) log("error: " + (e && e.message ? e.message : e));
  } finally {
    if (pending === session) pending = null;
    cleanupSession(session);
  }
}

async function stopLive() {
  const stopGen = ++liveGen;
  const pend = pending;
  pending = null;
  cleanupSession(pend);
  const cur = live;
  live = null;
  $("stats").textContent = "";
  if (cur) {
    cur.node.port.onmessage = null;
    cur.stream.getTracks().forEach((t) => t.stop());
    try {
      await cur.ctx.close();
    } catch (e) {
      if (stopGen === liveGen) log("error: " + (e && e.message ? e.message : e));
    }
  }
}

const guard = (fn) => () => fn().catch((e) => log("error: " + (e && e.message ? e.message : e)));
$("bench").onclick = guard(benchWorker);
$("offline").onclick = guard(benchOffline);
$("live").onclick = guard(startLive);
$("stop").onclick = guard(stopLive);
$("bypass").onchange = (e) => live && live.node.port.postMessage({ type: "bypass", value: live.rateBypass || e.target.checked });
log(`WebAssembly SIMD: ${simdSupported() ? "yes" : "no"}; auto build = ${simdSupported() ? "simd" : "scalar"}; UA ${navigator.userAgent}`);
