// Node/V8 (the engine Chrome uses) timing and parity for the built wasm binaries.
//   node web/demo/bench-node.mjs [pacedSeconds] [--weights <path.otow>] [--label <name>] [--build simd|scalar|both]
// Without --weights it times random Candidate A and B models (seed 1). With --weights it times that
// .otow file instead (any arch/width, e.g. a C=24 Candidate B); the row is named --label or the
// file name, and there is no parity check (the reference vectors are for the random models).
// Parity: if target/oto1/ref-*.f32 exist (`cargo run --release -p oto-dsp --no-default-features
// --features oto1 --example oto1_bench -- --write`), each build's output must be within -90 dB
// of the native scalar reference for the same random model and input.
import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join, basename } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const { otoInstance, benchAll, HOP, HOP_MS } = createRequire(import.meta.url)("./bench-core.js");
const f = (x) => x.toFixed(4);

function parseArgs(argv) {
  const a = { pacedSeconds: 5, weights: null, label: null, build: "both" };
  let positional = 0;
  for (let i = 0; i < argv.length; i++) {
    let arg = argv[i];
    let val = null;
    const eq = arg.startsWith("--") ? arg.indexOf("=") : -1;
    if (eq > 0) [arg, val] = [arg.slice(0, eq), arg.slice(eq + 1)];
    const take = () => {
      let v;
      if (val !== null) v = val;
      else {
        if (i + 1 >= argv.length) throw new Error(`${arg} needs a value`);
        if (argv[i + 1].startsWith("-")) throw new Error(`${arg} needs a value (use ${arg}=... for values starting with '-')`);
        v = argv[++i];
      }
      if (!v) throw new Error(`${arg} needs a non-empty value`);
      return v;
    };
    if (arg === "--weights") a.weights = take();
    else if (arg === "--label") a.label = take();
    else if (arg === "--build") a.build = take();
    else if (arg === "-h" || arg === "--help") a.help = true;
    else if (arg.startsWith("-") && Number.isNaN(Number(arg))) throw new Error(`unknown option ${arg}`);
    else if (positional++ === 0) a.pacedSeconds = Number(arg);
    else throw new Error(`unexpected argument ${arg}`);
  }
  const pacedHops = Math.round((a.pacedSeconds * 1000) / HOP_MS);
  if (!Number.isFinite(a.pacedSeconds) || a.pacedSeconds <= 0 || !Number.isSafeInteger(pacedHops) || pacedHops < 1) {
    throw new Error(`pacedSeconds must be a finite number yielding a safe positive hop count (>= ${(HOP_MS / 2000).toFixed(4)} s)`);
  }
  if (!["simd", "scalar", "both"].includes(a.build)) throw new Error(`--build must be simd, scalar or both (got ${a.build})`);
  return a;
}

let args;
try {
  args = parseArgs(process.argv.slice(2));
} catch (e) {
  console.error(`bench-node: ${e.message}\nusage: node web/demo/bench-node.mjs [pacedSeconds] [--weights <path.otow>] [--label <name>] [--build simd|scalar|both]`);
  process.exit(2);
}
if (args.help) {
  console.log("usage: node web/demo/bench-node.mjs [pacedSeconds] [--weights <path.otow>] [--label <name>] [--build simd|scalar|both]");
  process.exit(0);
}
const { pacedSeconds } = args;
const builds = args.build === "both" ? ["simd", "scalar"] : [args.build];
let weights = null;
if (args.weights) {
  let b;
  try {
    b = readFileSync(args.weights);
  } catch (e) {
    console.error(`bench-node: cannot read --weights ${args.weights}: ${e.message}`);
    process.exit(2);
  }
  weights = b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
}
// [arch, row name, weights]; arch is ignored by oto_init when weights are given.
const models = weights
  ? [[0, args.label || basename(args.weights), weights]]
  : [[1, "A"], [2, "B"]].map(([arch, name]) => [arch, args.label ? `${args.label}-${name}` : name, null]);

function errDb(a, b) {
  let num = 0, den = 0;
  for (let i = 0; i < a.length; i++) { num += (a[i] - b[i]) ** 2; den += b[i] ** 2; }
  return num === 0 ? -Infinity : 10 * Math.log10(num / den);
}
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };

let parityFailed = false;
for (const build of builds) {
  const module = new WebAssembly.Module(readFileSync(join(here, "wasm", `oto-${build}.wasm`)));
  for (const [arch, name, w] of models) {
    const ref = w ? null : join(root, "target", "oto1", `ref-${arch === 1 ? "a" : "b"}-out.f32`);
    let parity = w ? "custom weights, no parity" : "no reference vectors";
    if (ref && existsSync(ref)) {
      const x = f32(join(root, "target", "oto1", "ref-input.f32"));
      const want = f32(ref);
      const inst = otoInstance(module, { arch, seed: 1 });
      const got = new Float32Array(x.length);
      for (let h = 0; h < x.length / HOP; h++) {
        inst.input.set(x.subarray(h * HOP, (h + 1) * HOP));
        inst.w.oto_process_hop();
        got.set(inst.output, h * HOP);
      }
      const db = errDb(got, want);
      const ok = db <= -90;
      if (!ok) parityFailed = true;
      parity = `vs native scalar ${db.toFixed(1)} dB ${ok ? "PASS" : "FAIL"}`;
    }
    const r = await benchAll(module, { arch, seed: 1, weights: w, racingHops: 3000, pacedSeconds });
    const { racing: a, paced: p } = r;
    console.log(
      `${build.padEnd(6)} ${name} params ${r.params} MAC/hop ${r.macs} | ${parity} | warm-up ${r.warmup.hops} hops ${f(r.warmup.ms)} ms` +
        ` | racing mean ${f(a.mean)} p50 ${f(a.p50)} p99 ${f(a.p99)} max ${f(a.max)} ms` +
        ` | paced p50 ${f(p.p50)} p95 ${f(p.p95)} p99 ${f(p.p99)} max ${f(p.max)} ms`,
    );
  }
}
if (parityFailed) process.exitCode = 1;
