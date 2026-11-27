// Node/V8 (the engine Chrome uses) timing and parity for the built wasm binaries.
//   node web/demo/bench-node.mjs [pacedSeconds]
// Parity: if target/oto1/ref-*.f32 exist (`cargo run --release -p oto-dsp --no-default-features
// --features oto1 --example oto1_bench -- --write`), each build's output must be within -90 dB
// of the native scalar reference for the same random model and input.
import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const { otoInstance, benchAll, HOP } = createRequire(import.meta.url)("./bench-core.js");
const pacedSeconds = Number(process.argv[2] || 5);
const f = (x) => x.toFixed(4);

function errDb(a, b) {
  let num = 0, den = 0;
  for (let i = 0; i < a.length; i++) { num += (a[i] - b[i]) ** 2; den += b[i] ** 2; }
  return num === 0 ? -Infinity : 10 * Math.log10(num / den);
}
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };

let parityFailed = false;
for (const build of ["simd", "scalar"]) {
  const module = new WebAssembly.Module(readFileSync(join(here, "wasm", `oto-${build}.wasm`)));
  for (const [arch, name] of [[1, "a"], [2, "b"]]) {
    const ref = join(root, "target", "oto1", `ref-${name}-out.f32`);
    let parity = "no reference vectors";
    if (existsSync(ref)) {
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
    const r = await benchAll(module, { arch, seed: 1, racingHops: 3000, pacedSeconds });
    const { racing: a, paced: p } = r;
    console.log(
      `${build.padEnd(6)} ${name.toUpperCase()} params ${r.params} MAC/hop ${r.macs} | ${parity} | warm-up ${r.warmup.hops} hops ${f(r.warmup.ms)} ms` +
        ` | racing mean ${f(a.mean)} p50 ${f(a.p50)} p99 ${f(a.p99)} max ${f(a.max)} ms` +
        ` | paced p50 ${f(p.p50)} p95 ${f(p.p95)} p99 ${f(p.p99)} max ${f(p.max)} ms`,
    );
  }
}
if (parityFailed) process.exitCode = 1;
