#!/bin/sh
# Builds the two browser binaries of oto-wasm into web/demo/wasm/:
#   oto-simd.wasm    wasm32-unknown-unknown +simd128 kernels
#   oto-scalar.wasm  wasm32-unknown-unknown, no SIMD (the fallback WebAssembly.validate picks)
# then lists their imports (must be none) and sizes.
#
# The toolchain needs the wasm32-unknown-unknown target. rust-toolchain.toml pins a toolchain
# that may not have it; set OTO_WASM_TOOLCHAIN (e.g. `stable`) to use another one, or add it with
# `rustup target add wasm32-unknown-unknown --toolchain <pinned>`.
set -eu
cd "$(dirname "$0")/../.."
tc=${OTO_WASM_TOOLCHAIN:+"+$OTO_WASM_TOOLCHAIN"}
out=web/demo/wasm
mkdir -p "$out"

if ! rustc ${tc:-} --print target-libdir --target wasm32-unknown-unknown >/dev/null 2>&1 \
    || [ ! -d "$(rustc ${tc:-} --print target-libdir --target wasm32-unknown-unknown)" ]; then
    echo "error: toolchain ${OTO_WASM_TOOLCHAIN:-(pinned)} has no wasm32-unknown-unknown target." >&2
    echo "       rustup target add wasm32-unknown-unknown, or OTO_WASM_TOOLCHAIN=<toolchain with it>" >&2
    exit 1
fi

# Fat LTO and one codegen unit: smaller and faster wasm than the workspace's thin-LTO default.
export CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 CARGO_PROFILE_RELEASE_OPT_LEVEL=3

build() { # $1 = name, $2 = RUSTFLAGS
    RUSTFLAGS="$2" cargo ${tc:-} build -p oto-wasm --release --target wasm32-unknown-unknown \
        --target-dir "target/wasm-$1"
    cp "target/wasm-$1/wasm32-unknown-unknown/release/oto_wasm.wasm" "$out/oto-$1.wasm"
}
build simd "-C target-feature=+simd128"
build scalar ""

for f in "$out"/oto-*.wasm; do
    gz=$(gzip -9 -c "$f" | wc -c | tr -d ' ')
    br=$(node -e 'const z=require("zlib");console.log(z.brotliCompressSync(require("fs").readFileSync(process.argv[1]),{params:{[z.constants.BROTLI_PARAM_QUALITY]:11}}).length)' "$f" 2>/dev/null || echo "n/a")
    echo "$f: $(wc -c <"$f" | tr -d ' ') bytes raw, $gz gzip -9, $br brotli -q11"
done
if command -v node >/dev/null; then
    node -e '
const fs = require("fs");
for (const f of process.argv.slice(1)) {
  const m = new WebAssembly.Module(fs.readFileSync(f));
  const imports = WebAssembly.Module.imports(m);
  console.log(f + ": imports " + JSON.stringify(imports) + ", exports " + WebAssembly.Module.exports(m).map(e => e.name).join(" "));
  if (imports.length) process.exit(1);
}' "$out"/oto-*.wasm
fi
