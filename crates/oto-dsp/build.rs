//! Links clang's runtime when the `onnx` feature pulls in ONNX Runtime on macOS: pyke's
//! prebuilt static library calls `__isPlatformVersionAtLeast` (from `@available` checks in its
//! CoreML code), which lives in `libclang_rt.osx.a` and rustc does not link by default.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let onnx = std::env::var_os("CARGO_FEATURE_ONNX").is_some();
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if !(onnx && macos) {
        return;
    }
    let out = std::process::Command::new("xcrun")
        .args(["clang", "--print-resource-dir"])
        .output()
        .expect("xcrun clang --print-resource-dir");
    let dir = String::from_utf8(out.stdout).expect("utf-8 path");
    println!("cargo:rustc-link-search=native={}/lib/darwin", dir.trim());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}
