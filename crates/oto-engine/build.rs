//! Compiles in the HAL driver revision this engine ships with (`driver/macos/DRIVER_VERSION`),
//! the single source of truth also used by the driver's Makefile.

fn main() {
    let path = "../../driver/macos/DRIVER_VERSION";
    println!("cargo:rerun-if-changed={path}");
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let v: u32 = text
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{path} must hold a positive integer: {e}"));
    assert!(v > 0, "{path} must hold a positive integer");
    println!("cargo:rustc-env=OTO_DRIVER_VERSION={v}");
}
