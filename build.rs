use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::process::Command;

fn main() {
    // Read genesis phrase from environment at build time
    let phrase = env::var("YP_GENESIS_PHRASE")
        .expect("YP_GENESIS_PHRASE must be set at build time");

    // Security: refuse to build if the phrase is empty (would produce a
    // trivially guessable or zero seed).
    assert!(
        !phrase.is_empty(),
        "YP_GENESIS_PHRASE must be a non-empty string"
    );

    // Compute seed hash
    let mut hasher = DefaultHasher::new();
    phrase.hash(&mut hasher);
    let seed = hasher.finish();

    // Refuse to build with a zero seed (indicates a hashing failure).
    assert!(seed != 0, "YP_GENESIS_PHRASE produced a zero seed");

    // Write seed as a Rust constant in OUT_DIR
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let seed_path = Path::new(&out_dir).join("genesis_seed.rs");
    fs::write(&seed_path, format!("0x{:016x}u64", seed))
        .expect("Failed to write genesis seed");

    println!("cargo:rerun-if-env-changed=YP_GENESIS_PHRASE");

    // Security: verify the genesis phrase does not appear in any source file.
    // This prevents accidental leakage of the build-time secret into the
    // compiled binary or repository.
    let output = Command::new("grep")
        .args(["-r", &phrase, "src/"])
        .output()
        .expect("grep failed");
    if output.status.success() {
        panic!("SECURITY VIOLATION: genesis phrase found in source files");
    }
}
