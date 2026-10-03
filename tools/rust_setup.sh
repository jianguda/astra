#!/bin/bash
# Rust toolchains for the tests of RustEvo: rustup, the stable toolchains 1.71.0-1.91.0 (a sample is compiled with the
# toolchain of its target version, addressed as `1.x.0`) and nightly (for tests that enable unstable features).
# Set RUSTUP_DIST_SERVER / RUSTUP_UPDATE_ROOT beforehand to use a mirror.
set -e
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o rustup-init.sh
  sh rustup-init.sh -y --profile minimal --default-toolchain none
  rm -f rustup-init.sh
  source "$HOME/.cargo/env"
fi
for v in $(seq 71 91); do rustup toolchain install 1.$v.0 --profile minimal || echo "FAILED 1.$v.0"; done
rustup toolchain install nightly --profile minimal || echo "FAILED nightly"
rustup default 1.84.0
rustup toolchain list
