RUSTFLAGS="-C target-cpu=native" cargo build --release
cp target/release/rust-simd-samples ./input_data

# run in a subshell
(cd ./input_data && ./rust-simd-samples)
