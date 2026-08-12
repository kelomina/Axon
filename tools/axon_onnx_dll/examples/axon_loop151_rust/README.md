# Loop151 Rust caller

This Windows x64 example dynamically loads `axon_loop151_champion.dll` and
calls the frozen KVD C ABI. It performs both `kvd_scan_path` and
`kvd_scan_bytes`, releases each returned JSON buffer with `kvd_free`, and then
destroys the predictor with `kvd_destroy`.

The Rust declarations use `#[repr(C)]` and match the header's `kvd_config`
layout. `std::mem::size_of::<KvdConfig>()` is checked at runtime and must be
`96` on x64. Rust's `extern "C"` function pointers match the header's Windows
`KVD_CALL` (`__cdecl`) ABI; no exported function signature is changed.

Build and run from this directory:

```powershell
cargo build --release
cargo run --release -- `
  --dll "..\..\build\bin\Release\axon_loop151_champion.dll" `
  --runtime-config "..\..\..\..\dist\axon_loop151_native_20260717\runtime\loop151_native_runtime.json" `
  --sample "C:\samples\sample.exe"
```

The native runtime configuration resolves all bundled model weights relative
to its own directory; it does not start Python or load a bridge process. When
a path scan is constrained to a physical root, add
`--allowed-scan-root "C:\samples"`. Passing a Loop28 Stage-2 JSON is not
compatible with the Loop151 native runtime.
