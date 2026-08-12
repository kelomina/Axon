# JavaScript Loop151 caller

This Windows x64 Node.js example uses Koffi to call the frozen KVD ABI. It
checks the native `kvd_config` size, validates the bundled native runtime, scans with both
`kvd_scan_path` and `kvd_scan_bytes`, and releases returned buffers with
`kvd_free`.

```powershell
npm install
node .\axon_loop151_call.js `
  --dll "..\..\build\bin\Release\axon_loop151_champion.dll" `
  --runtime-config "..\..\..\..\dist\axon_loop151_native_20260717\runtime\loop151_native_runtime.json" `
  --sample "C:\samples\sample.exe"
```

`stage2_model_json_path` must point to `runtime/loop151_native_runtime.json`, not a
Loop28 Stage-2 HGB JSON. The native runtime config resolves all bundled model
weights relative to its own directory. Use `--allowed-scan-root` for a physical root when
the sample path traverses a directory link.
