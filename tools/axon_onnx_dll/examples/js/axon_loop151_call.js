"use strict";

const fs = require("node:fs");
const path = require("node:path");
const koffi = require("koffi");

function argument(name) {
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1] : undefined;
}

function usage() {
  console.error("Usage: node axon_loop151_call.js --dll <dll> --runtime-config <json> --sample <pe>");
  process.exitCode = 2;
}

function utf8Pointer(value) {
  return Buffer.from(`${value}\0`, "utf8");
}

function outputValue(holder) {
  if (holder && Object.prototype.hasOwnProperty.call(holder, "value")) return holder.value;
  if (Array.isArray(holder)) return holder[0];
  return holder;
}

function readDllJson(library, pointerHolder, lengthHolder) {
  const pointer = outputValue(pointerHolder);
  const length = Number(outputValue(lengthHolder) || 0);
  if (!pointer || length < 0) throw new Error("DLL returned an empty JSON buffer");
  try {
    return JSON.parse(Buffer.from(pointer).subarray(0, length).toString("utf8"));
  } finally {
    library.kvd_free(pointer);
  }
}

function main() {
  if (process.platform !== "win32" || process.arch !== "x64") throw new Error("This example requires Windows x64");
  const dllPath = argument("--dll");
  const runtimePath = argument("--runtime-config");
  const samplePath = argument("--sample");
  const allowedRoot = argument("--allowed-scan-root");
  if (!dllPath || !runtimePath || !samplePath) return usage();
  for (const required of [dllPath, runtimePath, samplePath]) {
    if (!fs.existsSync(required)) throw new Error(`Missing input: ${required}`);
  }

  const KvdConfig = koffi.struct("kvd_config", {
    model_path: "char *", model_normal_path: "char *", model_packed_path: "char *",
    family_classifier_json_path: "char *", allowed_scan_root: "char *", max_file_size: "uint",
    prediction_threshold: "float", onnx_model_path: "char *", onnx_model_normal_path: "char *",
    onnx_model_packed_path: "char *", stage2_model_json_path: "char *",
    archive_scanner_path: "char *", scan_nested: "int"
  });
  if (koffi.sizeof(KvdConfig) !== 96) throw new Error(`Unexpected kvd_config size: ${koffi.sizeof(KvdConfig)}`);

  const library = koffi.load(path.resolve(dllPath));
  const create = library.func("void * __cdecl kvd_create(const kvd_config *)");
  const destroy = library.func("void __cdecl kvd_destroy(void *)");
  const validate = library.func("int __cdecl kvd_validate_models(const kvd_config *, char **, size_t *)");
  const scanPath = library.func("int __cdecl kvd_scan_path(void *, const char *, char **, size_t *)");
  const scanBytes = library.func("int __cdecl kvd_scan_bytes(void *, const unsigned char *, size_t, char **, size_t *)");
  library.kvd_free = library.func("void __cdecl kvd_free(char *)");

  // Keep the backing buffers alive while native code reads these pointers.
  const runtimeBuffer = utf8Pointer(path.resolve(runtimePath));
  const rootBuffer = allowedRoot ? utf8Pointer(path.resolve(allowedRoot)) : null;
  const config = {
    model_path: null, model_normal_path: null, model_packed_path: null,
    family_classifier_json_path: null, allowed_scan_root: rootBuffer, max_file_size: 0,
    prediction_threshold: 0, onnx_model_path: null, onnx_model_normal_path: null,
    onnx_model_packed_path: null, stage2_model_json_path: runtimeBuffer,
    archive_scanner_path: null, scan_nested: 0
  };

  const validationPointer = {};
  const validationLength = {};
  const validationCode = validate(config, validationPointer, validationLength);
  const validation = readDllJson(library, validationPointer, validationLength);
  if (validationCode !== 0) throw new Error(`kvd_validate_models failed: ${JSON.stringify(validation)}`);
  const handle = create(config);
  if (!handle) throw new Error("kvd_create returned a null handle");

  try {
    const pathPointer = {};
    const pathLength = {};
    const pathCode = scanPath(handle, utf8Pointer(path.resolve(samplePath)), pathPointer, pathLength);
    const pathResult = readDllJson(library, pathPointer, pathLength);
    if (pathCode !== 0) throw new Error(`kvd_scan_path failed: ${JSON.stringify(pathResult)}`);

    const bytes = fs.readFileSync(samplePath);
    const bytesPointer = {};
    const bytesLength = {};
    const bytesCode = scanBytes(handle, bytes, bytes.length, bytesPointer, bytesLength);
    const bytesResult = readDllJson(library, bytesPointer, bytesLength);
    if (bytesCode !== 0) throw new Error(`kvd_scan_bytes failed: ${JSON.stringify(bytesResult)}`);
    if (pathResult.loop_id !== "Loop151" || bytesResult.loop_id !== "Loop151") throw new Error("Response is not tagged Loop151");
    console.log(JSON.stringify({
      schema: "axon_loop151_js_call_example_v1", calling_convention: "__cdecl",
      kvd_config_x64_size: koffi.sizeof(KvdConfig), path_result: pathResult,
      bytes_result: bytesResult, same_prediction: pathResult.prediction === bytesResult.prediction
    }, null, 2));
  } finally {
    destroy(handle);
  }
}

try { main(); } catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
