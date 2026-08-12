//! Minimal Rust caller for the Axon Loop151 champion DLL.
//!
//! The DLL keeps the frozen KVD ABI from `axon_onnx_predict.h`. In particular,
//! `extern "C"` below is the Rust spelling of the Windows `__cdecl` ABI used by
//! the header; on Windows x64 `__cdecl` uses the platform's single x64 ABI.
//!
//! Example:
//!
//! ```text
//! cargo run --release -- \
//!   --dll ..\\..\\build\\bin\\Release\\axon_loop151_champion.dll \
//!   --runtime-config ..\\..\\..\\..\\dist\\axon_loop151_native_20260717\\runtime\\loop151_native_runtime.json \
//!   --sample C:\\samples\\sample.exe
//! ```

use libloading::{Library, Symbol};
use serde_json::{json, Value};
use std::env;
use std::error::Error;
use std::ffi::CString;
use std::fmt::{Display, Formatter};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::slice;

#[repr(C)]
struct KvdConfig {
    model_path: *const c_char,
    model_normal_path: *const c_char,
    model_packed_path: *const c_char,
    family_classifier_json_path: *const c_char,
    allowed_scan_root: *const c_char,
    max_file_size: u32,
    prediction_threshold: f32,
    onnx_model_path: *const c_char,
    onnx_model_normal_path: *const c_char,
    onnx_model_packed_path: *const c_char,
    stage2_model_json_path: *const c_char,
    archive_scanner_path: *const c_char,
    scan_nested: i32,
}

impl KvdConfig {
    fn empty() -> Self {
        Self {
            model_path: ptr::null(),
            model_normal_path: ptr::null(),
            model_packed_path: ptr::null(),
            family_classifier_json_path: ptr::null(),
            allowed_scan_root: ptr::null(),
            max_file_size: 0,
            prediction_threshold: 0.0,
            onnx_model_path: ptr::null(),
            onnx_model_normal_path: ptr::null(),
            onnx_model_packed_path: ptr::null(),
            stage2_model_json_path: ptr::null(),
            archive_scanner_path: ptr::null(),
            scan_nested: 0,
        }
    }
}

type KvdHandle = c_void;
type KvdCreate = unsafe extern "C" fn(*const KvdConfig) -> *mut KvdHandle;
type KvdDestroy = unsafe extern "C" fn(*mut KvdHandle);
type KvdScanPath =
    unsafe extern "C" fn(*mut KvdHandle, *const c_char, *mut *mut c_char, *mut usize) -> i32;
type KvdScanBytes =
    unsafe extern "C" fn(*mut KvdHandle, *const u8, usize, *mut *mut c_char, *mut usize) -> i32;
type KvdFree = unsafe extern "C" fn(*mut c_char);
type KvdValidateModels =
    unsafe extern "C" fn(*const KvdConfig, *mut *mut c_char, *mut usize) -> i32;

struct Api<'library> {
    create: Symbol<'library, KvdCreate>,
    destroy: Symbol<'library, KvdDestroy>,
    scan_path: Symbol<'library, KvdScanPath>,
    scan_bytes: Symbol<'library, KvdScanBytes>,
    free: Symbol<'library, KvdFree>,
    validate_models: Symbol<'library, KvdValidateModels>,
}

impl<'library> Api<'library> {
    unsafe fn load(library: &'library Library) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            create: library.get(b"kvd_create\0")?,
            destroy: library.get(b"kvd_destroy\0")?,
            scan_path: library.get(b"kvd_scan_path\0")?,
            scan_bytes: library.get(b"kvd_scan_bytes\0")?,
            free: library.get(b"kvd_free\0")?,
            validate_models: library.get(b"kvd_validate_models\0")?,
        })
    }
}

#[derive(Debug)]
struct ExampleError(String);

impl Display for ExampleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ExampleError {}

fn fail(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(ExampleError(message.into()))
}

fn path_c_string(path: &Path, name: &str) -> Result<CString, Box<dyn Error>> {
    let text = path
        .to_str()
        .ok_or_else(|| fail(format!("{name} is not representable as UTF-8")))?;
    CString::new(text).map_err(|_| fail(format!("{name} contains an embedded NUL")))
}

unsafe fn take_json(
    api: &Api<'_>,
    pointer: *mut c_char,
    length: usize,
) -> Result<String, Box<dyn Error>> {
    if pointer.is_null() {
        return Err(fail("DLL returned a null JSON pointer"));
    }
    let bytes = slice::from_raw_parts(pointer.cast::<u8>(), length).to_vec();
    (api.free)(pointer);
    String::from_utf8(bytes).map_err(|error| fail(format!("DLL returned non-UTF-8 JSON: {error}")))
}

fn validate_models(api: &Api<'_>, config: &KvdConfig) -> Result<(), Box<dyn Error>> {
    let mut pointer = ptr::null_mut();
    let mut length = 0usize;
    let return_code = unsafe { (api.validate_models)(config, &mut pointer, &mut length) };
    let message = if pointer.is_null() {
        String::from("<no validation message>")
    } else {
        unsafe { take_json(api, pointer, length)? }
    };
    if return_code != 0 {
        return Err(fail(format!(
            "kvd_validate_models failed with {return_code}: {message}"
        )));
    }
    Ok(())
}

fn scan_path(api: &Api<'_>, handle: *mut KvdHandle, path: &Path) -> Result<Value, Box<dyn Error>> {
    let path_text = path_c_string(path, "sample path")?;
    let mut pointer = ptr::null_mut();
    let mut length = 0usize;
    let return_code =
        unsafe { (api.scan_path)(handle, path_text.as_ptr(), &mut pointer, &mut length) };
    let response = unsafe { take_json(api, pointer, length)? };
    if return_code != 0 {
        return Err(fail(format!(
            "kvd_scan_path failed with {return_code}: {response}"
        )));
    }
    serde_json::from_str(&response)
        .map_err(|error| fail(format!("kvd_scan_path returned invalid JSON: {error}")))
}

fn scan_bytes(
    api: &Api<'_>,
    handle: *mut KvdHandle,
    bytes: &[u8],
) -> Result<Value, Box<dyn Error>> {
    let mut pointer = ptr::null_mut();
    let mut length = 0usize;
    let return_code = unsafe {
        (api.scan_bytes)(
            handle,
            bytes.as_ptr(),
            bytes.len(),
            &mut pointer,
            &mut length,
        )
    };
    let response = unsafe { take_json(api, pointer, length)? };
    if return_code != 0 {
        return Err(fail(format!(
            "kvd_scan_bytes failed with {return_code}: {response}"
        )));
    }
    serde_json::from_str(&response)
        .map_err(|error| fail(format!("kvd_scan_bytes returned invalid JSON: {error}")))
}

fn loop151_prediction(response: &Value, source: &str) -> Result<i64, Box<dyn Error>> {
    if response.get("loop_id").and_then(Value::as_str) != Some("Loop151") {
        return Err(fail(format!("{source} response is not tagged Loop151")));
    }
    match response.get("prediction").and_then(Value::as_i64) {
        Some(prediction @ (0 | 1)) => Ok(prediction),
        Some(value) => Err(fail(format!(
            "{source} returned invalid prediction {value}"
        ))),
        None => Err(fail(format!("{source} response has no integer prediction"))),
    }
}

struct Arguments {
    dll: PathBuf,
    runtime_config: PathBuf,
    sample: PathBuf,
    allowed_scan_root: Option<PathBuf>,
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} --dll <axon_loop151_champion.dll> \\
  --runtime-config <runtime/loop151_native_runtime.json> \\
  --sample <sample.exe> [--allowed-scan-root <directory>]"
    );
}

fn parse_arguments() -> Result<Option<Arguments>, Box<dyn Error>> {
    let mut arguments = env::args_os();
    let program = arguments
        .next()
        .unwrap_or_else(|| "axon-loop151-rust-example".into());
    let program_text = program.to_string_lossy().into_owned();
    let mut dll = None;
    let mut runtime_config = None;
    let mut sample = None;
    let mut allowed_scan_root = None;

    while let Some(flag) = arguments.next() {
        let flag_text = flag.to_string_lossy();
        if flag_text == "--help" || flag_text == "-h" {
            usage(&program_text);
            return Ok(None);
        }
        let value = arguments
            .next()
            .ok_or_else(|| fail(format!("missing value for {flag_text}")))?;
        match flag_text.as_ref() {
            "--dll" => dll = Some(PathBuf::from(value)),
            "--runtime-config" => runtime_config = Some(PathBuf::from(value)),
            "--sample" => sample = Some(PathBuf::from(value)),
            "--allowed-scan-root" => allowed_scan_root = Some(PathBuf::from(value)),
            _ => return Err(fail(format!("unknown argument {flag_text}"))),
        }
    }

    let arguments = Arguments {
        dll: dll.ok_or_else(|| fail("--dll is required"))?,
        runtime_config: runtime_config.ok_or_else(|| fail("--runtime-config is required"))?,
        sample: sample.ok_or_else(|| fail("--sample is required"))?,
        allowed_scan_root,
    };
    Ok(Some(arguments))
}

fn run(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let runtime_config = path_c_string(&arguments.runtime_config, "runtime config")?;
    let allowed_scan_root = arguments
        .allowed_scan_root
        .as_deref()
        .map(|path| path_c_string(path, "allowed scan root"))
        .transpose()?;
    let sample_bytes = std::fs::read(&arguments.sample)?;

    if std::mem::size_of::<KvdConfig>() != 96 {
        return Err(fail(format!(
            "unexpected kvd_config size {}; expected 96 on x64",
            std::mem::size_of::<KvdConfig>()
        )));
    }

    let config = KvdConfig {
        stage2_model_json_path: runtime_config.as_ptr(),
        allowed_scan_root: allowed_scan_root
            .as_ref()
            .map_or(ptr::null(), |value| value.as_ptr()),
        ..KvdConfig::empty()
    };

    let library = unsafe { Library::new(&arguments.dll)? };
    let api = unsafe { Api::load(&library)? };
    validate_models(&api, &config)?;

    let handle = unsafe { (api.create)(&config) };
    if handle.is_null() {
        return Err(fail("kvd_create returned a null handle"));
    }

    let scans = (
        scan_path(&api, handle, &arguments.sample),
        scan_bytes(&api, handle, &sample_bytes),
    );
    unsafe { (api.destroy)(handle) };
    let path_response = scans.0?;
    let bytes_response = scans.1?;
    let path_prediction = loop151_prediction(&path_response, "kvd_scan_path")?;
    let bytes_prediction = loop151_prediction(&bytes_response, "kvd_scan_bytes")?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema": "axon_loop151_rust_call_example_v1",
            "calling_convention": "__cdecl",
            "kvd_config_x64_size": std::mem::size_of::<KvdConfig>(),
            "dll": arguments.dll,
            "runtime_config": arguments.runtime_config,
            "sample": arguments.sample,
            "path_prediction": path_prediction,
            "bytes_prediction": bytes_prediction,
            "same_prediction": path_prediction == bytes_prediction,
            "path_result": path_response,
            "bytes_result": bytes_response,
        }))?
    );
    if path_prediction != bytes_prediction {
        return Err(fail("path and byte scans disagree on prediction"));
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(arguments) = parse_arguments()? else {
        return Ok(());
    };
    if !cfg!(target_os = "windows") {
        return Err(fail(
            "the Loop151 champion DLL and its __cdecl KVD ABI are Windows-only",
        ));
    }
    run(arguments)
}
