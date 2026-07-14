use std::env;
use std::ffi::{CStr, CString};
use std::io::Read;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const VERSION_CSTR: &[u8] = b"axon_predict_dll/0.1.0\0";
const MAX_PYTHON_STDOUT_BYTES: usize = 1024 * 1024;
const MAX_PYTHON_STDERR_BYTES: usize = 64 * 1024;
const PYTHON_TIMEOUT: Duration = Duration::from_secs(300);

fn json_error(code: &str, message: impl AsRef<str>) -> String {
    serde_json::json!({
        "ok": false,
        "error_code": code,
        "error": message.as_ref(),
    })
    .to_string()
}

fn cstring_from_string(text: String) -> *mut c_char {
    let sanitized = text.replace('\0', "\\u0000");
    CString::new(sanitized)
        .unwrap_or_else(|_| {
            CString::new("{\"ok\":false,\"error\":\"failed to build response\"}").unwrap()
        })
        .into_raw()
}

fn read_c_string(ptr: *const c_char, name: &str) -> Result<String, String> {
    if ptr.is_null() {
        return Err(format!("{name} is null"));
    }
    let value = unsafe { CStr::from_ptr(ptr) };
    value
        .to_str()
        .map(|text| text.to_string())
        .map_err(|err| format!("{name} must be valid UTF-8: {err}"))
}

fn default_project_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("AXON_PROJECT_ROOT").map(PathBuf::from) {
        candidates.push(path);
    }
    if let Ok(path) = env::current_dir() {
        candidates.push(path.clone());
        candidates.push(path.join(".."));
    }
    if let Ok(path) = env::current_exe() {
        if let Some(parent) = path.parent() {
            candidates.push(parent.to_path_buf());
            candidates.push(parent.join(".."));
            candidates.push(parent.join("..").join("..").join(".."));
        }
    }
    if let Some(path) = current_module_dir() {
        candidates.push(path.clone());
        candidates.push(path.join(".."));
    }

    candidates.into_iter().find_map(|path| {
        let candidate = path.canonicalize().ok()?;
        if candidate.join("src").join("predict_api.py").exists() {
            Some(candidate)
        } else {
            None
        }
    })
}

fn default_python(project_root: &Path) -> PathBuf {
    env::var_os("AXON_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| project_root.join("vnev").join("Scripts").join("python.exe"))
}

struct LimitedOutput {
    status: Option<ExitStatus>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_exceeded: bool,
    stderr_exceeded: bool,
    timed_out: bool,
}

fn read_limited_to_end<R: Read>(mut reader: R, limit: usize) -> (Vec<u8>, bool) {
    let mut captured = Vec::new();
    let mut exceeded = false;
    let mut buffer = [0_u8; 4096];
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            Err(_) => break,
        };
        let remaining = limit.saturating_sub(captured.len());
        if remaining > 0 {
            let keep = remaining.min(read);
            captured.extend_from_slice(&buffer[..keep]);
        }
        if read > remaining {
            exceeded = true;
        }
    }
    (captured, exceeded)
}

fn run_python_predict_process(
    python: &Path,
    project_root: &Path,
    pythonpath: &str,
    request_json: &str,
) -> Result<LimitedOutput, String> {
    let mut child = Command::new(python)
        .arg("-m")
        .arg("predict_api")
        .arg("--request-json")
        .arg(request_json)
        .current_dir(project_root)
        .env("PYTHONPATH", pythonpath)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| err.to_string())?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "failed to capture python stdout".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "failed to capture python stderr".to_string())?;

    let stdout_thread = thread::spawn(move || read_limited_to_end(stdout, MAX_PYTHON_STDOUT_BYTES));
    let stderr_thread = thread::spawn(move || read_limited_to_end(stderr, MAX_PYTHON_STDERR_BYTES));

    let started = Instant::now();
    let (status, timed_out) = loop {
        match child.try_wait() {
            Ok(Some(exit_status)) => {
                break (Some(exit_status), false);
            }
            Ok(None) => {
                if started.elapsed() > PYTHON_TIMEOUT {
                    let _ = child.kill();
                    break (child.wait().ok(), true);
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(err.to_string());
            }
        }
    };

    let (stdout, stdout_exceeded) = stdout_thread.join().unwrap_or_else(|_| (Vec::new(), true));
    let (stderr, stderr_exceeded) = stderr_thread.join().unwrap_or_else(|_| (Vec::new(), true));
    Ok(LimitedOutput {
        status,
        stdout,
        stderr,
        stdout_exceeded,
        stderr_exceeded,
        timed_out,
    })
}

#[cfg(windows)]
fn current_module_dir() -> Option<PathBuf> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    type Hmodule = *mut std::ffi::c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleExW(
            dwFlags: u32,
            lpModuleName: *const u16,
            phModule: *mut Hmodule,
        ) -> i32;
        fn GetModuleFileNameW(hModule: Hmodule, lpFilename: *mut u16, nSize: u32) -> u32;
    }

    const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x0000_0002;
    const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x0000_0004;

    let mut module: Hmodule = std::ptr::null_mut();
    let flags =
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    let address = current_module_dir as *const () as *const u16;
    let ok = unsafe { GetModuleHandleExW(flags, address, &mut module) };
    if ok == 0 || module.is_null() {
        return None;
    }

    let mut buffer = vec![0u16; 32768];
    let len = unsafe { GetModuleFileNameW(module, buffer.as_mut_ptr(), buffer.len() as u32) };
    if len == 0 {
        return None;
    }
    buffer.truncate(len as usize);
    let path = PathBuf::from(OsString::from_wide(&buffer));
    path.parent().map(Path::to_path_buf)
}

#[cfg(not(windows))]
fn current_module_dir() -> Option<PathBuf> {
    None
}

fn call_python_predict(request_json: &str) -> String {
    let Some(project_root) = default_project_root() else {
        return json_error(
            "project_root_not_found",
            "Set AXON_PROJECT_ROOT to the Axon project directory.",
        );
    };
    let python = default_python(&project_root);
    if !python.exists() {
        return json_error(
            "python_not_found",
            format!(
                "Python interpreter not found: {}. Set AXON_PYTHON to override.",
                python.display()
            ),
        );
    }

    let src_path = project_root.join("src");
    let mut pythonpath = src_path.to_string_lossy().to_string();
    if let Some(existing) = env::var_os("PYTHONPATH") {
        pythonpath.push(';');
        pythonpath.push_str(&existing.to_string_lossy());
    }

    let output = run_python_predict_process(&python, &project_root, &pythonpath, request_json);

    match output {
        Ok(output) if output.timed_out => json_error(
            "python_predict_timeout",
            "predict_api exceeded the 300 second timeout",
        ),
        Ok(output) if output.stdout_exceeded => json_error(
            "python_predict_output_too_large",
            format!("predict_api stdout exceeded {MAX_PYTHON_STDOUT_BYTES} bytes"),
        ),
        Ok(output)
            if output
                .status
                .map(|status| status.success())
                .unwrap_or(false) =>
        {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if stdout.is_empty() {
                json_error("empty_python_response", "predict_api returned no JSON")
            } else {
                stdout
            }
        }
        Ok(output) => json_error(
            "python_predict_failed",
            format!(
                "exit={}; stdout{}={}; stderr{}={}",
                output
                    .status
                    .map(|status| status.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                if output.stdout_exceeded {
                    "_truncated"
                } else {
                    ""
                },
                String::from_utf8_lossy(&output.stdout).trim(),
                if output.stderr_exceeded {
                    "_truncated"
                } else {
                    ""
                },
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        ),
        Err(err) => json_error("python_spawn_failed", err.to_string()),
    }
}

/// Predict with a UTF-8 JSON request and return a heap-allocated UTF-8 JSON response.
///
/// Caller owns the returned pointer and must release it with `axon_string_free`.
#[no_mangle]
pub extern "C" fn axon_predict_json(request_json: *const c_char) -> *mut c_char {
    let request_json = match read_c_string(request_json, "request_json") {
        Ok(value) => value,
        Err(err) => return cstring_from_string(json_error("invalid_argument", err)),
    };
    cstring_from_string(call_python_predict(&request_json))
}

/// Free a string returned by this DLL.
#[no_mangle]
pub extern "C" fn axon_string_free(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(ptr);
    }
}

/// Return DLL version as a static UTF-8 C string. Do not free this pointer.
#[no_mangle]
pub extern "C" fn axon_version() -> *const c_char {
    VERSION_CSTR.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_request_returns_json_error() {
        let ptr = axon_predict_json(std::ptr::null());
        assert!(!ptr.is_null());
        let text = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string();
        axon_string_free(ptr);
        assert!(text.contains("invalid_argument"));
    }

    #[test]
    fn version_is_static_c_string() {
        let ptr = axon_version();
        assert!(!ptr.is_null());
        let text = unsafe { CStr::from_ptr(ptr) }.to_string_lossy();
        assert_eq!(text, "axon_predict_dll/0.1.0");
    }

    #[test]
    fn limited_reader_caps_large_output() {
        let data = vec![b'x'; MAX_PYTHON_STDERR_BYTES + 17];
        let (captured, exceeded) =
            read_limited_to_end(std::io::Cursor::new(data), MAX_PYTHON_STDERR_BYTES);
        assert!(exceeded);
        assert_eq!(captured.len(), MAX_PYTHON_STDERR_BYTES);
    }
}
