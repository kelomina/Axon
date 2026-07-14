use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

#[derive(Debug, Parser)]
#[command(name = "axon-archive-scanner")]
#[command(about = "Strict nested archive scanner for Axon v2.6")]
struct Args {
    #[arg(long)]
    input: PathBuf,

    #[arg(long, value_enum, default_value = "text")]
    output: OutputMode,

    #[arg(long, default_value_t = 4)]
    max_depth: usize,

    #[arg(long, default_value_t = 4096)]
    max_files: usize,

    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    max_total_bytes: u64,

    #[arg(long, default_value_t = 128 * 1024 * 1024)]
    max_file_bytes: u64,

    #[arg(long)]
    keep_temp: bool,

    #[arg(long)]
    temp_root: Option<PathBuf>,
}

#[derive(Clone, Debug, ValueEnum)]
enum OutputMode {
    Text,
    Json,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FileKind {
    Pe,
    Zip,
    SevenZ,
    Rar,
    Msi,
    Cab,
    Other,
}

impl FileKind {
    fn as_label(self) -> &'static str {
        match self {
            FileKind::Pe => "pe",
            FileKind::Zip => "zip",
            FileKind::SevenZ => "7z",
            FileKind::Rar => "rar",
            FileKind::Msi => "msi",
            FileKind::Cab => "cab",
            FileKind::Other => "other",
        }
    }

    fn is_archive(self) -> bool {
        matches!(
            self,
            FileKind::Zip | FileKind::SevenZ | FileKind::Rar | FileKind::Msi | FileKind::Cab
        )
    }

    fn is_candidate(self) -> bool {
        matches!(
            self,
            FileKind::Pe
                | FileKind::Zip
                | FileKind::SevenZ
                | FileKind::Rar
                | FileKind::Msi
                | FileKind::Cab
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ScanStatus {
    Scanned,
    Candidate,
    Skipped,
    Blocked,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReportEntry {
    id: usize,
    parent_id: Option<usize>,
    depth: usize,
    logical_path: String,
    extracted_path: Option<String>,
    kind: FileKind,
    size: u64,
    sha256: Option<String>,
    candidate_for_axon: bool,
    archive: bool,
    training_label_policy: String,
    status: ScanStatus,
    reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScanLimits {
    max_depth: usize,
    max_files: usize,
    max_total_bytes: u64,
    max_file_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScanSummary {
    total_entries: usize,
    candidate_entries: usize,
    blocked_entries: usize,
    error_entries: usize,
    total_observed_bytes: u64,
    root_verdict_policy: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScanReport {
    version: u32,
    input: String,
    temp_dir: Option<String>,
    limits: ScanLimits,
    summary: ScanSummary,
    entries: Vec<ReportEntry>,
}

#[derive(Debug, Clone)]
struct QueueItem {
    entry_id: usize,
    path: PathBuf,
    depth: usize,
    logical_path: String,
}

struct Scanner {
    args: Args,
    temp_dir: TempDir,
    entries: Vec<ReportEntry>,
    queue: VecDeque<QueueItem>,
    next_id: usize,
    total_observed_bytes: u64,
}

impl Scanner {
    fn new(args: Args) -> Result<Self> {
        let temp_dir = if let Some(root) = &args.temp_root {
            fs::create_dir_all(root)
                .with_context(|| format!("failed to create temp root {}", root.display()))?;
            tempfile::Builder::new()
                .prefix("axon-archive-scanner-")
                .tempdir_in(root)
                .with_context(|| format!("failed to create temp dir under {}", root.display()))?
        } else {
            tempfile::Builder::new()
                .prefix("axon-archive-scanner-")
                .tempdir()
                .context("failed to create temp dir")?
        };

        Ok(Self {
            args,
            temp_dir,
            entries: Vec::new(),
            queue: VecDeque::new(),
            next_id: 0,
            total_observed_bytes: 0,
        })
    }

    fn scan(mut self) -> Result<ScanReport> {
        let input = self
            .args
            .input
            .canonicalize()
            .with_context(|| format!("input does not exist: {}", self.args.input.display()))?;
        if !input.is_file() {
            bail!("input must be a file: {}", input.display());
        }

        let root_kind = detect_kind(&input)?;
        let root_size = file_size(&input)?;
        let root_sha = sha256_file(&input).ok();
        let root_id = self.add_entry(EntryInput {
            parent_id: None,
            depth: 0,
            logical_path: input
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| input.display().to_string()),
            extracted_path: Some(input.display().to_string()),
            kind: root_kind,
            size: root_size,
            sha256: root_sha,
            status: if root_kind.is_candidate() {
                ScanStatus::Candidate
            } else {
                ScanStatus::Scanned
            },
            reason: None,
        });

        if root_kind.is_archive() {
            self.queue.push_back(QueueItem {
                entry_id: root_id,
                path: input,
                depth: 0,
                logical_path: self.entries[root_id].logical_path.clone(),
            });
        }

        while let Some(item) = self.queue.pop_front() {
            if self.entries.len() >= self.args.max_files {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/<scan-stopped>", item.logical_path),
                    "max file count reached",
                );
                break;
            }

            if item.depth >= self.args.max_depth {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/<max-depth>", item.logical_path),
                    "max nested depth reached",
                );
                continue;
            }

            if let Err(err) = self.extract_archive(&item) {
                self.add_entry(EntryInput {
                    parent_id: Some(item.entry_id),
                    depth: item.depth + 1,
                    logical_path: format!("{}/<extract-error>", item.logical_path),
                    extracted_path: None,
                    kind: FileKind::Other,
                    size: 0,
                    sha256: None,
                    status: ScanStatus::Error,
                    reason: Some(err.to_string()),
                });
            }
        }

        let blocked_entries = self
            .entries
            .iter()
            .filter(|e| e.status == ScanStatus::Blocked)
            .count();
        let error_entries = self
            .entries
            .iter()
            .filter(|e| e.status == ScanStatus::Error)
            .count();
        let candidate_entries = self.entries.iter().filter(|e| e.candidate_for_axon).count();

        let temp_dir_path = if self.args.keep_temp {
            let path = self.temp_dir.path().to_path_buf();
            // Python integration needs extracted files to survive the Rust process.
            // The caller is responsible for deleting this directory after prediction.
            let _persisted_path = self.temp_dir.keep();
            Some(path.display().to_string())
        } else {
            None
        };

        Ok(ScanReport {
            version: 1,
            input: self.args.input.display().to_string(),
            temp_dir: temp_dir_path,
            limits: ScanLimits {
                max_depth: self.args.max_depth,
                max_files: self.args.max_files,
                max_total_bytes: self.args.max_total_bytes,
                max_file_bytes: self.args.max_file_bytes,
            },
            summary: ScanSummary {
                total_entries: self.entries.len(),
                candidate_entries,
                blocked_entries,
                error_entries,
                total_observed_bytes: self.total_observed_bytes,
                root_verdict_policy: "runtime: any malicious inner PE triggers parent alert; training: inner labels remain unknown unless explicitly labeled".to_string(),
            },
            entries: self.entries,
        })
    }

    fn extract_archive(&mut self, item: &QueueItem) -> Result<()> {
        let kind = detect_kind(&item.path)?;
        match kind {
            FileKind::Zip => self.extract_zip(item),
            FileKind::SevenZ => self.extract_7z(item),
            FileKind::Cab => self.extract_cab(item),
            FileKind::Msi => self.extract_msi(item),
            FileKind::Rar => {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/<rar-unsupported>", item.logical_path),
                    "rar extraction is unsupported without an external backend; local 7z is intentionally not used",
                );
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn extract_zip(&mut self, item: &QueueItem) -> Result<()> {
        let file = File::open(&item.path)
            .with_context(|| format!("failed to open zip {}", item.path.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("failed to read zip {}", item.path.display()))?;
        let out_dir = self.archive_output_dir(item.entry_id)?;

        for index in 0..archive.len() {
            if self.entries.len() >= self.args.max_files {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/<scan-stopped>", item.logical_path),
                    "max file count reached",
                );
                break;
            }

            let mut file = archive
                .by_index(index)
                .with_context(|| format!("failed to read zip entry {}", index))?;
            if file.is_dir() {
                continue;
            }

            let enclosed = match file.enclosed_name() {
                Some(path) => path.to_owned(),
                None => {
                    self.add_blocked_child(
                        Some(item.entry_id),
                        item.depth + 1,
                        format!("{}/{}", item.logical_path, file.name()),
                        "unsafe zip entry path",
                    );
                    continue;
                }
            };
            if is_suspicious_relative_path(&enclosed) {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/{}", item.logical_path, enclosed.display()),
                    "unsafe nested path component",
                );
                continue;
            }

            let uncompressed = file.size();
            if uncompressed > self.args.max_file_bytes {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/{}", item.logical_path, enclosed.display()),
                    "entry exceeds max file bytes",
                );
                continue;
            }
            self.write_reader_entry(item, &out_dir, enclosed, uncompressed, &mut file)?;
        }
        Ok(())
    }

    fn extract_7z(&mut self, item: &QueueItem) -> Result<()> {
        let out_dir = self.archive_output_dir(item.entry_id)?;
        sevenz_rust::decompress_file_with_extract_fn(&item.path, &out_dir, |entry, reader, _| {
            if entry.is_directory() {
                return Ok(true);
            }

            if self.entries.len() >= self.args.max_files {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/<scan-stopped>", item.logical_path),
                    "max file count reached",
                );
                return Ok(false);
            }

            let relative = match archive_name_to_relative_path(entry.name()) {
                Some(path) => path,
                None => {
                    self.add_blocked_child(
                        Some(item.entry_id),
                        item.depth + 1,
                        format!("{}/{}", item.logical_path, entry.name()),
                        "unsafe 7z entry path",
                    );
                    return Ok(true);
                }
            };
            if entry.size() > self.args.max_file_bytes {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/{}", item.logical_path, relative.display()),
                    "entry exceeds max file bytes",
                );
                return Ok(true);
            }

            if let Err(err) =
                self.write_reader_entry(item, &out_dir, relative, entry.size(), reader)
            {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/{}", item.logical_path, entry.name()),
                    &err.to_string(),
                );
            }
            Ok(true)
        })
        .with_context(|| format!("failed to extract 7z {}", item.path.display()))?;

        Ok(())
    }

    fn extract_cab(&mut self, item: &QueueItem) -> Result<()> {
        let file = File::open(&item.path)
            .with_context(|| format!("failed to open cab {}", item.path.display()))?;
        let mut cabinet = cab::Cabinet::new(file)
            .with_context(|| format!("failed to read cab {}", item.path.display()))?;
        let out_dir = self.archive_output_dir(item.entry_id)?;
        let mut files = Vec::new();
        let remaining_slots = self.args.max_files.saturating_sub(self.entries.len());
        let mut truncated_by_limit = false;
        'collect_cab: for folder in cabinet.folder_entries() {
            for file in folder.file_entries() {
                if files.len() >= remaining_slots {
                    truncated_by_limit = true;
                    break 'collect_cab;
                }
                files.push((file.name().to_string(), file.uncompressed_size() as u64));
            }
        }
        for (name, size) in files {
            if self.entries.len() >= self.args.max_files {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/<scan-stopped>", item.logical_path),
                    "max file count reached",
                );
                break;
            }

            let relative = match archive_name_to_relative_path(&name) {
                Some(path) => path,
                None => {
                    self.add_blocked_child(
                        Some(item.entry_id),
                        item.depth + 1,
                        format!("{}/{}", item.logical_path, name),
                        "unsafe cab entry path",
                    );
                    continue;
                }
            };
            if size > self.args.max_file_bytes {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/{}", item.logical_path, relative.display()),
                    "entry exceeds max file bytes",
                );
                continue;
            }

            let mut reader = cabinet
                .read_file(&name)
                .with_context(|| format!("failed to read cab entry {}", name))?;
            self.write_reader_entry(item, &out_dir, relative, size, &mut reader)?;
        }
        if truncated_by_limit {
            self.add_blocked_child(
                Some(item.entry_id),
                item.depth + 1,
                format!("{}/<scan-stopped>", item.logical_path),
                "max file count reached",
            );
        }

        Ok(())
    }

    fn extract_msi(&mut self, item: &QueueItem) -> Result<()> {
        let mut package = msi::open(&item.path)
            .with_context(|| format!("failed to open msi {}", item.path.display()))?;
        let out_dir = self.archive_output_dir(item.entry_id)?;
        let remaining_slots = self.args.max_files.saturating_sub(self.entries.len());
        let mut stream_names: Vec<String> = package.streams().take(remaining_slots + 1).collect();
        let truncated_by_limit = stream_names.len() > remaining_slots;
        if truncated_by_limit {
            stream_names.truncate(remaining_slots);
        }

        if stream_names.is_empty() {
            if truncated_by_limit {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/<scan-stopped>", item.logical_path),
                    "max file count reached",
                );
                return Ok(());
            }
            self.add_blocked_child(
                Some(item.entry_id),
                item.depth + 1,
                format!("{}/<msi-no-embedded-streams>", item.logical_path),
                "msi parsed, but no embedded binary streams were found; installed file table reconstruction is not performed in v1",
            );
            return Ok(());
        }

        for (index, stream_name) in stream_names.iter().enumerate() {
            if self.entries.len() >= self.args.max_files {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/<scan-stopped>", item.logical_path),
                    "max file count reached",
                );
                break;
            }

            let mut reader = package
                .read_stream(stream_name)
                .with_context(|| format!("failed to read msi stream {}", stream_name))?;
            let size = reader
                .seek(SeekFrom::End(0))
                .with_context(|| format!("failed to size msi stream {}", stream_name))?;
            reader
                .seek(SeekFrom::Start(0))
                .with_context(|| format!("failed to rewind msi stream {}", stream_name))?;
            if size > self.args.max_file_bytes {
                self.add_blocked_child(
                    Some(item.entry_id),
                    item.depth + 1,
                    format!("{}/{}", item.logical_path, stream_name),
                    "entry exceeds max file bytes",
                );
                continue;
            }

            let relative = PathBuf::from(format!(
                "msi_stream_{index:04}_{}",
                sanitize_msi_stream_name(stream_name)
            ));
            self.write_reader_entry(item, &out_dir, relative, size, &mut reader)?;
        }
        if truncated_by_limit {
            self.add_blocked_child(
                Some(item.entry_id),
                item.depth + 1,
                format!("{}/<scan-stopped>", item.logical_path),
                "max file count reached",
            );
        }

        Ok(())
    }

    fn write_reader_entry(
        &mut self,
        item: &QueueItem,
        out_dir: &Path,
        relative: PathBuf,
        size: u64,
        reader: &mut dyn Read,
    ) -> Result<()> {
        self.try_add_observed_bytes(
            size,
            Some(item.entry_id),
            item.depth + 1,
            &item.logical_path,
        )?;

        let out_path = safe_join(out_dir, &relative)
            .ok_or_else(|| anyhow!("unsafe output path {}", relative.display()))?;
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let mut out_file = File::create(&out_path)
            .with_context(|| format!("failed to create {}", out_path.display()))?;
        let copy_limit = self.args.max_file_bytes.saturating_add(1);
        let mut limited_reader = reader.take(copy_limit);
        let copied = std::io::copy(&mut limited_reader, &mut out_file)
            .with_context(|| format!("failed to extract {}", relative.display()))?;
        out_file.flush().ok();
        drop(out_file);

        if copied > self.args.max_file_bytes {
            let logical_child = format!(
                "{}/{}",
                item.logical_path,
                relative.to_string_lossy().replace('\\', "/")
            );
            let _ = fs::remove_file(&out_path);
            self.add_blocked_child(
                Some(item.entry_id),
                item.depth + 1,
                logical_child,
                "actual extracted bytes exceed max file bytes",
            );
            return Ok(());
        }
        if copied > size {
            if let Err(err) = self.try_add_observed_bytes(
                copied - size,
                Some(item.entry_id),
                item.depth + 1,
                &item.logical_path,
            ) {
                let _ = fs::remove_file(&out_path);
                return Err(err);
            }
        }

        self.record_extracted_file(item, relative, out_path)
    }

    fn record_extracted_file(
        &mut self,
        item: &QueueItem,
        relative: PathBuf,
        out_path: PathBuf,
    ) -> Result<()> {
        let kind = detect_kind(&out_path)?;
        let size = file_size(&out_path)?;
        let logical_child = format!(
            "{}/{}",
            item.logical_path,
            relative.to_string_lossy().replace('\\', "/")
        );
        let id = self.add_entry(EntryInput {
            parent_id: Some(item.entry_id),
            depth: item.depth + 1,
            logical_path: logical_child.clone(),
            extracted_path: Some(out_path.display().to_string()),
            kind,
            size,
            sha256: sha256_file(&out_path).ok(),
            status: if kind.is_candidate() {
                ScanStatus::Candidate
            } else {
                ScanStatus::Scanned
            },
            reason: None,
        });
        if kind.is_archive() {
            self.queue.push_back(QueueItem {
                entry_id: id,
                path: out_path,
                depth: item.depth + 1,
                logical_path: logical_child,
            });
        }
        Ok(())
    }

    fn archive_output_dir(&self, entry_id: usize) -> Result<PathBuf> {
        let path = self.temp_dir.path().join(format!("entry_{entry_id}"));
        fs::create_dir_all(&path)
            .with_context(|| format!("failed to create {}", path.display()))?;
        Ok(path)
    }

    fn try_add_observed_bytes(
        &mut self,
        size: u64,
        parent_id: Option<usize>,
        depth: usize,
        parent_logical_path: &str,
    ) -> Result<()> {
        match self.total_observed_bytes.checked_add(size) {
            Some(next) if next <= self.args.max_total_bytes => {
                self.total_observed_bytes = next;
                Ok(())
            }
            _ => {
                self.add_entry(EntryInput {
                    parent_id,
                    depth,
                    logical_path: format!("{parent_logical_path}/<max-total-bytes>"),
                    extracted_path: None,
                    kind: FileKind::Other,
                    size: 0,
                    sha256: None,
                    status: ScanStatus::Blocked,
                    reason: Some("max total extracted bytes reached".to_string()),
                });
                bail!("max total extracted bytes reached")
            }
        }
    }

    fn add_blocked_child(
        &mut self,
        parent_id: Option<usize>,
        depth: usize,
        logical_path: String,
        reason: &str,
    ) {
        self.add_entry(EntryInput {
            parent_id,
            depth,
            logical_path,
            extracted_path: None,
            kind: FileKind::Other,
            size: 0,
            sha256: None,
            status: ScanStatus::Blocked,
            reason: Some(reason.to_string()),
        });
    }

    fn add_entry(&mut self, input: EntryInput) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        let archive = input.kind.is_archive();
        let candidate_for_axon = input.kind.is_candidate();
        self.entries.push(ReportEntry {
            id,
            parent_id: input.parent_id,
            depth: input.depth,
            logical_path: input.logical_path,
            extracted_path: input.extracted_path,
            kind: input.kind,
            size: input.size,
            sha256: input.sha256,
            candidate_for_axon,
            archive,
            training_label_policy: if candidate_for_axon {
                "unknown_training_label: do not inherit parent archive/MSI directory label automatically".to_string()
            } else {
                "not_training_candidate".to_string()
            },
            status: input.status,
            reason: input.reason,
        });
        id
    }
}

struct EntryInput {
    parent_id: Option<usize>,
    depth: usize,
    logical_path: String,
    extracted_path: Option<String>,
    kind: FileKind,
    size: u64,
    sha256: Option<String>,
    status: ScanStatus,
    reason: Option<String>,
}

fn detect_kind(path: &Path) -> Result<FileKind> {
    let mut buf = [0u8; 16];
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let read_len = file
        .read(&mut buf)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let sig = &buf[..read_len];

    if sig.starts_with(b"MZ") {
        return Ok(FileKind::Pe);
    }
    if sig.starts_with(b"PK\x03\x04")
        || sig.starts_with(b"PK\x05\x06")
        || sig.starts_with(b"PK\x07\x08")
    {
        return Ok(FileKind::Zip);
    }
    if sig.starts_with(&[0x37, 0x7a, 0xbc, 0xaf, 0x27, 0x1c]) {
        return Ok(FileKind::SevenZ);
    }
    if sig.starts_with(b"Rar!\x1a\x07\x00") || sig.starts_with(b"Rar!\x1a\x07\x01\x00") {
        return Ok(FileKind::Rar);
    }
    if sig.starts_with(&[0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]) {
        return Ok(FileKind::Msi);
    }
    if sig.starts_with(b"MSCF") {
        return Ok(FileKind::Cab);
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    Ok(match ext.as_str() {
        "exe" | "dll" | "sys" | "scr" => FileKind::Pe,
        "zip" => FileKind::Zip,
        "7z" => FileKind::SevenZ,
        "rar" => FileKind::Rar,
        "msi" => FileKind::Msi,
        "cab" => FileKind::Cab,
        _ => FileKind::Other,
    })
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn is_suspicious_relative_path(path: &Path) -> bool {
    path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_) | Component::CurDir
        )
    })
}

fn archive_name_to_relative_path(name: &str) -> Option<PathBuf> {
    let normalized = name.replace('\\', "/");
    let path = PathBuf::from(normalized);
    if is_suspicious_relative_path(&path) {
        None
    } else {
        Some(path)
    }
}

fn safe_join(root: &Path, relative: &Path) -> Option<PathBuf> {
    if is_suspicious_relative_path(relative) {
        return None;
    }
    Some(root.join(relative))
}

fn sanitize_msi_stream_name(name: &str) -> String {
    let mut sanitized = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }
    if sanitized.is_empty() {
        "stream.bin".to_string()
    } else {
        sanitized
    }
}

fn print_text(report: &ScanReport) {
    println!("Axon Archive Scanner");
    println!("Input: {}", report.input);
    println!(
        "Entries: {} | Candidates: {} | Blocked: {} | Errors: {}",
        report.summary.total_entries,
        report.summary.candidate_entries,
        report.summary.blocked_entries,
        report.summary.error_entries
    );
    println!("Training labels: inner archive/MSI contents are unknown unless explicitly labeled.");
    println!();
    for entry in &report.entries {
        let indent = "  ".repeat(entry.depth);
        let marker = if entry.candidate_for_axon { "*" } else { "-" };
        let mut line = format!(
            "{indent}{marker} [{}] {} size={} status={:?}",
            entry.kind.as_label(),
            entry.logical_path,
            entry.size,
            entry.status
        );
        if let Some(reason) = &entry.reason {
            line.push_str(&format!(" reason={reason}"));
        }
        println!("{line}");
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let output_mode = args.output.clone();
    let keep_temp = args.keep_temp;
    let scanner = Scanner::new(args)?;
    let report = scanner.scan()?;

    match output_mode {
        OutputMode::Text => print_text(&report),
        OutputMode::Json => {
            let stdout = std::io::stdout();
            let mut handle = stdout.lock();
            serde_json::to_writer_pretty(&mut handle, &report)?;
            writeln!(handle)?;
        }
    }

    if keep_temp {
        eprintln!(
            "[axon-archive-scanner] temporary files kept at: {}",
            report.temp_dir.as_deref().unwrap_or("")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cab::{CabinetBuilder, CompressionType};
    use msi::{Package, PackageType};
    use zip::write::SimpleFileOptions;

    fn zip_with_files(path: &Path, files: &[(&str, &[u8])]) {
        let file = File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        for (name, bytes) in files {
            zip.start_file(*name, SimpleFileOptions::default()).unwrap();
            zip.write_all(bytes).unwrap();
        }
        zip.finish().unwrap();
    }

    fn test_args(input: PathBuf) -> Args {
        Args {
            input,
            output: OutputMode::Json,
            max_depth: 4,
            max_files: 16,
            max_total_bytes: 1024 * 1024,
            max_file_bytes: 1024 * 1024,
            keep_temp: false,
            temp_root: None,
        }
    }

    #[test]
    fn detects_magic_before_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sample.txt");
        fs::write(&path, b"MZhello").unwrap();
        assert_eq!(detect_kind(&path).unwrap(), FileKind::Pe);
    }

    #[test]
    fn scans_plain_pe_without_archive_recursion() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plain.exe");
        fs::write(&path, b"MZplain").unwrap();

        let report = Scanner::new(test_args(path)).unwrap().scan().unwrap();

        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.entries[0].kind, FileKind::Pe);
        assert!(report.entries[0].sha256.is_some());
    }

    #[test]
    fn detects_archive_magic_headers() {
        let dir = tempfile::tempdir().unwrap();
        let msi = dir.path().join("sample.bin");
        let sevenz = dir.path().join("sample.dat");
        let rar = dir.path().join("sample.raw");
        fs::write(&msi, [0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1, 0, 0]).unwrap();
        fs::write(&sevenz, [0x37, 0x7a, 0xbc, 0xaf, 0x27, 0x1c, 0, 0]).unwrap();
        fs::write(&rar, b"Rar!\x1a\x07\x00hello").unwrap();

        assert_eq!(detect_kind(&msi).unwrap(), FileKind::Msi);
        assert_eq!(detect_kind(&sevenz).unwrap(), FileKind::SevenZ);
        assert_eq!(detect_kind(&rar).unwrap(), FileKind::Rar);
    }

    #[test]
    fn rejects_suspicious_paths() {
        assert!(is_suspicious_relative_path(Path::new("../evil.exe")));
        assert!(is_suspicious_relative_path(Path::new("/evil.exe")));
        assert!(archive_name_to_relative_path("..\\evil.exe").is_none());
        assert!(!is_suspicious_relative_path(Path::new("nested/good.exe")));
    }

    #[test]
    fn rar_is_detected_but_blocked_without_external_backend() {
        let dir = tempfile::tempdir().unwrap();
        let rar_path = dir.path().join("sample.rar");
        fs::write(&rar_path, b"Rar!\x1a\x07\x00payload").unwrap();

        let report = Scanner::new(test_args(rar_path)).unwrap().scan().unwrap();

        assert!(report.entries.iter().any(|entry| {
            entry.status == ScanStatus::Blocked
                && entry
                    .reason
                    .as_deref()
                    .unwrap_or("")
                    .contains("local 7z is intentionally not used")
        }));
    }

    #[test]
    fn extracts_msi_binary_stream_candidates_without_7z() {
        let dir = tempfile::tempdir().unwrap();
        let msi_path = dir.path().join("embedded.msi");
        {
            let file = File::create(&msi_path).unwrap();
            let mut package = Package::create(PackageType::Installer, file).unwrap();
            let mut stream = package.write_stream("evil_payload").unwrap();
            stream.write_all(b"MZinside-msi").unwrap();
            stream.flush().unwrap();
            package.flush().unwrap();
        }

        let report = Scanner::new(test_args(msi_path)).unwrap().scan().unwrap();
        let pe = report
            .entries
            .iter()
            .find(|entry| entry.kind == FileKind::Pe)
            .unwrap();
        assert!(pe.logical_path.contains("evil_payload"));
        assert!(pe.training_label_policy.contains("unknown_training_label"));
    }

    #[test]
    fn scans_nested_zip_and_marks_unknown_training_label() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("outer.zip");
        zip_with_files(&zip_path, &[("inner.exe", b"MZinner")]);

        let args = test_args(zip_path);
        let report = Scanner::new(args).unwrap().scan().unwrap();
        let pe = report
            .entries
            .iter()
            .find(|entry| entry.kind == FileKind::Pe)
            .unwrap();
        assert!(pe.logical_path.ends_with("inner.exe"));
        assert!(pe.candidate_for_axon);
        assert!(pe.training_label_policy.contains("unknown_training_label"));
    }

    #[test]
    fn scans_7z_without_local_7z_binary() {
        let dir = tempfile::tempdir().unwrap();
        let payload_dir = dir.path().join("payload");
        fs::create_dir_all(&payload_dir).unwrap();
        fs::write(payload_dir.join("inner.exe"), b"MZinner").unwrap();
        let archive_path = dir.path().join("outer.7z");
        sevenz_rust::compress_to_path(&payload_dir, &archive_path).unwrap();

        let report = Scanner::new(test_args(archive_path))
            .unwrap()
            .scan()
            .unwrap();
        let pe = report
            .entries
            .iter()
            .find(|entry| entry.kind == FileKind::Pe)
            .unwrap();
        assert!(pe.logical_path.ends_with("inner.exe"));
        assert!(pe.candidate_for_axon);
    }

    #[test]
    fn scans_cab_without_local_7z_binary() {
        let dir = tempfile::tempdir().unwrap();
        let cab_path = dir.path().join("outer.cab");
        {
            let mut builder = CabinetBuilder::new();
            builder
                .add_folder(CompressionType::None)
                .add_file("inner.exe");
            let file = File::create(&cab_path).unwrap();
            let mut writer = builder.build(file).unwrap();
            let mut inner = writer.next_file().unwrap().unwrap();
            inner.write_all(b"MZinner").unwrap();
            drop(inner);
            writer.finish().unwrap();
        }

        let report = Scanner::new(test_args(cab_path)).unwrap().scan().unwrap();
        let pe = report
            .entries
            .iter()
            .find(|entry| entry.kind == FileKind::Pe)
            .unwrap();
        assert!(pe.logical_path.ends_with("inner.exe"));
        assert!(pe.candidate_for_axon);
    }

    #[test]
    fn blocks_zip_slip_paths() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("evil.zip");
        zip_with_files(&zip_path, &[("../evil.exe", b"MZevil")]);

        let report = Scanner::new(test_args(zip_path)).unwrap().scan().unwrap();

        assert!(report.entries.iter().any(|entry| {
            entry.status == ScanStatus::Blocked
                && entry
                    .reason
                    .as_deref()
                    .unwrap_or("")
                    .contains("unsafe zip entry path")
        }));
        assert!(!dir.path().join("evil.exe").exists());
    }

    #[test]
    fn blocks_when_max_depth_reached() {
        let dir = tempfile::tempdir().unwrap();
        let inner_zip = dir.path().join("inner.zip");
        zip_with_files(&inner_zip, &[("inner.exe", b"MZinner")]);
        let inner_bytes = fs::read(&inner_zip).unwrap();
        let outer_zip = dir.path().join("outer.zip");
        zip_with_files(&outer_zip, &[("inner.zip", &inner_bytes)]);

        let mut args = test_args(outer_zip);
        args.max_depth = 1;
        let report = Scanner::new(args).unwrap().scan().unwrap();

        assert!(report.entries.iter().any(|entry| {
            entry.status == ScanStatus::Blocked
                && entry
                    .reason
                    .as_deref()
                    .unwrap_or("")
                    .contains("max nested depth")
        }));
    }

    #[test]
    fn blocks_when_max_file_count_reached() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("many.zip");
        zip_with_files(
            &zip_path,
            &[("a.exe", b"MZa"), ("b.exe", b"MZb"), ("c.exe", b"MZc")],
        );

        let mut args = test_args(zip_path);
        args.max_files = 2;
        let report = Scanner::new(args).unwrap().scan().unwrap();

        assert!(report.summary.blocked_entries >= 1);
        assert!(report.entries.len() <= 3);
    }

    #[test]
    fn blocks_when_single_file_size_limit_reached() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("large.zip");
        zip_with_files(&zip_path, &[("large.exe", b"MZlarge-payload")]);

        let mut args = test_args(zip_path);
        args.max_file_bytes = 4;
        let report = Scanner::new(args).unwrap().scan().unwrap();

        assert!(report.entries.iter().any(|entry| {
            entry.status == ScanStatus::Blocked
                && entry
                    .reason
                    .as_deref()
                    .unwrap_or("")
                    .contains("entry exceeds max file bytes")
        }));
    }

    #[test]
    fn write_reader_entry_blocks_actual_bytes_beyond_limit() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root.zip");
        fs::write(&root, b"PK\x03\x04root").unwrap();
        let mut args = test_args(root.clone());
        args.max_file_bytes = 4;
        let mut scanner = Scanner::new(args).unwrap();
        let parent_id = scanner.add_entry(EntryInput {
            parent_id: None,
            depth: 0,
            logical_path: "root.zip".to_string(),
            extracted_path: Some(root.display().to_string()),
            kind: FileKind::Zip,
            size: 8,
            sha256: None,
            status: ScanStatus::Scanned,
            reason: None,
        });
        let item = QueueItem {
            entry_id: parent_id,
            path: root,
            depth: 0,
            logical_path: "root.zip".to_string(),
        };
        let out_dir = scanner.archive_output_dir(parent_id).unwrap();
        let out_path = out_dir.join("payload.exe");
        let mut reader = std::io::Cursor::new(b"MZabcdef".to_vec());

        scanner
            .write_reader_entry(
                &item,
                &out_dir,
                PathBuf::from("payload.exe"),
                2,
                &mut reader,
            )
            .unwrap();

        assert!(!out_path.exists());
        assert!(scanner.entries.iter().any(|entry| {
            entry.status == ScanStatus::Blocked
                && entry
                    .reason
                    .as_deref()
                    .unwrap_or("")
                    .contains("actual extracted bytes exceed max file bytes")
        }));
        assert!(!scanner
            .entries
            .iter()
            .any(|entry| entry.logical_path.ends_with("payload.exe") && entry.candidate_for_axon));
    }

    #[test]
    fn blocks_when_total_size_limit_reached() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("total.zip");
        zip_with_files(&zip_path, &[("a.exe", b"MZa"), ("b.exe", b"MZb")]);

        let mut args = test_args(zip_path);
        args.max_total_bytes = 4;
        let report = Scanner::new(args).unwrap().scan().unwrap();

        assert!(report.entries.iter().any(|entry| {
            entry.status == ScanStatus::Blocked
                && entry
                    .reason
                    .as_deref()
                    .unwrap_or("")
                    .contains("max total extracted bytes reached")
        }));
    }

    #[test]
    fn report_has_stable_json_shape() {
        let dir = tempfile::tempdir().unwrap();
        let zip_path = dir.path().join("outer.zip");
        zip_with_files(&zip_path, &[("inner.exe", b"MZinner")]);

        let report = Scanner::new(test_args(zip_path)).unwrap().scan().unwrap();
        let value = serde_json::to_value(&report).unwrap();

        assert_eq!(value["version"], 1);
        assert!(value["entries"].is_array());
        assert!(value["summary"]["candidate_entries"].as_u64().unwrap() >= 1);
    }
}
