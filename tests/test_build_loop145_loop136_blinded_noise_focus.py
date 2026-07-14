import csv
import json

from scripts.build_loop145_loop136_blinded_noise_focus import (
    PUBLIC_FIELDNAMES,
    build_loop145_focus,
)


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def base_content_row(source_sha256="a" * 64):
    row = {
        "source_path": "data/hidden/sample.exe",
        "source_sha256": source_sha256,
        "cache_path": "data/.cache/sample.npz",
    }
    for name in [
        "content_is_dll",
        "content_export_count_log",
        "content_dir_export_log_size",
        "content_dir_security_log_size",
        "content_overlay_log_size",
        "content_resource_entry_count_log",
        "content_resource_type_count_log",
        "content_dir_resource_size_ratio",
        "content_dir_resource_log_size",
        "content_overlay_entropy",
        "content_import_api_count_log",
        "content_avg_imports_per_dll",
        "content_image_base_log",
        "v2_resource_data_entry_count_log",
        "v2_resource_type_icon_count_log",
        "v2_resource_type_version_count_log",
        "v2_resource_type_manifest_count_log",
        "v2_resource_type_dialog_count_log",
        "v2_last_section_entropy",
        "v2_section_max_virtual_raw_ratio_log",
        "v2_api_file_mutation_ratio",
        "v2_import_dll_version_api_ratio",
        "string_benign_vendor_count_log",
        "string_version_resource_count_log",
        "string_script_exec_count_log",
        "string_script_exec_present",
    ]:
        row[name] = "0"
    row["content_overlay_log_size"] = "9.0"
    row["content_overlay_entropy"] = "0.91"
    row["v2_section_max_virtual_raw_ratio_log"] = "4.2"
    row["v2_api_file_mutation_ratio"] = "0.05"
    return row


def test_build_loop145_focus_blinds_public_identity_and_writes_private_map(tmp_path):
    neighbor_csv = tmp_path / "neighbor.csv"
    content_csv = tmp_path / "content.csv"
    focus_csv = tmp_path / "focus.csv"
    private_csv = tmp_path / "private.csv"
    output_json = tmp_path / "summary.json"
    sha = "a" * 64
    write_csv(
        neighbor_csv,
        [
            {
                "support_bucket": "neighbors_support_model_prediction",
                "priority": "0",
                "reason": "model_fn",
                "error_type": "FN",
                "source_path": "data/private/sample.exe",
                "source_sha256": sha,
                "label": "1",
                "prediction": "0",
                "prob_malicious": "0.01",
                "opposite_label_ratio": "0.95",
                "nearest_similarity": "0.99",
            }
        ],
        [
            "support_bucket",
            "priority",
            "reason",
            "error_type",
            "source_path",
            "source_sha256",
            "label",
            "prediction",
            "prob_malicious",
            "opposite_label_ratio",
            "nearest_similarity",
        ],
    )
    content_row = base_content_row(sha)
    write_csv(content_csv, [content_row], list(content_row))

    payload = build_loop145_focus(
        neighbor_csv=neighbor_csv,
        content_review_csv=content_csv,
        output_focus_csv=focus_csv,
        output_private_map_csv=private_csv,
        output_json=output_json,
        max_rows=20,
    )

    assert payload["focus_rows"] == 1
    public_text = focus_csv.read_text(encoding="utf-8-sig")
    assert "source_sha256" not in public_text
    assert "prob_malicious" not in public_text
    assert sha not in public_text
    assert "overlay_present" in public_text
    private_text = private_csv.read_text(encoding="utf-8-sig")
    assert sha in private_text
    assert "prob_malicious" in private_text

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["priority_band_counts"]["critical"] == 1
    assert summary["error_counts"]["fn"] == 1


def test_public_fieldnames_do_not_expose_forbidden_tokens():
    forbidden = [
        "source",
        "sha",
        "hash",
        "cache",
        "path",
        "sample_index",
        "filename",
        "directory",
        "extension",
        "score",
        "prob",
        "threshold",
        "prediction",
        "neighbor",
        "similarity",
    ]
    unsafe = []
    for field in PUBLIC_FIELDNAMES:
        parts = [part for part in field.casefold().replace("-", "_").split("_") if part]
        if any(token in parts for token in forbidden):
            unsafe.append(field)
    assert unsafe == []
