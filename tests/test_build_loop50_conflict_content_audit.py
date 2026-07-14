import csv
import json

import numpy as np

from scripts.build_loop50_conflict_content_audit import build_audit


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_loop50_content_audit_detects_cache_mismatch(tmp_path):
    source = tmp_path / "sample.bin"
    source.write_bytes(b"MZnotreallyape")
    cache = tmp_path / "sample.npz"
    np.savez(
        cache,
        byte_sequence=np.zeros(8192, dtype=np.uint8),
        pe_features=np.zeros(256, dtype=np.float32),
        stat_features=np.zeros(49, dtype=np.float32),
        lightweight_features=np.zeros(256, dtype=np.float32),
        label=np.array(1, dtype=np.int64),
        source_sha256=np.array("wrong"),
    )

    queue_csv = tmp_path / "queue.csv"
    split_csv = tmp_path / "split.csv"
    manifest_json = tmp_path / "manifest.json"
    output_csv = tmp_path / "audit.csv"
    output_json = tmp_path / "audit.json"
    rows = [
        {
            "review_priority_rank": "1",
            "review_lane": "A_unfixed_severe_conflict",
            "conflict_bucket": "severe_fp_conflict_prob_ge_0.99",
            "source_sha256": "abc",
            "source_path": str(source),
            "sample_index": "10",
            "label": "0",
            "loop28_error_type": "FP",
            "loop28_score": "0.99",
            "manual_label_verdict": "",
            "recommended_action": "",
        }
    ]
    _write_csv(queue_csv, rows)
    _write_csv(split_csv, [{"source_path": str(source), "label": "0", "sample_index": "10", "split": "test"}])
    manifest_json.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "source_path": str(source),
                        "label": 0,
                        "cache_path": str(cache),
                        "source_sha256": "abc",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_audit(
        queue_csv=queue_csv,
        split_csv=split_csv,
        manifest_json=manifest_json,
        output_csv=output_csv,
        output_json=output_json,
        lane=None,
        limit=None,
    )

    assert report["rows"] == 1
    assert report["objective_issue_row_count"] == 1
    assert output_csv.exists()
    assert "cache_label_mismatch" in output_csv.read_text(encoding="utf-8")
    assert "cache_source_sha256_mismatch" in output_csv.read_text(encoding="utf-8")


def test_loop50_content_audit_accepts_loop63_queue_fields(tmp_path):
    source = tmp_path / "sample.exe"
    source.write_bytes(b"MZnotreallyape")
    cache = tmp_path / "sample.npz"
    np.savez(
        cache,
        byte_sequence=np.zeros(8192, dtype=np.uint8),
        pe_features=np.zeros(256, dtype=np.float32),
        stat_features=np.zeros(49, dtype=np.float32),
        lightweight_features=np.zeros(256, dtype=np.float32),
        label=np.array(1, dtype=np.int64),
        source_sha256=np.array("abc"),
    )

    queue_csv = tmp_path / "loop63_queue.csv"
    split_csv = tmp_path / "split.csv"
    manifest_json = tmp_path / "manifest.json"
    output_csv = tmp_path / "audit.csv"
    output_json = tmp_path / "audit.json"
    rows = [
        {
            "review_priority_rank": "1",
            "review_lane": "A_persistent_error_in_high_conflict_queue",
            "loop39_conflict_bucket": "severe_fn_conflict_prob_le_0.01",
            "source_sha256": "abc",
            "source_path": str(source),
            "sample_index": "10",
            "label": "1",
            "loop57_error_type": "FN",
            "loop57_final_prob": "0.001",
            "manual_label_verdict": "",
            "recommended_action": "",
        }
    ]
    _write_csv(queue_csv, rows)
    _write_csv(split_csv, [{"source_path": str(source), "label": "1", "sample_index": "10", "split": "test"}])
    manifest_json.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "source_path": str(source),
                        "label": 1,
                        "cache_path": str(cache),
                        "source_sha256": "abc",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_audit(
        queue_csv=queue_csv,
        split_csv=split_csv,
        manifest_json=manifest_json,
        output_csv=output_csv,
        output_json=output_json,
        lane="A_persistent_error_in_high_conflict_queue",
        limit=None,
    )
    audit_text = output_csv.read_text(encoding="utf-8")

    assert report["rows"] == 1
    assert report["error_type_counts"] == {"FN": 1}
    assert "loop57_error_type" in audit_text
    assert "severe_fn_conflict_prob_le_0.01" in audit_text
