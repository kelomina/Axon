import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_loop28_conflict_adjudication_queue import build_queue  # noqa: E402


def _write_residual(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "source_path",
        "source_sha256",
        "sample_index",
        "label",
        "loop28_error_type",
        "loop28_score",
        "loop37_score",
        "byte_ngram_score",
        "loop26_blend_score",
        "noise_bucket",
        "corrected_by_any_compared_model",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_loop28_conflict_queue_selects_high_conflicts_and_clears_manual_fields(tmp_path: Path):
    residual = tmp_path / "residual.csv"
    _write_residual(
        residual,
        [
            {
                "source_path": r"E:\repo\data\待拉黑\2026-03\2026-03-01\a.exe",
                "source_sha256": "sha-a",
                "sample_index": "1",
                "label": "1",
                "loop28_error_type": "FN",
                "loop28_score": "0.001",
                "loop37_score": "0.1",
                "byte_ngram_score": "0.2",
                "loop26_blend_score": "0.1",
                "noise_bucket": "severe_fn_conflict_prob_le_0.01",
                "corrected_by_any_compared_model": "False",
            },
            {
                "source_path": r"E:\repo\data\待加入白名单\b.exe",
                "source_sha256": "sha-b",
                "sample_index": "2",
                "label": "0",
                "loop28_error_type": "FP",
                "loop28_score": "0.52",
                "loop37_score": "0.4",
                "byte_ngram_score": "0.4",
                "loop26_blend_score": "0.4",
                "noise_bucket": "near_threshold_error_le_0.05",
                "corrected_by_any_compared_model": "True",
            },
            {
                "source_path": r"E:\repo\data\待拉黑\2026-03\2026-03-01\a-copy.exe",
                "source_sha256": "sha-a",
                "sample_index": "3",
                "label": "1",
                "loop28_error_type": "FN",
                "loop28_score": "0.002",
                "loop37_score": "0.1",
                "byte_ngram_score": "0.2",
                "loop26_blend_score": "0.1",
                "noise_bucket": "severe_fn_conflict_prob_le_0.01",
                "corrected_by_any_compared_model": "False",
            },
        ],
    )

    summary = build_queue(
        residual_csv=residual,
        output_csv=tmp_path / "queue.csv",
        output_json=tmp_path / "queue.json",
    )
    rows = list(csv.DictReader((tmp_path / "queue.csv").open("r", encoding="utf-8-sig")))

    assert summary["rows"] == 2
    assert summary["lane_counts"] == {"A_unfixed_severe_conflict": 2}
    assert summary["duplicate_sha_groups"] == 1
    assert summary["duplicate_sha_extra_rows"] == 1
    assert rows[0]["manual_label_verdict"] == ""
    assert rows[0]["recommended_action"] == ""
    assert rows[0]["duplicate_sha_group"] == "True"
    assert rows[0]["source_sha256_group_size"] == "2"
    assert "Re-sample one fresh valid candidate" in rows[0]["replacement_rule"]
