import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from summarize_loop28_residual_strata import summarize  # noqa: E402


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_loop28_residual_strata_joins_noise_by_identity(tmp_path: Path):
    overlap = tmp_path / "overlap.csv"
    noise = tmp_path / "noise.csv"
    _write_csv(
        overlap,
        [
            "source_path",
            "source_sha256",
            "sample_index",
            "label",
            "loop28_error_type",
            "loop28_score",
            "loop37_error_type",
            "byte_ngram_error_type",
            "loop26_blend_error_type",
        ],
        [
            {
                "source_path": "data/a.exe",
                "source_sha256": "sha-a",
                "sample_index": "1",
                "label": "1",
                "loop28_error_type": "FN",
                "loop28_score": "0.1",
                "loop37_error_type": "",
                "byte_ngram_error_type": "FN",
                "loop26_blend_error_type": "FN",
            },
            {
                "source_path": "data/b.exe",
                "source_sha256": "sha-b",
                "sample_index": "1",
                "label": "0",
                "loop28_error_type": "FP",
                "loop28_score": "0.9",
                "loop37_error_type": "FP",
                "byte_ngram_error_type": "FP",
                "loop26_blend_error_type": "FP",
            },
        ],
    )
    _write_csv(
        noise,
        ["source_path", "source_sha256", "sample_index", "noise_bucket"],
        [
            {
                "source_path": "data/a.exe",
                "source_sha256": "sha-a",
                "sample_index": "99",
                "noise_bucket": "severe_fn_conflict_prob_le_0.01",
            }
        ],
    )

    summary = summarize(
        overlap_csv=overlap,
        noise_csv=noise,
        output_json=tmp_path / "summary.json",
        output_csv=tmp_path / "details.csv",
    )

    assert summary["loop28_errors"] == 2
    assert summary["corrected_by"]["loop37"] == 1
    assert summary["noise_bucket_counts_on_loop28_errors"]["severe_fn_conflict_prob_le_0.01"] == 1
    assert summary["noise_bucket_counts_on_loop28_errors"]["not_suspected"] == 1
