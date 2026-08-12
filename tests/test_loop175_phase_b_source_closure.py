from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.phase_b_source_closure import (  # noqa: E402
    SOURCE_CLOSURE_SCHEMA,
    SOURCE_RELATIVE_PATHS,
    PhaseBSourceClosureError,
    _binding,
    _prefix_binding,
    _safe_project_file,
    _write_exclusive_canonical,
    build_phase_b_source_closure,
    validate_phase_b_source_closure,
)


@pytest.fixture(scope="module")
def live_payload() -> dict[str, object]:
    return build_phase_b_source_closure(PROJECT_ROOT)


def test_live_closure_is_complete_canonical_and_identity_free(
    live_payload: dict[str, object],
) -> None:
    validate_phase_b_source_closure(PROJECT_ROOT, live_payload)
    assert live_payload["schema"] == SOURCE_CLOSURE_SCHEMA
    assert len(live_payload["source_files"]) == len(SOURCE_RELATIVE_PATHS)  # type: ignore[arg-type]
    evidence = live_payload["evidence_bindings"]
    assert "full_shape_gpu_smoke" in evidence  # type: ignore[operator]
    assert all(  # type: ignore[union-attr]
        binding["path"] != "reports/roadmap_9997/loop175/phase_b_full_shape_gpu_smoke.json"
        for binding in live_payload["source_files"]
    )
    boundaries = live_payload["aggregate_boundaries"]
    assert boundaries == {  # type: ignore[comparison-overlap]
        "identity_fields_persisted": False,
        "raw_pe_files_opened": 0,
        "val_rows_opened": 0,
        "test10k_rows_opened": 0,
        "legacy_full_rows_opened": 0,
        "sealed_window_rows_opened": 0,
        "prediction_rows_opened": 0,
        "model_fits": 0,
        "training_runs": 0,
    }
    encoded = json.dumps(
        live_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii") + b"\n"
    assert json.loads(encoded) == live_payload


def test_closure_rejects_missing_fields_and_sha_drift(live_payload: dict[str, object]) -> None:
    missing = copy.deepcopy(live_payload)
    missing.pop("decision")
    with pytest.raises(PhaseBSourceClosureError, match="fields"):
        validate_phase_b_source_closure(PROJECT_ROOT, missing)

    drifted = copy.deepcopy(live_payload)
    drifted["source_files"][0]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(PhaseBSourceClosureError, match="drifted"):
        validate_phase_b_source_closure(PROJECT_ROOT, drifted)


def test_binding_rejects_escape_link_and_expected_sha_drift(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source.txt"
    source.write_bytes(b"source")
    with pytest.raises(PhaseBSourceClosureError, match="escapes|canonical"):
        _safe_project_file(root, "../source.txt")
    with pytest.raises(PhaseBSourceClosureError, match="drifted"):
        _binding(root, "source.txt", expected_sha256="0" * 64)

    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(PhaseBSourceClosureError, match="link|reparse"):
        _safe_project_file(root, "link.txt")


def test_train_prefix_binding_reads_only_the_exact_prefix(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    prefix = b"train-only-prefix\n"
    (root / "split.csv").write_bytes(prefix + b"forbidden-later-split-bytes")
    binding = _prefix_binding(
        root,
        "split.csv",
        prefix_bytes=len(prefix),
        expected_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
    )
    assert binding["bytes"] == len(prefix)
    assert binding["read_scope"] == "exact_train_prefix_only_no_later_split_bytes"


def test_exclusive_writer_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "root"
    output_parent = root / "out"
    output_parent.mkdir(parents=True)
    payload = {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "source_files": [],
        "decision": "static-test",
    }
    receipt = _write_exclusive_canonical(root, "out/closure.json", payload)
    assert receipt.sha256 == hashlib.sha256(receipt.path.read_bytes()).hexdigest()
    assert receipt.path.read_bytes().endswith(b"\n")
    with pytest.raises(PhaseBSourceClosureError, match="overwrite"):
        _write_exclusive_canonical(root, "out/closure.json", payload)
    with pytest.raises(PhaseBSourceClosureError, match="escapes|canonical"):
        _write_exclusive_canonical(root, "../closure.json", payload)


def test_windows_path_objects_are_not_accepted_as_posix_closure_paths() -> None:
    with pytest.raises(PhaseBSourceClosureError, match="POSIX-relative"):
        _safe_project_file(PROJECT_ROOT, Path("src\\loop175\\phase_b_contract.py"))
