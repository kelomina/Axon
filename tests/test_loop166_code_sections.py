from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for search_path in (SRC_DIR, SCRIPTS_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from run_loop166_code_section_extractor_probe import run_probe  # noqa: E402

from loop166.code_sections import (  # noqa: E402
    IMAGE_SCN_MEM_EXECUTE,
    extract_executable_code,
    plan_executable_spans,
)


class FakeSection:
    def __init__(self, start: int, size: int, *, executable: bool = True):
        self.PointerToRawData = start
        self.SizeOfRawData = size
        self.Characteristics = IMAGE_SCN_MEM_EXECUTE if executable else 0


class FakePE:
    def __init__(self, sections):
        self.sections = sections
        self.closed = False

    def get_warnings(self):
        return ["warning-a"]

    def close(self):
        self.closed = True


def test_span_plan_merges_overlaps_without_repeating_code_bytes():
    plan = plan_executable_spans(
        [
            FakeSection(2, 5),
            FakeSection(5, 5),
            FakeSection(15, 3),
            FakeSection(1, 2, executable=False),
        ],
        file_size=20,
    )

    assert plan.spans == ((2, 10), (15, 18))
    assert plan.declared_executable_sections == 3
    assert plan.declared_raw_bytes == 13
    assert plan.overlap_bytes_removed == 2
    assert plan.missing_reason is None


def test_any_out_of_bounds_executable_span_fails_the_whole_sample():
    plan = plan_executable_spans(
        [FakeSection(2, 5), FakeSection(18, 4)],
        file_size=20,
    )

    assert plan.spans == ()
    assert plan.missing_reason == "invalid_executable_section_span"


def test_extractor_concatenates_merged_file_order_spans():
    parsed = FakePE([FakeSection(8, 4), FakeSection(2, 4)])
    extraction = extract_executable_code(
        bytes(range(16)),
        pe_factory=lambda **_kwargs: parsed,
    )

    assert extraction.available is True
    assert extraction.spans == ((2, 6), (8, 12))
    assert extraction.code_bytes == bytes([2, 3, 4, 5, 8, 9, 10, 11])
    assert extraction.parser_warning_count == 1
    assert parsed.closed is True


def test_canonical_phase_a_probe_is_train_only_and_non_promotable():
    payload = run_probe()

    assert payload["counts"]["denominator"] == 256
    assert payload["counts"]["success"] + payload["counts"]["missing"] == 256
    assert payload["counts"]["silent_drop"] == 0
    assert payload["protocol"]["training_performed"] is False
    assert payload["protocol"]["quality_metrics_computed"] is False
    assert payload["protocol"]["val_test_or_full_access"] is False
    assert payload["protocol"]["public_key_required"] is False
    assert payload["ready_for"]["five_fold_oof"] is False
    assert payload["ready_for"]["promotion"] is False
    assert payload["target_status"]["target_achieved"] is False


def test_phase_a_decision_manifest_binds_final_artifacts():
    manifest_path = (
        PROJECT_ROOT
        / "manifests"
        / "roadmap_9997"
        / "loop166_code_section_foundation"
        / "phase_a_decision.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    for binding in payload["bindings"].values():
        raw = (PROJECT_ROOT / binding["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
    assert payload["decision"] == "phase_a_extractor_gate_pass"
    assert payload["ready_for"]["five_fold_oof"] is False
    assert payload["target_achieved"] is False
