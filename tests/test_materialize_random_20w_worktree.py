import csv
import shutil
import sys
import uuid
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import materialize_random_20w_worktree as materialize_module  # noqa: E402
from materialize_random_20w_worktree import link_or_verify, materialize_worktree  # noqa: E402


@contextmanager
def _case_dir(name: str):
    path = Path(__file__).resolve().parents[1] / ".tmp_test_artifacts" / name / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_sample(path: Path, payload: bytes = b"MZpayload") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_materialize_worktree_preserves_relative_structure_with_hardlinks():
    with _case_dir("random_20w_materialize") as tmp_path:
        source_root = tmp_path / "data"
        worktree_root = tmp_path / "worktree"
        split_path = tmp_path / "split.csv"
        out_split = tmp_path / "worktree_split.csv"
        summary_json = tmp_path / "summary.json"

        src_a = _write_sample(source_root / "待加入白名单" / "a.exe")
        src_b = _write_sample(source_root / "待拉黑" / "2024-01" / "b.exe")
        with split_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_path", "label", "sample_index", "split"])
            writer.writeheader()
            writer.writerow({"source_path": str(src_a), "label": "0", "sample_index": "0", "split": "train"})
            writer.writerow({"source_path": str(src_b), "label": "1", "sample_index": "1", "split": "test"})

        summary = materialize_worktree(
            source_split=split_path,
            source_data_dir=source_root,
            worktree_root=worktree_root,
            output_split=out_split,
            summary_json=summary_json,
            link_mode="hardlink",
            workers=2,
        )

        rewritten = list(csv.DictReader(out_split.open("r", encoding="utf-8-sig", newline="")))
        summary_text = summary_json.read_text(encoding="utf-8")
        assert summary["total_rows"] == 2
        assert summary["link_status_counts"]["linked"] == 2
        assert "axon_random_20w_worktree_v1" in summary_text
        assert rewritten[0]["original_source_path"] == str(src_a)
        assert Path(rewritten[0]["source_path"]).exists()
        assert rewritten[1]["original_source_path"] == str(src_b)


def test_materialize_worktree_rejects_out_of_root_paths():
    with _case_dir("random_20w_materialize_outside") as tmp_path:
        source_root = tmp_path / "data"
        outside = tmp_path / "outside.exe"
        source_root.mkdir(parents=True, exist_ok=True)
        _write_sample(outside)
        split_path = tmp_path / "split.csv"
        split_path.write_text(
            "source_path,label,sample_index,split\n"
            f"{outside},0,0,train\n",
            encoding="utf-8-sig",
        )

        try:
            materialize_worktree(
                source_split=split_path,
                source_data_dir=source_root,
                worktree_root=tmp_path / "worktree",
                output_split=tmp_path / "out.csv",
                summary_json=tmp_path / "summary.json",
                link_mode="hardlink",
                workers=1,
            )
        except ValueError as exc:
            assert "outside source data root" in str(exc)
        else:
            raise AssertionError("Expected out-of-root path to fail")


def test_link_or_verify_rejects_same_size_different_payload():
    with _case_dir("random_20w_materialize_conflict") as tmp_path:
        source = _write_sample(tmp_path / "source.exe", b"MZAAAA")
        destination = _write_sample(tmp_path / "destination.exe", b"MZBBBB")

        try:
            link_or_verify(source, destination, mode="copy")
        except FileExistsError as exc:
            assert "different payload" in str(exc)
        else:
            raise AssertionError("Expected same-size payload conflict to fail")


def test_materialize_worktree_writes_failure_report_for_missing_source():
    with _case_dir("random_20w_materialize_failure") as tmp_path:
        source_root = tmp_path / "data"
        source_root.mkdir(parents=True, exist_ok=True)
        split_path = tmp_path / "split.csv"
        out_split = tmp_path / "worktree_split.csv"
        missing = source_root / "missing.exe"
        with split_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_path", "label", "sample_index", "split"])
            writer.writeheader()
            writer.writerow({"source_path": str(missing), "label": "1", "sample_index": "0", "split": "train"})

        summary = materialize_worktree(
            source_split=split_path,
            source_data_dir=source_root,
            worktree_root=tmp_path / "worktree",
            output_split=out_split,
            summary_json=tmp_path / "summary.json",
            link_mode="hardlink",
            workers=1,
        )

        rewritten = list(csv.DictReader(out_split.open("r", encoding="utf-8-sig", newline="")))
        assert rewritten == []
        assert summary["planned_rows"] == 1
        assert summary["failed_rows"] == 1
        assert summary["ready_for_cache_recovery"] is False
        assert summary["failure_examples"][0]["source_path"] == str(missing)


def test_materialize_worktree_keeps_order_with_bounded_parallel_pending(monkeypatch):
    with _case_dir("random_20w_materialize_parallel_order") as tmp_path:
        source_root = tmp_path / "data"
        split_path = tmp_path / "split.csv"
        out_split = tmp_path / "worktree_split.csv"
        rows = []
        for index in range(6):
            source = _write_sample(source_root / f"{index}.exe", f"MZ{index}".encode("ascii"))
            rows.append({"source_path": str(source), "label": str(index % 2), "sample_index": str(index), "split": "test"})
        with split_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_path", "label", "sample_index", "split"])
            writer.writeheader()
            writer.writerows(rows)

        class FakeExecutor:
            max_observed = 0
            current_pending = 0

            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, payload):
                future = Future()
                future.set_result(fn(payload))
                FakeExecutor.current_pending += 1
                FakeExecutor.max_observed = max(FakeExecutor.max_observed, FakeExecutor.current_pending)
                return future

        original_flush = materialize_module._flush_ready

        def tracking_flush(completed_by_index, next_to_write, *args, **kwargs):
            next_index = original_flush(completed_by_index, next_to_write, *args, **kwargs)
            FakeExecutor.current_pending = len(completed_by_index)
            return next_index

        monkeypatch.setattr(materialize_module, "ThreadPoolExecutor", FakeExecutor)
        monkeypatch.setattr(materialize_module, "_flush_ready", tracking_flush)

        summary = materialize_worktree(
            source_split=split_path,
            source_data_dir=source_root,
            worktree_root=tmp_path / "worktree",
            output_split=out_split,
            summary_json=tmp_path / "summary.json",
            link_mode="hardlink",
            workers=2,
            max_pending=3,
        )

        rewritten = list(csv.DictReader(out_split.open("r", encoding="utf-8-sig", newline="")))
        assert [row["sample_index"] for row in rewritten] == [str(index) for index in range(6)]
        assert FakeExecutor.max_observed <= 3
        assert summary["max_pending_tasks"] == 3
        assert summary["failed_rows"] == 0
