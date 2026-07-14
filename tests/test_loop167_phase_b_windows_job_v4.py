from __future__ import annotations

import os

import pytest

from src.loop167_phase_b.windows_job_v4 import WindowsJob, WindowsJobError, probe_windows_job_ready


def test_job_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        WindowsJob.create(memory_limit_bytes=0)


def test_job_probe_is_nonassigning_and_reports_platform_readiness() -> None:
    ready, detail = probe_windows_job_ready(memory_limit_bytes=64 * 1024 * 1024)
    if os.name == "nt":
        assert ready is True
        assert detail is None
    else:
        assert ready is False
        assert detail is not None


def test_assigning_a_closed_job_is_rejected() -> None:
    job = WindowsJob(handle=1, memory_limit_bytes=1, _closed=True)
    with pytest.raises(WindowsJobError, match="closed"):
        job.assign_current_process()
