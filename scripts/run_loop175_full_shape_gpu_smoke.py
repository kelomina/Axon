#!/usr/bin/env python3
"""Measure one production-shape Loop175 microbatch without opening raw data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loop175.model import RegionNet  # noqa: E402
from src.loop175.phase_b_contract import (  # noqa: E402
    load_phase_b_protocol,
    sha256_file,
    validate_bound_evidence,
    write_exclusive_json,
)
from src.loop175.phase_b_training import FailClosedRegionNet  # noqa: E402


def run_smoke(*, iterations: int = 6) -> dict[str, object]:
    if iterations < 4:
        raise ValueError("iterations must include at least two warmup and two measured steps")
    protocol = load_phase_b_protocol(PROJECT_ROOT)
    validate_bound_evidence(PROJECT_ROOT, protocol)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Loop175 production-shape smoke requires BF16 CUDA")

    torch.manual_seed(175)
    torch.cuda.manual_seed_all(175)
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = FailClosedRegionNet(RegionNet()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-2)
    tokens = torch.randint(0, 256, (2, 16, 8192), device=device, dtype=torch.int64)
    lengths = torch.full((2, 16), 8192, device=device, dtype=torch.int64)
    region_types = torch.arange(16, device=device, dtype=torch.int64).remainder(5).add(1).repeat(2, 1)
    buckets = torch.arange(16, device=device, dtype=torch.int64).repeat(2, 1)
    b0 = torch.randn(2, 571, device=device)
    labels = torch.tensor([0, 1], device=device)

    step_seconds: list[float] = []
    losses: list[float] = []
    for _iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(tokens, lengths, region_types, buckets, buckets, b0)
            loss = functional.cross_entropy(output["fusion_logits"].float(), labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))

    stable = step_seconds[2:]
    stable_mean = sum(stable) / len(stable)
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    finite = all(torch.isfinite(torch.tensor(losses)).tolist())
    projected_epoch_seconds = stable_mean * 8_000
    projected_twelve_epoch_hours = projected_epoch_seconds * 12 / 3_600
    maximum_gpu = int(protocol.payload["resource_contract"]["maximum_gpu_allocated_bytes"])
    passed = finite and peak_allocated <= maximum_gpu and projected_twelve_epoch_hours <= 6.0
    return {
        "schema": "axon_loop175_full_shape_gpu_smoke_v1",
        "loop_id": "Loop175",
        "claim_scope": "synthetic_resource_and_throughput_only_not_model_quality_or_training_authorization",
        "inputs": {
            "protocol_sha256": protocol.sha256,
            "model_sha256": sha256_file(PROJECT_ROOT / "src/loop175/model.py"),
            "phase_b_training_sha256": sha256_file(
                PROJECT_ROOT / "src/loop175/phase_b_training.py"
            ),
            "shape": [2, 16, 8192],
            "b0_shape": [2, 571],
            "autocast": "bf16",
            "iterations": iterations,
            "warmup_iterations": 2,
            "raw_rows_opened": 0,
            "val_test_or_full_rows_opened": 0,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "measurements": {
            "step_seconds": step_seconds,
            "stable_mean_seconds_per_microbatch": stable_mean,
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "finite": finite,
            "projected_seconds_per_16000_row_epoch": projected_epoch_seconds,
            "projected_hours_per_12_epoch_arm_fold": projected_twelve_epoch_hours,
        },
        "gates": {
            "maximum_gpu_allocated_bytes": maximum_gpu,
            "maximum_hours_per_arm_fold": 6.0,
            "passed": passed,
        },
        "decision": (
            "full_shape_resource_gate_pass_phase_b_implementation_may_continue"
            if passed
            else "close_loop175_current_regionnet_recipe"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=6)
    arguments = parser.parse_args()
    payload = run_smoke(iterations=arguments.iterations)
    write_exclusive_json(arguments.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if payload["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

