#!/usr/bin/env python3
"""Export a trusted Axon checkpoint to ONNX for the C++ prediction DLL.

这个脚本只做一件事：把 PyTorch checkpoint 转成 ONNX 推理文件。
训练仍然在 Python 里完成；C++ DLL 只加载导出的 ONNX 文件做预测。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import AxonExperimentConfig  # noqa: E402
from model import AxonMalwareModel  # noqa: E402
from security import load_safe_checkpoint  # noqa: E402


def _prefer_utf8_stdio() -> None:
    """Avoid Windows GBK console failures when PyTorch exporter prints Unicode."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


class AxonOnnxWrapper(torch.nn.Module):
    """Small export wrapper that returns logits instead of a Python dict."""

    def __init__(self, model: AxonMalwareModel):
        super().__init__()
        self.model = model

    def forward(
        self,
        byte_seq: torch.Tensor,
        pe_features: torch.Tensor,
        stat_features: torch.Tensor,
    ) -> torch.Tensor:
        # 这里调用原模型的 forward。原模型返回字典，ONNX 更适合导出单个张量，
        # 所以我们只取里面的 logits，也就是分类器给出的两个原始分数。
        return self.model(byte_seq, pe_features, stat_features=stat_features)["logits"]


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    verify: bool = False,
    byte_length: Optional[int] = None,
) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()

    checkpoint = load_safe_checkpoint(checkpoint_path, map_location="cpu")
    config = AxonExperimentConfig.from_dict(dict(checkpoint["config"]))
    model = AxonMalwareModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    wrapper = AxonOnnxWrapper(model).eval()

    export_byte_length = int(byte_length or config.max_byte_length)
    byte_seq = torch.zeros(1, export_byte_length, dtype=torch.long)
    pe_features = torch.zeros(1, config.pe_feature_dim, dtype=torch.float32)
    stat_features = torch.zeros(1, config.stat_feature_dim, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            (byte_seq, pe_features, stat_features),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["byte_seq", "pe_features", "stat_features"],
            output_names=["logits"],
            dynamic_axes={
                "byte_seq": {0: "batch"},
                "pe_features": {0: "batch"},
                "stat_features": {0: "batch"},
                "logits": {0: "batch"},
            },
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "onnx export dependency"
        raise RuntimeError(
            "ONNX export dependency is missing: "
            f"{missing}. Install export dependencies with:\n"
            '& "E:\\Project\\python\\Axon_v2.6Exp\\vnev\\Scripts\\pip.exe" '
            "install onnx onnxscript"
        ) from exc

    summary = {
        "checkpoint": str(checkpoint_path),
        "onnx": str(output_path),
        "max_byte_length": export_byte_length,
        "pe_feature_dim": int(config.pe_feature_dim),
        "stat_feature_dim": int(config.stat_feature_dim),
        "pe_schema_version": str(config.pe_schema_version),
        "opset": int(opset),
        "verified_with_onnxruntime": False,
    }

    if verify:
        try:
            import numpy as np
            import onnxruntime as ort

            with torch.no_grad():
                torch_logits = wrapper(byte_seq, pe_features, stat_features).detach().numpy()
            session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            ort_logits = session.run(
                ["logits"],
                {
                    "byte_seq": byte_seq.numpy().astype(np.int64),
                    "pe_features": pe_features.numpy().astype(np.float32),
                    "stat_features": stat_features.numpy().astype(np.float32),
                },
            )[0]
            max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))
            summary["verified_with_onnxruntime"] = True
            summary["max_abs_diff"] = max_abs_diff
        except ImportError as exc:
            summary["verify_warning"] = f"onnxruntime or numpy is not installed: {exc}"

    summary_path = output_path.with_suffix(output_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Export Axon checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trusted .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--byte-length", type=int, default=None,
                        help="导出 byte_seq 输入长度（默认取 config.max_byte_length）")
    parser.add_argument("--verify", action="store_true", help="Verify with onnxruntime if available")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    _prefer_utf8_stdio()
    args = parse_args(argv)
    try:
        summary = export_onnx(args.checkpoint, args.output, opset=args.opset,
                              verify=args.verify, byte_length=args.byte_length)
    except Exception as exc:  # noqa: BLE001 - CLI users need a concise action item.
        print(f"[Error] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
