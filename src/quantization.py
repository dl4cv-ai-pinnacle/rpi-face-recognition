"""INT8 ONNX model quantization helper.

Origin: Valenia src/runtime_utils.py — quantize_onnx_model + QuantizationReport.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.onnx_session import suppress_stderr_fd


@dataclass(frozen=True)
class QuantizationReport:
    model_path: str
    quantized_model_path: str
    reused_existing: bool
    quantize_ms: float
    original_size_mb: float
    quantized_size_mb: float

    @property
    def size_delta_mb(self) -> float:
        return self.quantized_size_mb - self.original_size_mb


def quantize_onnx_model(
    model_path: Path,
    quantized_path: Path | None = None,
    *,
    overwrite: bool = False,
) -> tuple[Path, QuantizationReport]:
    """Dynamically quantize MatMul/Gemm ops to INT8."""
    if quantized_path is None:
        quantized_path = model_path.with_name(f"{model_path.stem}.int8{model_path.suffix}")
    if quantized_path.resolve() == model_path.resolve():
        msg = "Quantized model path must differ from the source model path"
        raise ValueError(msg)

    quantized_path.parent.mkdir(parents=True, exist_ok=True)
    reused_existing = quantized_path.exists() and not overwrite
    quantize_ms = 0.0

    if not reused_existing:
        try:
            with suppress_stderr_fd():
                from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception as exc:
            raise RuntimeError(
                "ONNX quantization is not available in this environment. "
                "Use a compatible build environment to create the quantized model."
            ) from exc

        t0 = time.perf_counter()
        quantize_dynamic(
            model_input=str(model_path),
            model_output=str(quantized_path),
            op_types_to_quantize=["MatMul", "Gemm"],
            weight_type=QuantType.QInt8,
            per_channel=False,
        )
        quantize_ms = (time.perf_counter() - t0) * 1000.0

    def _size_mb(path: Path) -> float:
        return float(path.stat().st_size) / (1024.0 * 1024.0)

    return quantized_path, QuantizationReport(
        model_path=str(model_path),
        quantized_model_path=str(quantized_path),
        reused_existing=reused_existing,
        quantize_ms=quantize_ms,
        original_size_mb=_size_mb(model_path),
        quantized_size_mb=_size_mb(quantized_path),
    )
