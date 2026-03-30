#!/usr/bin/env python3
"""Evaluate face-recognition verification quality on LFW pairs.

Ported from slop/valenia/scripts/evaluate_lfw.py to use the unified pipeline.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from src.config import load_config
from src.contracts import Float32Array, Float64Array, Int32Array
from src.metrics import MemoryStats, enforce_memory_cap
from src.pipeline import FacePipeline, build_pipeline

# ---------------------------------------------------------------------------
# Config override helpers (same pattern as benchmark_pipeline.py)
# ---------------------------------------------------------------------------


def _auto_cast(value: str) -> object:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_overrides(raw: dict[str, object], overrides: list[str]) -> None:
    for override in overrides:
        key, _, value = override.partition("=")
        parts = key.split(".")
        target = raw
        for part in parts[:-1]:
            target = target[part]  # type: ignore[assignment]
        target[parts[-1]] = _auto_cast(value)  # type: ignore[index]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairSample:
    path1: Path
    path2: Path
    is_same: int


@dataclass
class EmbeddingRecord:
    embedding: Float32Array | None
    detect_ms: float
    embed_ms: float
    had_face: bool
    image_ok: bool


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate face-recognition verification quality on LFW pairs."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML pipeline config (default: config.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override, e.g. embedding.quantize_int8=true (repeatable)",
    )
    parser.add_argument("--lfw-dir", default="data/lfw/lfw_funneled")
    parser.add_argument("--train-pairs", default="data/lfw/pairsDevTrain.txt")
    parser.add_argument("--test-pairs", default="data/lfw/pairsDevTest.txt")
    parser.add_argument("--view2-pairs", default="", help="Optional full LFW view2 pairs.txt")
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--max-view2-pairs", type=int, default=0)
    parser.add_argument("--view2-folds", type=int, default=10)
    parser.add_argument("--view2-pairs-per-class", type=int, default=300)
    parser.add_argument("--threshold-steps", type=int, default=2001)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--ram-cap-mb",
        type=float,
        default=4096.0,
        help="Abort if current or peak RSS exceeds this limit; <=0 disables the check",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pair file parsing
# ---------------------------------------------------------------------------


def parse_pairs_file(pairs_file: Path, lfw_dir: Path, max_pairs: int = 0) -> list[PairSample]:
    lines = [line.strip() for line in pairs_file.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise RuntimeError(f"No lines in pairs file: {pairs_file}")

    start_idx = 1 if lines[0].split()[0].isdigit() else 0
    samples: list[PairSample] = []

    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) == 3:
            person = parts[0]
            idx1 = int(parts[1])
            idx2 = int(parts[2])
            path1 = lfw_dir / person / f"{person}_{idx1:04d}.jpg"
            path2 = lfw_dir / person / f"{person}_{idx2:04d}.jpg"
            samples.append(PairSample(path1=path1, path2=path2, is_same=1))
        elif len(parts) == 4:
            person1 = parts[0]
            idx1 = int(parts[1])
            person2 = parts[2]
            idx2 = int(parts[3])
            path1 = lfw_dir / person1 / f"{person1}_{idx1:04d}.jpg"
            path2 = lfw_dir / person2 / f"{person2}_{idx2:04d}.jpg"
            samples.append(PairSample(path1=path1, path2=path2, is_same=0))
        else:
            raise RuntimeError(f"Unexpected pairs format line: {line}")

        if max_pairs > 0 and len(samples) >= max_pairs:
            break

    return samples


def parse_view2_pairs_file(
    pairs_file: Path,
    lfw_dir: Path,
    max_pairs: int = 0,
) -> tuple[list[PairSample], int, int]:
    lines = [line.strip() for line in pairs_file.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise RuntimeError(f"No lines in view2 pairs file: {pairs_file}")

    header = lines[0].split()
    if len(header) < 2 or not header[0].isdigit() or not header[1].isdigit():
        raise RuntimeError("Expected first line in view2 file to be '<folds> <pairs_per_class>'")
    folds = int(header[0])
    pairs_per_class = int(header[1])
    fold_size = pairs_per_class * 2

    samples: list[PairSample] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 3:
            person = parts[0]
            idx1 = int(parts[1])
            idx2 = int(parts[2])
            path1 = lfw_dir / person / f"{person}_{idx1:04d}.jpg"
            path2 = lfw_dir / person / f"{person}_{idx2:04d}.jpg"
            samples.append(PairSample(path1=path1, path2=path2, is_same=1))
        elif len(parts) == 4:
            person1 = parts[0]
            idx1 = int(parts[1])
            person2 = parts[2]
            idx2 = int(parts[3])
            path1 = lfw_dir / person1 / f"{person1}_{idx1:04d}.jpg"
            path2 = lfw_dir / person2 / f"{person2}_{idx2:04d}.jpg"
            samples.append(PairSample(path1=path1, path2=path2, is_same=0))
        else:
            raise RuntimeError(f"Unexpected pairs format line: {line}")
        if max_pairs > 0 and len(samples) >= max_pairs:
            break

    return samples, folds, fold_size


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_embedding(
    image_path: Path,
    pipeline: FacePipeline,
) -> EmbeddingRecord:
    image = cv2.imread(str(image_path))
    if image is None:
        return EmbeddingRecord(
            embedding=None,
            detect_ms=0.0,
            embed_ms=0.0,
            had_face=False,
            image_ok=False,
        )
    image = np.asarray(image, dtype=np.uint8)

    det, detect_ms = pipeline.detect(image)

    if det.kps is None or len(det.boxes) == 0:
        return EmbeddingRecord(
            embedding=None,
            detect_ms=detect_ms,
            embed_ms=0.0,
            had_face=False,
            image_ok=True,
        )

    best_idx = int(np.argmax(det.boxes[:, 4]))
    embedding, embed_ms = pipeline.embed_from_kps(image, det.kps[best_idx])

    return EmbeddingRecord(
        embedding=embedding,
        detect_ms=detect_ms,
        embed_ms=embed_ms,
        had_face=True,
        image_ok=True,
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def score_pairs(
    pairs: list[PairSample],
    embedding_cache: dict[str, EmbeddingRecord],
) -> tuple[Float32Array, Int32Array, int]:
    scores: list[float] = []
    labels: list[int] = []
    dropped = 0

    for pair in pairs:
        rec1 = embedding_cache[str(pair.path1)]
        rec2 = embedding_cache[str(pair.path2)]
        if rec1.embedding is None or rec2.embedding is None:
            dropped += 1
            continue
        scores.append(float(np.dot(rec1.embedding, rec2.embedding)))
        labels.append(pair.is_same)

    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32), dropped


def accuracy_at_threshold(scores: Float32Array, labels: Int32Array, threshold: float) -> float:
    preds = scores >= threshold
    return float(np.mean((preds.astype(np.int32) == labels).astype(np.float32)))


def find_best_threshold(
    scores: Float32Array,
    labels: Int32Array,
    steps: int,
) -> tuple[float, float]:
    if len(scores) == 0:
        return 0.5, 0.0
    thresholds = np.linspace(-1.0, 1.0, num=steps, dtype=np.float32)
    best_thr = 0.5
    best_acc = -1.0
    for thr in thresholds:
        acc = accuracy_at_threshold(scores, labels, float(thr))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


# ---------------------------------------------------------------------------
# ROC / AUC / EER
# ---------------------------------------------------------------------------


def roc_curve_points(
    scores: Float32Array,
    labels: Int32Array,
    steps: int,
) -> tuple[Float32Array, Float64Array, Float64Array]:
    thresholds = np.linspace(1.0, -1.0, num=steps, dtype=np.float32)
    positives = max(1, int(np.sum(labels == 1)))
    negatives = max(1, int(np.sum(labels == 0)))

    tpr_list: list[float] = []
    fpr_list: list[float] = []
    for thr in thresholds:
        preds = scores >= thr
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tpr_list.append(tp / positives)
        fpr_list.append(fp / negatives)

    fpr = np.array(fpr_list, dtype=np.float64)
    tpr = np.array(tpr_list, dtype=np.float64)
    return thresholds, fpr, tpr


def compute_auc(fpr: Float64Array, tpr: Float64Array) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def compute_eer(
    thresholds: Float32Array,
    fpr: Float64Array,
    tpr: Float64Array,
) -> tuple[float, float]:
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = (float(fpr[idx]) + float(fnr[idx])) / 2.0
    return eer, float(thresholds[idx])


def tar_at_far(
    thresholds: Float32Array,
    fpr: Float64Array,
    tpr: Float64Array,
    target_far: float,
) -> tuple[float, float]:
    valid = np.where(fpr <= target_far)[0]
    if len(valid) == 0:
        return 0.0, float(thresholds[-1])
    idx = valid[int(np.argmax(tpr[valid]))]
    return float(tpr[idx]), float(thresholds[idx])


# ---------------------------------------------------------------------------
# View2 10-fold cross-validation
# ---------------------------------------------------------------------------


def evaluate_view2_cv(
    scores: Float32Array,
    labels: Int32Array,
    folds: int,
    fold_size: int,
    threshold_steps: int,
) -> tuple[float, float, float]:
    available_folds = min(folds, len(scores) // fold_size)
    if available_folds < 2:
        return 0.0, 0.0, 0.0

    fold_accs: list[float] = []
    fold_thresholds: list[float] = []

    for fold_idx in range(available_folds):
        start = fold_idx * fold_size
        end = start + fold_size

        test_scores = scores[start:end]
        test_labels = labels[start:end]
        train_scores = np.concatenate([scores[:start], scores[end:]], axis=0)
        train_labels = np.concatenate([labels[:start], labels[end:]], axis=0)

        threshold, _ = find_best_threshold(
            train_scores,
            train_labels,
            steps=max(101, threshold_steps),
        )
        acc = accuracy_at_threshold(test_scores, test_labels, threshold)
        fold_accs.append(acc)
        fold_thresholds.append(threshold)

    return (
        float(statistics.fmean(fold_accs)),
        float(np.std(np.array(fold_accs, dtype=np.float64))),
        float(statistics.fmean(fold_thresholds)),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_runtime_metrics(
    startup_memory: MemoryStats,
    post_init_memory: MemoryStats,
    final_memory: MemoryStats,
) -> None:
    print(
        "memory RSS (startup -> post-init -> final current): "
        f"{startup_memory.current_rss_mb:.2f} -> {post_init_memory.current_rss_mb:.2f} "
        f"-> {final_memory.current_rss_mb:.2f} MiB"
    )
    print(f"peak RSS: {final_memory.peak_rss_mb:.2f} MiB")
    print(
        "model init RSS delta: "
        f"{post_init_memory.current_rss_mb - startup_memory.current_rss_mb:.2f} MiB"
    )


def fmt(value: float) -> str:
    return f"{value:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        args = parse_args()
        startup_memory = enforce_memory_cap(args.ram_cap_mb, "startup")

        # --- load config with overrides ---
        with open(args.config, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _apply_overrides(raw, args.override)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(raw, tmp)
            tmp_path = tmp.name
        config = load_config(tmp_path)
        Path(tmp_path).unlink()

        # --- resolve paths ---
        lfw_dir = Path(args.lfw_dir)
        if not lfw_dir.is_absolute():
            lfw_dir = ROOT / lfw_dir
        train_pairs_file = Path(args.train_pairs)
        if not train_pairs_file.is_absolute():
            train_pairs_file = ROOT / train_pairs_file
        test_pairs_file = Path(args.test_pairs)
        if not test_pairs_file.is_absolute():
            test_pairs_file = ROOT / test_pairs_file
        view2_pairs_file: Path | None = None
        if args.view2_pairs:
            view2_pairs_file = Path(args.view2_pairs)
            if not view2_pairs_file.is_absolute():
                view2_pairs_file = ROOT / view2_pairs_file

        required_paths = [lfw_dir, train_pairs_file, test_pairs_file]
        if view2_pairs_file is not None:
            required_paths.append(view2_pairs_file)

        for path in required_paths:
            if not path.exists():
                print(f"Missing required path: {path}")
                return 2

        # --- build pipeline ---
        pipeline = build_pipeline(config)
        post_init_memory = enforce_memory_cap(args.ram_cap_mb, "pipeline initialization")

        # --- parse pair files ---
        train_pairs = parse_pairs_file(
            train_pairs_file,
            lfw_dir=lfw_dir,
            max_pairs=max(0, args.max_train_pairs),
        )
        test_pairs = parse_pairs_file(
            test_pairs_file,
            lfw_dir=lfw_dir,
            max_pairs=max(0, args.max_test_pairs),
        )
        view2_pairs: list[PairSample] = []
        view2_folds = max(2, args.view2_folds)
        view2_fold_size = max(2, args.view2_pairs_per_class) * 2
        if view2_pairs_file is not None:
            view2_pairs, parsed_folds, parsed_fold_size = parse_view2_pairs_file(
                view2_pairs_file,
                lfw_dir=lfw_dir,
                max_pairs=max(0, args.max_view2_pairs),
            )
            # Use header values by default; allow CLI overrides when max-view2-pairs is set.
            if args.max_view2_pairs == 0:
                view2_folds = parsed_folds
                view2_fold_size = parsed_fold_size

        all_paths = sorted(
            {str(pair.path1) for pair in train_pairs + test_pairs + view2_pairs}
            | {str(pair.path2) for pair in train_pairs + test_pairs + view2_pairs}
        )

        print(f"train pairs requested: {len(train_pairs)}")
        print(f"test pairs requested: {len(test_pairs)}")
        if view2_pairs_file is not None:
            print(f"view2 pairs requested: {len(view2_pairs)}")
        print(f"unique images to process: {len(all_paths)}")

        # --- extract embeddings ---
        cache: dict[str, EmbeddingRecord] = {}
        detect_ms: list[float] = []
        embed_ms: list[float] = []

        t_embed_start = time.perf_counter()
        for idx, path_str in enumerate(all_paths, start=1):
            rec = extract_embedding(Path(path_str), pipeline)
            cache[path_str] = rec
            if rec.image_ok:
                detect_ms.append(rec.detect_ms)
            if rec.embedding is not None:
                embed_ms.append(rec.embed_ms)
            if idx % 250 == 0 or idx == len(all_paths):
                enforce_memory_cap(args.ram_cap_mb, f"LFW embedding pass {idx}")
            if idx % 500 == 0 or idx == len(all_paths):
                print(f"processed images: {idx}/{len(all_paths)}")
        t_embed_end = time.perf_counter()

        # --- score pairs ---
        train_scores, train_labels, train_dropped = score_pairs(train_pairs, cache)
        test_scores, test_labels, test_dropped = score_pairs(test_pairs, cache)
        view2_scores = np.array([], dtype=np.float32)
        view2_labels = np.array([], dtype=np.int32)
        view2_dropped = 0
        if view2_pairs_file is not None:
            view2_scores, view2_labels, view2_dropped = score_pairs(view2_pairs, cache)

        # --- threshold search + metrics ---
        best_thr, train_best_acc = find_best_threshold(
            train_scores, train_labels, steps=max(101, args.threshold_steps)
        )
        test_acc = accuracy_at_threshold(test_scores, test_labels, threshold=best_thr)

        thresholds, test_fpr, test_tpr = roc_curve_points(
            test_scores,
            test_labels,
            steps=max(101, args.threshold_steps),
        )
        test_auc = compute_auc(test_fpr, test_tpr)
        test_eer, eer_thr = compute_eer(thresholds, test_fpr, test_tpr)
        tar_1e2, tar_thr = tar_at_far(thresholds, test_fpr, test_tpr, target_far=1e-2)
        view2_cv_acc = 0.0
        view2_cv_std = 0.0
        view2_cv_thr = 0.0
        if view2_pairs_file is not None and len(view2_scores) > 0:
            view2_cv_acc, view2_cv_std, view2_cv_thr = evaluate_view2_cv(
                view2_scores,
                view2_labels,
                folds=view2_folds,
                fold_size=view2_fold_size,
                threshold_steps=args.threshold_steps,
            )

        # --- aggregate stats ---
        image_ok_count = sum(1 for rec in cache.values() if rec.image_ok)
        face_ok_count = sum(1 for rec in cache.values() if rec.embedding is not None)

        preprocess_s = t_embed_end - t_embed_start
        total_pairs_scored = len(train_scores) + len(test_scores)
        total_pairs_requested = len(train_pairs) + len(test_pairs)
        if view2_pairs_file is not None:
            total_pairs_scored += len(view2_scores)
            total_pairs_requested += len(view2_pairs)

        images_per_s = (len(all_paths) / preprocess_s) if preprocess_s > 0 else 0.0
        pairs_per_s = (total_pairs_scored / preprocess_s) if preprocess_s > 0 else 0.0

        detect_avg = statistics.fmean(detect_ms) if detect_ms else 0.0
        embed_avg = statistics.fmean(embed_ms) if embed_ms else 0.0
        final_memory = enforce_memory_cap(args.ram_cap_mb, "LFW evaluation completion")

        # --- print summary ---
        print("\n=== LFW Validation Summary ===")
        print(
            f"train usable pairs: {len(train_scores)} / {len(train_pairs)} "
            f"(dropped {train_dropped})"
        )
        print(f"test usable pairs: {len(test_scores)} / {len(test_pairs)} (dropped {test_dropped})")
        if view2_pairs_file is not None:
            print(
                f"view2 usable pairs: {len(view2_scores)} / {len(view2_pairs)} "
                f"(dropped {view2_dropped})"
            )
        print(f"image decode success: {image_ok_count} / {len(all_paths)}")
        print(f"face embedding success: {face_ok_count} / {len(all_paths)}")
        print(
            f"face detection+embedding success rate: {fmt(face_ok_count / max(1, len(all_paths)))}"
        )
        print(f"best threshold (train): {fmt(best_thr)}")
        print(f"train accuracy @ best threshold: {fmt(train_best_acc)}")
        print(f"test accuracy @ train threshold: {fmt(test_acc)}")
        print(f"test ROC-AUC: {fmt(test_auc)}")
        print(f"test EER: {fmt(test_eer)} at threshold {fmt(eer_thr)}")
        print(f"test TAR@FAR<=1e-2: {fmt(tar_1e2)} at threshold {fmt(tar_thr)}")
        if view2_pairs_file is not None and len(view2_scores) > 0:
            print(
                f"view2 10-fold accuracy: {fmt(view2_cv_acc)} "
                f"(std {fmt(view2_cv_std)}), mean threshold {fmt(view2_cv_thr)}"
            )
        print(f"avg detection latency/image: {detect_avg:.2f} ms")
        print(f"avg embedding latency/successful face: {embed_avg:.2f} ms")
        print(f"preprocess throughput: {images_per_s:.2f} images/s")
        print(f"effective throughput: {pairs_per_s:.2f} scored pairs/s")
        print_runtime_metrics(
            startup_memory=startup_memory,
            post_init_memory=post_init_memory,
            final_memory=final_memory,
        )

        # --- JSON report ---
        result = {
            "requested": {
                "train_pairs": len(train_pairs),
                "test_pairs": len(test_pairs),
                "total_pairs": total_pairs_requested,
                "unique_images": len(all_paths),
                "view2_pairs": len(view2_pairs),
            },
            "usable": {
                "train_pairs": int(len(train_scores)),
                "test_pairs": int(len(test_scores)),
                "train_dropped": int(train_dropped),
                "test_dropped": int(test_dropped),
                "view2_pairs": int(len(view2_scores)),
                "view2_dropped": int(view2_dropped),
                "image_decode_success": int(image_ok_count),
                "face_embedding_success": int(face_ok_count),
            },
            "metrics": {
                "best_threshold_train": float(best_thr),
                "train_accuracy": float(train_best_acc),
                "test_accuracy": float(test_acc),
                "test_auc": float(test_auc),
                "test_eer": float(test_eer),
                "test_eer_threshold": float(eer_thr),
                "test_tar_far_1e-2": float(tar_1e2),
                "test_tar_far_1e-2_threshold": float(tar_thr),
                "view2_cv_accuracy": float(view2_cv_acc),
                "view2_cv_std": float(view2_cv_std),
                "view2_cv_mean_threshold": float(view2_cv_thr),
            },
            "latency_ms": {
                "detection_avg_per_image": float(detect_avg),
                "embedding_avg_per_successful_face": float(embed_avg),
            },
            "throughput": {
                "preprocess_seconds": float(preprocess_s),
                "images_per_second": float(images_per_s),
                "pairs_per_second": float(pairs_per_s),
                "scored_pairs": int(total_pairs_scored),
            },
            "memory_mb": {
                "startup_current_rss": float(startup_memory.current_rss_mb),
                "post_init_current_rss": float(post_init_memory.current_rss_mb),
                "final_current_rss": float(final_memory.current_rss_mb),
                "final_peak_rss": float(final_memory.peak_rss_mb),
                "model_init_delta_rss": float(
                    post_init_memory.current_rss_mb - startup_memory.current_rss_mb
                ),
            },
            "config": {
                "config_file": str(args.config),
                "overrides": args.override,
                "train_pairs_file": str(train_pairs_file),
                "test_pairs_file": str(test_pairs_file),
                "view2_pairs_file": str(view2_pairs_file) if view2_pairs_file else "",
                "view2_folds": int(view2_folds),
                "view2_fold_size": int(view2_fold_size),
                "ram_cap_mb": float(args.ram_cap_mb),
            },
        }

        if args.output_json:
            out_path = Path(args.output_json)
            if not out_path.is_absolute():
                out_path = ROOT / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Saved JSON report: {out_path}")

        return 0
    except MemoryError as exc:
        print(exc)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
