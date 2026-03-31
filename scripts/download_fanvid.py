#!/usr/bin/env python3
"""Download and prepare the FANVID celebrity subset for video benchmarks.

Downloads per-frame bounding-box + identity annotations from HuggingFace,
then fetches YouTube clips via yt-dlp and extracts LR frames (180x320).
Mugshot gallery images are downloaded for enrollment.

Each video is processed as a single unit to keep disk usage bounded:
  download (if needed) → extract frames to RAM → delete video → write frames

At most ``--workers`` videos are held on disk simultaneously.  Already-cached
videos are processed first so disk space is reclaimed before new downloads.

Prerequisites:
    sudo apt install yt-dlp ffmpeg  # or: pip install yt-dlp
    # cv2 is already a project dependency

Usage:
    # Download 500 clips (default) across all identities
    uv run --python 3.13 python scripts/download_fanvid.py

    # Limit to 200 clips for a quick test
    uv run --python 3.13 python scripts/download_fanvid.py --max-clips 200

    # Download only test split
    uv run --python 3.13 python scripts/download_fanvid.py --split test --max-clips 100

    # Resume after partial download (already-extracted clips are skipped)
    uv run --python 3.13 python scripts/download_fanvid.py

    # Use 4 download workers
    uv run --python 3.13 python scripts/download_fanvid.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HF_BASE = (
    "https://huggingface.co/datasets/"
    "kv1388/FANVID-Face_and_License_Plate_Recognition_in_Low-Resolution_Videos"
    "/resolve/main"
)
_ANNOTATIONS_URL = f"{_HF_BASE}/data/celebrity_annotations_LR.csv"
_CLIPS_URL = f"{_HF_BASE}/data/dataset_celebs.csv"
_MUGSHOTS_URL = f"{_HF_BASE}/data/Celebrity_mugshot.csv"

_LR_WIDTH = 320
_LR_HEIGHT = 180
_JPEG_QUALITY = 95

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "fanvid"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class _ClipTask:
    clip_dir: Path
    start_frame: int
    end_frame: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, label: str = "") -> None:
    """Download *url* to *dest*, skipping if *dest* already exists."""
    if dest.exists():
        print(f"  [skip] {label or dest.name} (already exists)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    display = label or dest.name
    print(f"  Downloading {display} ...")
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310
    except urllib.error.HTTPError as exc:
        print(f"  [FAIL] {display}: HTTP {exc.code}")
        raise


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from a URL like ``…watch?v=ID`` or ``youtu.be/ID``."""
    m = re.search(r"(?:v=|youtu\.be/)([\w-]+)", url)
    return m.group(1) if m else None


def _fix_video_id(row: dict[str, str]) -> str:
    """Return a usable Video ID, falling back to the You_Tube_URL column.

    The upstream CSV has ``#NAME?`` for IDs that start with ``-`` (Excel
    formula corruption).
    """
    vid = row.get("Video ID", "").strip()
    if vid and vid != "#NAME?":
        return vid
    url = row.get("You_Tube_URL", "").strip()
    if url:
        extracted = _extract_video_id(url)
        if extracted:
            return extracted
    return vid  # return whatever we have as a last resort


def _check_yt_dlp() -> str:
    """Return the yt-dlp executable path, or exit with an error message."""
    for candidate in ("yt-dlp", "yt_dlp"):
        if shutil.which(candidate):
            return candidate
    print(
        "ERROR: yt-dlp not found. Install it with:\n"
        "  sudo apt install yt-dlp   # Debian/Ubuntu\n"
        "  pip install yt-dlp         # or via pip",
        file=sys.stderr,
    )
    sys.exit(1)


def _clip_is_extracted(clip_dir: Path, expected_frames: int) -> bool:
    """Check if a clip directory already has enough extracted frames."""
    if not clip_dir.exists():
        return False
    n_frames = len(list(clip_dir.glob("*.jpg")))
    if n_frames == 0:
        # Support legacy PNG extraction.
        n_frames = len(list(clip_dir.glob("*.png")))
    return n_frames >= expected_frames * 0.9


# ---------------------------------------------------------------------------
# Download annotations from HuggingFace
# ---------------------------------------------------------------------------


def download_annotations(out_dir: Path) -> tuple[Path, Path, Path]:
    """Download the three CSV files we need.  Returns (clips, annotations, mugshots) paths."""
    ann_dir = out_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    clips_csv = ann_dir / "dataset_celebs.csv"
    annotations_csv = ann_dir / "celebrity_annotations_LR.csv"
    mugshots_csv = ann_dir / "Celebrity_mugshot.csv"

    print("=== Downloading annotation CSVs from HuggingFace ===")
    _download_file(_CLIPS_URL, clips_csv, "dataset_celebs.csv")
    _download_file(_ANNOTATIONS_URL, annotations_csv, "celebrity_annotations_LR.csv")
    _download_file(_MUGSHOTS_URL, mugshots_csv, "Celebrity_mugshot.csv")

    return clips_csv, annotations_csv, mugshots_csv


# ---------------------------------------------------------------------------
# Select clips to download
# ---------------------------------------------------------------------------


def select_clips(
    clips_csv: Path,
    *,
    max_clips: int,
    split: str,
) -> list[dict[str, str]]:
    """Select up to *max_clips* celebrity clips, balanced across identities.

    Balancing ensures we get clips from many different people rather than
    exhausting one identity's clips before touching another.
    """
    rows = _read_csv(clips_csv)

    # Filter by split.
    if split != "both":
        rows = [r for r in rows if r["Split"].strip().lower() == split]

    if not rows:
        print(f"No clips found for split={split!r}")
        sys.exit(1)

    # Group by identity.
    by_identity: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        name = row["Name"].strip()
        by_identity.setdefault(name, []).append(row)

    # Round-robin across identities until we reach max_clips.
    selected: list[dict[str, str]] = []
    identity_names = sorted(by_identity.keys())
    idx_per_identity = {name: 0 for name in identity_names}

    while len(selected) < max_clips:
        added_any = False
        for name in identity_names:
            if len(selected) >= max_clips:
                break
            clips = by_identity[name]
            idx = idx_per_identity[name]
            if idx < len(clips):
                selected.append(clips[idx])
                idx_per_identity[name] = idx + 1
                added_any = True
        if not added_any:
            break  # All clips exhausted.

    unique_ids = {r["Name"].strip() for r in selected}
    unique_videos = {_fix_video_id(r) for r in selected}
    print(
        f"\nSelected {len(selected)} clips "
        f"({len(unique_ids)} identities, {len(unique_videos)} YouTube videos)"
    )
    return selected


# ---------------------------------------------------------------------------
# Group clips by video and filter already-extracted
# ---------------------------------------------------------------------------


def _group_clips_by_video(
    selected: list[dict[str, str]],
    out_dir: Path,
) -> dict[str, list[_ClipTask]]:
    """Group selected clips by video ID, skipping already-extracted clips."""
    by_video: dict[str, list[_ClipTask]] = {}
    skipped = 0

    for row in selected:
        vid = _fix_video_id(row)
        name = row["Name"].strip()
        clip_id = row["Clip ID"].strip()
        split = row["Split"].strip().lower()
        start_f = int(row["Start frame"].strip())
        end_f = int(row["End frame"].strip())
        clip_dir = out_dir / "frames" / split / name / clip_id

        if _clip_is_extracted(clip_dir, end_f - start_f + 1):
            skipped += 1
            continue

        by_video.setdefault(vid, []).append(
            _ClipTask(clip_dir=clip_dir, start_frame=start_f, end_frame=end_f)
        )

    if skipped:
        print(f"  ({skipped} clips already extracted, skipped)")
    return by_video


# ---------------------------------------------------------------------------
# Per-video pipeline: download → extract to RAM → delete video → write frames
# ---------------------------------------------------------------------------


def _download_one_video(video_id: str, cache_dir: Path, yt_dlp: str) -> bool:
    """Download a single YouTube video. Returns True on success."""
    video_path = cache_dir / f"{video_id}.mp4"
    if video_path.exists():
        return True

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        yt_dlp,
        "--no-warnings",
        "-f",
        # Prefer H.264 (avc1) — the Pi has no AV1 hardware decoder, so
        # av01 streams can't be read by OpenCV/ffmpeg on this platform.
        "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
        "bestvideo[ext=mp4][vcodec!=av01]+bestaudio[ext=m4a]/"
        "best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(video_path),
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not video_path.exists():
        print(f"    [FAIL] yt-dlp failed for {video_id}: {result.stderr[:200]}")
        return False
    return True


def _process_one_video(
    video_id: str,
    clips: list[_ClipTask],
    cache_dir: Path,
    yt_dlp: str,
    keep_video: bool,
) -> tuple[str, int, int]:
    """Process one video end-to-end.

    1. Download if not cached
    2. Decode + resize clip frames into in-memory JPEG buffers
    3. Delete the video file to reclaim disk space
    4. Write the (much smaller) JPEG frames to disk

    Returns ``(video_id, clips_ok, clips_failed)``.
    """
    video_path = cache_dir / f"{video_id}.mp4"

    # Step 1: download if needed.
    if not video_path.exists() and not _download_one_video(video_id, cache_dir, yt_dlp):
        return video_id, 0, len(clips)

    # Step 2: extract all clip frames into memory.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [FAIL] cannot open {video_id}.mp4")
        return video_id, 0, len(clips)

    # Sort clips by start frame to minimize video seeking.
    sorted_clips = sorted(clips, key=lambda c: c.start_frame)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]

    # Accumulate (output_path, jpeg_bytes) per clip.
    all_buffers: list[tuple[Path, bytes]] = []
    clips_ok = 0

    for clip in sorted_clips:
        clip_frames: list[tuple[Path, bytes]] = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip.start_frame - 1)

        for frame_no in range(clip.start_frame, clip.end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            lr = cv2.resize(frame, (_LR_WIDTH, _LR_HEIGHT), interpolation=cv2.INTER_CUBIC)
            success, buf = cv2.imencode(".jpg", lr, encode_params)
            if success:
                out_path = clip.clip_dir / f"{frame_no:05d}.jpg"
                clip_frames.append((out_path, buf.tobytes()))

        if clip_frames:
            all_buffers.extend(clip_frames)
            clips_ok += 1

    cap.release()

    # Step 3: delete video (and any yt-dlp partial files) to free disk.
    if not keep_video:
        video_path.unlink(missing_ok=True)
        for partial in cache_dir.glob(f"{video_id}.*"):
            partial.unlink(missing_ok=True)

    # Step 4: flush frames to disk (much smaller than the deleted video).
    for path, jpeg_bytes in all_buffers:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg_bytes)

    return video_id, clips_ok, len(clips) - clips_ok


# ---------------------------------------------------------------------------
# Orchestrate parallel video processing
# ---------------------------------------------------------------------------


def process_videos(
    clips_by_video: dict[str, list[_ClipTask]],
    cache_dir: Path,
    yt_dlp: str,
    *,
    workers: int,
    keep_videos: bool,
) -> tuple[int, int]:
    """Process all videos through the download→extract→delete→write pipeline.

    Already-cached videos are submitted first so their disk space is reclaimed
    before new downloads begin.

    Returns ``(total_clips_ok, total_clips_fail)``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached: list[str] = []
    to_download: list[str] = []
    for vid in clips_by_video:
        if (cache_dir / f"{vid}.mp4").exists():
            cached.append(vid)
        else:
            to_download.append(vid)

    total_videos = len(clips_by_video)
    total_clips = sum(len(c) for c in clips_by_video.values())
    print(
        f"\n=== Processing {total_videos} videos ({total_clips} clips): "
        f"{len(cached)} cached, {len(to_download)} to download, "
        f"{workers} workers ==="
    )

    # Process cached videos first to free disk space before downloading more.
    all_video_ids = cached + to_download

    total_ok = 0
    total_fail = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _process_one_video,
                vid,
                clips_by_video[vid],
                cache_dir,
                yt_dlp,
                keep_videos,
            ): vid
            for vid in all_video_ids
        }

        for i, future in enumerate(as_completed(futures), 1):
            vid = futures[future]
            try:
                _, ok, fail = future.result()
                total_ok += ok
                total_fail += fail
            except Exception as exc:
                n_clips = len(clips_by_video[vid])
                total_fail += n_clips
                print(f"    [FAIL] {vid}: {exc}")

            if i % 20 == 0 or i == total_videos:
                print(f"  [{i}/{total_videos}] videos processed ...")

    return total_ok, total_fail


# ---------------------------------------------------------------------------
# Download mugshot gallery images
# ---------------------------------------------------------------------------


def download_mugshots(
    mugshots_csv: Path,
    out_dir: Path,
    identities: set[str],
) -> int:
    """Download mugshot reference images for the selected identities.

    Returns the number of successfully downloaded mugshots.
    """
    print("\n=== Downloading mugshot gallery images ===")
    rows = _read_csv(mugshots_csv)
    enroll_dir = out_dir / "enrollment"
    enroll_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for row in rows:
        name = row["CelebName"].strip()
        if name not in identities:
            continue
        url = row["RefImage"].strip()
        if not url:
            continue

        subject_dir = enroll_dir / name
        subject_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(url).suffix or ".jpg"
        dest = subject_dir / f"mugshot{ext}"

        if dest.exists():
            print(f"  [skip] {name} (already exists)")
            downloaded += 1
            continue

        print(f"  Downloading mugshot for {name} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                dest.write_bytes(resp.read())
            downloaded += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")

    print(f"Downloaded {downloaded}/{len(identities)} mugshots")
    return downloaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download FANVID celebrity clips for video face-recognition benchmarks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--max-clips",
        type=int,
        default=500,
        help="Maximum number of clips to download (default: 500)",
    )
    p.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to download (default: both)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel video-processing threads (default: 3)",
    )
    p.add_argument(
        "--keep-videos",
        action="store_true",
        help="Keep downloaded YouTube videos after extraction (default: delete to save disk)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    yt_dlp = _check_yt_dlp()

    # Step 1: Download annotation CSVs.
    clips_csv, annotations_csv, mugshots_csv = download_annotations(out_dir)

    # Step 2: Select clips.
    selected = select_clips(clips_csv, max_clips=args.max_clips, split=args.split)

    # Step 3: Group by video, filtering out already-extracted clips.
    clips_by_video = _group_clips_by_video(selected, out_dir)

    if not clips_by_video:
        print("\nAll clips already extracted — nothing to do.")
    else:
        # Step 4: Process videos (download → extract to RAM → delete → write).
        cache_dir = out_dir / ".video_cache"
        ok, fail = process_videos(
            clips_by_video,
            cache_dir,
            yt_dlp,
            workers=args.workers,
            keep_videos=args.keep_videos,
        )
        print(f"\nClips: {ok} succeeded, {fail} failed")

    # Step 5: Download mugshot gallery images.
    identities = {r["Name"].strip() for r in selected}
    download_mugshots(mugshots_csv, out_dir, identities)

    # Step 6: Clean up any remaining cache files (partials from failed runs).
    cache_dir = out_dir / ".video_cache"
    if cache_dir.exists() and not args.keep_videos:
        remaining = list(cache_dir.iterdir())
        if remaining:
            size_mb = sum(f.stat().st_size for f in remaining if f.is_file()) / 1024 / 1024
            print(f"\nCleaning up {len(remaining)} leftover cache files ({size_mb:.0f} MB) ...")
            shutil.rmtree(cache_dir)
        else:
            cache_dir.rmdir()

    # Summary.
    frames_dir = out_dir / "frames"
    n_frames = 0
    if frames_dir.exists():
        n_frames = len(list(frames_dir.rglob("*.jpg")))
        n_frames += len(list(frames_dir.rglob("*.png")))
    print("\n=== Done ===")
    print(f"Frames: {n_frames} image files")
    print(f"Identities: {len(identities)}")
    print(f"Output: {out_dir}")
    print(f"\nAnnotations: {annotations_csv}")
    print(f"Enrollment:  {out_dir / 'enrollment'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
