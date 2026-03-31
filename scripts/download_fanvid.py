#!/usr/bin/env python3
"""Download and prepare the FANVID celebrity subset for video benchmarks.

Downloads per-frame bounding-box + identity annotations from HuggingFace,
then fetches YouTube clips via yt-dlp and extracts LR frames (180x320).
Mugshot gallery images are downloaded for enrollment.

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "fanvid"


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
# Download YouTube videos and extract frames
# ---------------------------------------------------------------------------


def _download_and_extract_clip(
    row: dict[str, str],
    out_dir: Path,
    yt_dlp: str,
) -> bool:
    """Download one YouTube video (if needed) and extract LR frames for the clip.

    Returns True on success, False on failure.
    """
    name = row["Name"].strip()
    clip_id = row["Clip ID"].strip()
    video_id = _fix_video_id(row)
    split = row["Split"].strip().lower()
    start_frame = int(row["Start frame"].strip())
    end_frame = int(row["End frame"].strip())
    clip_dir = out_dir / "frames" / split / name / clip_id
    # Check if already extracted.
    expected_frames = end_frame - start_frame + 1
    if clip_dir.exists() and len(list(clip_dir.glob("*.png"))) >= expected_frames * 0.9:
        return True  # Already done (allow 10% tolerance for edge cases).

    # Download full YouTube video to a temp directory (re-used across clips
    # from the same video via the cache dir).
    cache_dir = out_dir / ".video_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_path = cache_dir / f"{video_id}.mp4"

    if not video_path.exists():
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
            "--merge-output-format", "mp4",
            "-o", str(video_path),
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not video_path.exists():
            print(f"    [FAIL] yt-dlp failed for {video_id}: {result.stderr[:200]}")
            return False

    # Extract and resize frames.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [FAIL] Cannot open {video_path}")
        return False

    clip_dir.mkdir(parents=True, exist_ok=True)
    # Seek to start frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    extracted = 0
    for frame_no in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        lr_frame = cv2.resize(frame, (_LR_WIDTH, _LR_HEIGHT), interpolation=cv2.INTER_CUBIC)
        out_path = clip_dir / f"{frame_no:05d}.png"
        cv2.imwrite(str(out_path), lr_frame)
        extracted += 1

    cap.release()
    return extracted > 0


def download_clips(
    selected: list[dict[str, str]],
    out_dir: Path,
    yt_dlp: str,
) -> tuple[int, int]:
    """Download and extract frames for all selected clips.

    Returns (success_count, fail_count).
    """
    print(f"\n=== Downloading {len(selected)} clips ===")
    # Group by video ID to avoid redundant downloads.
    by_video: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        vid = _fix_video_id(row)
        by_video.setdefault(vid, []).append(row)

    ok = 0
    fail = 0
    for v_idx, (video_id, clips) in enumerate(sorted(by_video.items()), 1):
        n_videos = len(by_video)
        print(f"\n[{v_idx}/{n_videos}] Video {video_id} ({len(clips)} clip(s))")
        for clip in clips:
            name = clip["Name"].strip()
            cid = clip["Clip ID"].strip()
            label = f"  {name}/clip_{cid}"
            # Check if already done.
            split = clip["Split"].strip().lower()
            clip_dir = out_dir / "frames" / split / name / cid
            end_f = int(clip["End frame"].strip())
            start_f = int(clip["Start frame"].strip())
            expected = end_f - start_f + 1
            if clip_dir.exists() and len(list(clip_dir.glob("*.png"))) >= expected * 0.9:
                print(f"{label} [skip, already extracted]")
                ok += 1
                continue

            print(f"{label} extracting frames {start_f}-{end_f} ...")
            if _download_and_extract_clip(clip, out_dir, yt_dlp):
                ok += 1
            else:
                fail += 1

    return ok, fail


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
        "--keep-videos",
        action="store_true",
        help="Keep downloaded YouTube videos in .video_cache/ (default: delete after extraction)",
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

    # Step 3: Download and extract frames.
    ok, fail = download_clips(selected, out_dir, yt_dlp)

    # Step 4: Download mugshot gallery images.
    identities = {r["Name"].strip() for r in selected}
    download_mugshots(mugshots_csv, out_dir, identities)

    # Step 5: Clean up video cache unless --keep-videos.
    cache_dir = out_dir / ".video_cache"
    if cache_dir.exists() and not args.keep_videos:
        size_mb = sum(f.stat().st_size for f in cache_dir.iterdir() if f.is_file()) / 1024 / 1024
        print(f"\nCleaning up video cache ({size_mb:.0f} MB) ...")
        shutil.rmtree(cache_dir)

    # Summary.
    n_frames_dirs = list((out_dir / "frames").rglob("*.png"))
    print("\n=== Done ===")
    print(f"Clips: {ok} succeeded, {fail} failed")
    print(f"Frames: {len(n_frames_dirs)} PNG files")
    print(f"Identities: {len(identities)}")
    print(f"Output: {out_dir}")
    print(f"\nAnnotations: {annotations_csv}")
    print(f"Enrollment:  {out_dir / 'enrollment'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
