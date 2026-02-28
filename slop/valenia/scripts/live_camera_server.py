#!/usr/bin/env python3
"""Serve a live Pi camera MJPEG stream with face detections over HTTP."""

from __future__ import annotations

import argparse
import html
import sys
import threading
import time
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arcface import ArcFaceEmbedder
from gallery import EnrollmentResult, GalleryMatch, GalleryStore
from pipeline import FacePipeline
from runtime_utils import MemoryStats, UInt8Array, enforce_memory_cap
from scrfd import SCRFDDetector
from tracking import SimpleFaceTracker, Track

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Valenia Live Camera</title>
  <style>
    body {
      margin: 0;
      font-family: sans-serif;
      background: #101820;
      color: #f5f5f5;
      display: grid;
      min-height: 100vh;
      place-items: center;
    }
    main {
      width: min(100%, 960px);
      padding: 1rem;
      box-sizing: border-box;
    }
    h1 {
      margin: 0 0 0.5rem;
      font-size: 1.5rem;
    }
    p {
      margin: 0 0 1rem;
      opacity: 0.85;
    }
    section {
      margin: 0 0 1rem;
      padding: 1rem;
      border: 1px solid #2b3a42;
      border-radius: 0.5rem;
      background: rgba(255, 255, 255, 0.03);
    }
    form {
      display: grid;
      gap: 0.75rem;
    }
    label {
      display: grid;
      gap: 0.35rem;
      font-size: 0.95rem;
    }
    input,
    button {
      font: inherit;
    }
    input[type="text"],
    input[type="file"] {
      padding: 0.6rem;
      border-radius: 0.4rem;
      border: 1px solid #2b3a42;
      background: #13232d;
      color: #f5f5f5;
    }
    button {
      width: fit-content;
      padding: 0.7rem 1rem;
      border: 0;
      border-radius: 0.4rem;
      background: #3aa17e;
      color: #08120d;
      font-weight: 700;
      cursor: pointer;
    }
    img {
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid #2b3a42;
      border-radius: 0.5rem;
      background: #000;
    }
    code {
      color: #b9fbc0;
    }
  </style>
</head>
<body>
  <main>
    <h1>Valenia Live Camera</h1>
    <p>MJPEG stream: <code>/stream.mjpg</code></p>
    <section>
      <form action="/enroll" method="post" enctype="multipart/form-data">
        <label>
          Name
          <input type="text" name="name" required>
        </label>
        <label>
          Photos
          <input type="file" name="photos" accept="image/*" multiple required>
        </label>
        <button type="submit">Enroll Identity</button>
      </form>
    </section>
    <img src="/stream.mjpg" alt="Live camera stream">
  </main>
</body>
</html>
"""


@dataclass
class StreamSnapshot:
    frame_id: int
    jpeg_bytes: bytes | None
    error: str | None


@dataclass(frozen=True)
class OverlayFace:
    track: Track
    match: GalleryMatch | None


@dataclass(frozen=True)
class TrackingOverlay:
    faces: list[OverlayFace]
    detect_ms: float
    track_ms: float
    embed_ms_total: float
    gallery_size: int


class CameraStreamer:
    """Capture frames in one background loop and keep only the latest JPEG."""

    def __init__(
        self,
        args: argparse.Namespace,
        pipeline: FacePipeline,
        gallery: GalleryStore,
    ) -> None:
        self.args = args
        self.pipeline = pipeline
        self.gallery = gallery
        self._camera = None
        self._condition = threading.Condition()
        self._frame_id = 0
        self._jpeg_bytes: bytes | None = None
        self._error: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_interval = 1.0 / args.fps if args.fps > 0 else 0.0
        self._frames_processed = 0
        self._memory: MemoryStats = enforce_memory_cap(args.ram_cap_mb, "live camera startup")
        self._tracker = SimpleFaceTracker(
            iou_threshold=args.track_iou_thresh,
            max_missed=args.track_max_missed,
            smoothing=args.track_smoothing,
        )
        self._track_matches: dict[int, GalleryMatch] = {}

    def start(self) -> None:
        from picamera2 import Picamera2

        try:
            self._camera = Picamera2()
        except IndexError as exc:
            raise RuntimeError(
                "No camera detected by Picamera2. Check cable/enablement and rerun."
            ) from exc

        config = self._camera.create_video_configuration(
            main={"size": (self.args.width, self.args.height), "format": "RGB888"}
        )
        self._camera.configure(config)
        self._camera.start()

        time.sleep(1.0)

        self._thread = threading.Thread(target=self._capture_loop, name="live-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._camera is not None:
            self._camera.stop()
        with self._condition:
            self._condition.notify_all()

    def wait_for_frame(self, last_seen: int, timeout: float = 5.0) -> StreamSnapshot:
        with self._condition:
            if (
                self._frame_id == last_seen
                and self._error is None
                and not self._stop_event.is_set()
            ):
                self._condition.wait(timeout=timeout)
            return StreamSnapshot(
                frame_id=self._frame_id,
                jpeg_bytes=self._jpeg_bytes,
                error=self._error,
            )

    @property
    def memory(self) -> MemoryStats:
        return self._memory

    def _capture_loop(self) -> None:
        camera = self._camera
        if camera is None:
            self._publish(None, error="Camera was not initialized")
            return

        next_frame_due = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                frame_rgb = np.asarray(camera.capture_array(), dtype=np.uint8)
                frame_bgr = np.asarray(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
                det, detect_ms = self.pipeline.detect(frame_bgr)
                track_t0 = time.perf_counter()
                tracks = self._tracker.update(det.boxes, det.kps, max_tracks=self.args.max_faces)
                track_ms = (time.perf_counter() - track_t0) * 1000.0

                live_ids = {track.track_id for track in tracks}
                self._track_matches = {
                    track_id: match
                    for track_id, match in self._track_matches.items()
                    if track_id in live_ids
                }

                overlay_faces: list[OverlayFace] = []
                embed_ms_total = 0.0
                for track in tracks:
                    match = self._track_matches.get(track.track_id)
                    if track.matched and track.kps is not None:
                        emb, emb_ms = self.pipeline.embed_from_kps(frame_bgr, track.kps)
                        embed_ms_total += emb_ms
                        match = self.gallery.match(emb, threshold=self.args.match_threshold)
                        self._track_matches[track.track_id] = match
                    overlay_faces.append(OverlayFace(track=track, match=match))

                annotate_in_place(
                    frame_bgr,
                    TrackingOverlay(
                        faces=overlay_faces,
                        detect_ms=detect_ms,
                        track_ms=track_ms,
                        embed_ms_total=embed_ms_total,
                        gallery_size=self.gallery.count(),
                    ),
                )
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("OpenCV failed to encode a JPEG frame")
                self._frames_processed += 1
                if self._frames_processed % 25 == 0:
                    self._memory = enforce_memory_cap(
                        self.args.ram_cap_mb, f"live camera frame {self._frames_processed}"
                    )
                self._publish(encoded.tobytes())
            except Exception as exc:  # pragma: no cover - runtime hardware path
                self._publish(None, error=str(exc))
                break

            if self._frame_interval <= 0:
                continue

            next_frame_due += self._frame_interval
            sleep_for = next_frame_due - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame_due = time.perf_counter()

    def _publish(self, jpeg_bytes: bytes | None, error: str | None = None) -> None:
        with self._condition:
            self._frame_id += 1
            self._jpeg_bytes = jpeg_bytes
            self._error = error
            self._condition.notify_all()


class LiveCameraHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    streamer: CameraStreamer
    gallery: GalleryStore


class LiveCameraHandler(BaseHTTPRequestHandler):
    server_version = "ValeniaLiveCamera/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._serve_index()
            return
        if self.path == "/stream.mjpg":
            self._serve_mjpeg()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/enroll":
            self._handle_enroll()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    @property
    def streamer(self) -> CameraStreamer:
        return self.server.streamer  # type: ignore[attr-defined]

    @property
    def gallery(self) -> GalleryStore:
        return self.server.gallery  # type: ignore[attr-defined]

    def _serve_index(self) -> None:
        body = HTML_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        last_seen = -1
        while True:
            snapshot = self.streamer.wait_for_frame(last_seen)
            if snapshot.error is not None:
                break
            if snapshot.frame_id == last_seen or snapshot.jpeg_bytes is None:
                continue

            last_seen = snapshot.frame_id
            payload = snapshot.jpeg_bytes
            try:
                self.wfile.write(b"--FRAME\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def _handle_enroll(self) -> None:
        try:
            name, uploads = self._parse_enroll_request()
            result = self.gallery.enroll(name, uploads, self.streamer.pipeline)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Enrollment failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime path
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Enrollment failed",
                str(exc),
            )
            return

        self._serve_message_page(
            HTTPStatus.OK,
            "Enrollment saved",
            _format_enrollment_message(result),
        )

    def _parse_enroll_request(self) -> tuple[str, list[tuple[str, bytes]]]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Expected multipart/form-data upload")

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc
        if content_length <= 0:
            raise ValueError("Empty request body")

        payload = self.rfile.read(content_length)
        envelope = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + payload
        message = BytesParser(policy=email_default_policy).parsebytes(envelope)
        if not message.is_multipart():
            raise ValueError("Expected multipart form fields")

        name = ""
        uploads: list[tuple[str, bytes]] = []
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            field_name = part.get_param("name", header="Content-Disposition")
            if field_name is None:
                continue
            if field_name == "name":
                text_value = part.get_content()
                if isinstance(text_value, str):
                    name = text_value.strip()
                continue
            if field_name == "photos":
                body = part.get_payload(decode=True)
                if not isinstance(body, bytes):
                    continue
                filename = part.get_filename() or "upload"
                uploads.append((filename, body))

        if not name:
            raise ValueError("Name is required")
        if not uploads:
            raise ValueError("At least one photo is required")
        return name, uploads

    def _serve_message_page(self, status: HTTPStatus, title: str, message: str) -> None:
        body = _render_message_page(title, message).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def annotate_in_place(frame_bgr: UInt8Array, overlay: TrackingOverlay) -> None:
    fresh_tracks = 0
    recognized_faces = 0
    for face in overlay.faces:
        track = face.track
        box = track.box
        x1, y1, x2, y2, score = box
        color = (0, 255, 0) if track.matched else (0, 191, 255)
        if track.matched:
            fresh_tracks += 1
        if face.match is not None and face.match.matched:
            color = (64, 224, 208)
            recognized_faces += 1
        cv2.rectangle(
            frame_bgr,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color=color,
            thickness=2,
        )
        label = f"id={track.track_id} {score:.2f}"
        if not track.matched:
            label = f"{label} hold"
        elif face.match is not None and face.match.matched and face.match.name is not None:
            label = f"{label} {face.match.name} {face.match.score:.2f}"
        cv2.putText(
            frame_bgr,
            label,
            (int(x1), max(16, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        if track.kps is None:
            continue
        for point in track.kps:
            cv2.circle(frame_bgr, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

    status = (
        f"tracks={len(overlay.faces)} "
        f"fresh={fresh_tracks} "
        f"known={recognized_faces}/{overlay.gallery_size} "
        f"det={overlay.detect_ms:.1f}ms "
        f"track={overlay.track_ms:.1f}ms "
        f"emb={overlay.embed_ms_total:.1f}ms"
    )
    cv2.putText(
        frame_bgr,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame_bgr,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )


def build_pipeline(args: argparse.Namespace) -> FacePipeline:
    det_model = ROOT / args.det_model
    rec_model = ROOT / args.rec_model
    if not det_model.exists() or not rec_model.exists():
        raise FileNotFoundError("Missing model files. Run: ./scripts/download_models.sh")

    detector = SCRFDDetector(str(det_model), det_thresh=args.det_thresh)
    embedder = ArcFaceEmbedder(str(rec_model))
    return FacePipeline(
        detector=detector,
        embedder=embedder,
        det_size=parse_size(args.det_size),
        max_faces=args.max_faces,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--det-model", default="models/buffalo_sc/det_500m.onnx")
    parser.add_argument("--rec-model", default="models/buffalo_sc/w600k_mbf.onnx")
    parser.add_argument("--det-size", default="320x320")
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--max-faces", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--track-max-missed",
        type=int,
        default=3,
        help="Keep unmatched tracks for this many frames before dropping them",
    )
    parser.add_argument(
        "--track-iou-thresh",
        type=float,
        default=0.3,
        help="Minimum IoU to match a detection to an existing track",
    )
    parser.add_argument(
        "--track-smoothing",
        type=float,
        default=0.65,
        help="Detection weight for smoothing track boxes and landmarks (0..1)",
    )
    parser.add_argument(
        "--gallery-dir",
        default="data/gallery",
        help="Directory for enrolled gallery data, relative to slop/valenia/",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.228,
        help="Cosine similarity threshold for matching an enrolled identity",
    )
    parser.add_argument(
        "--ram-cap-mb",
        type=float,
        default=4096.0,
        help="Abort if current or peak RSS exceeds this limit; <=0 disables the check",
    )
    return parser.parse_args()


def parse_size(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", maxsplit=1)
    return int(width), int(height)


def main() -> int:
    args = parse_args()

    try:
        enforce_memory_cap(args.ram_cap_mb, "live camera startup")
        pipeline = build_pipeline(args)
        gallery = GalleryStore(ROOT / args.gallery_dir)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    except MemoryError as exc:
        print(exc)
        return 4

    streamer = CameraStreamer(args, pipeline, gallery)
    try:
        streamer.start()
    except RuntimeError as exc:
        print(exc)
        return 3
    except MemoryError as exc:
        print(exc)
        return 4

    server: LiveCameraHTTPServer | None = None
    try:
        server = LiveCameraHTTPServer((args.host, args.port), LiveCameraHandler)
        server.streamer = streamer
        server.gallery = gallery
        print(f"Serving MJPEG on http://{args.host}:{args.port}/")
        server.serve_forever(poll_interval=0.5)
    except OSError as exc:
        print(f"Failed to bind {args.host}:{args.port}: {exc}")
        return 4
    except MemoryError as exc:
        print(exc)
        return 4
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        streamer.stop()
    return 0


def _render_message_page(title: str, message: str) -> str:
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{safe_title}</title>
  <style>
    body {{ margin: 0; font-family: sans-serif; background: #101820; color: #f5f5f5;
      min-height: 100vh; display: grid; place-items: center; }}
    main {{ width: min(100%, 720px); padding: 1rem; box-sizing: border-box; }}
    article {{ border: 1px solid #2b3a42; border-radius: 0.5rem; padding: 1rem;
      background: rgba(255, 255, 255, 0.03); }}
    a {{ color: #b9fbc0; }}
    pre {{ white-space: pre-wrap; font-family: monospace; }}
  </style>
</head>
<body>
  <main>
    <article>
      <h1>{safe_title}</h1>
      <pre>{safe_message}</pre>
      <p><a href=\"/\">Back to live stream</a></p>
    </article>
  </main>
</body>
</html>
"""


def _format_enrollment_message(result: EnrollmentResult) -> str:
    lines = [
        f"Saved identity: {result.name} ({result.slug})",
        f"Accepted photos: {result.sample_count}",
    ]
    if result.accepted_files:
        lines.append("Accepted:")
        lines.extend(f"- {name}" for name in result.accepted_files)
    if result.rejected_files:
        lines.append("Rejected:")
        lines.extend(f"- {name}" for name in result.rejected_files)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
