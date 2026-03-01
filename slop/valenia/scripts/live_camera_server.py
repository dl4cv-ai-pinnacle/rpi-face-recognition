#!/usr/bin/env python3
"""Serve a live Pi camera MJPEG stream with face detections over HTTP."""

from __future__ import annotations

import argparse
import html
import json
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

from contracts import PipelineLike
from gallery import EnrollmentResult, GalleryStore
from live_runtime import LiveRuntime, LiveRuntimeConfig, annotate_in_place
from pipeline_factory import PipelineSpec, build_face_pipeline, parse_size, resolve_project_path
from runtime_utils import MemoryStats, enforce_memory_cap, get_memory_stats

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Valenia Live Camera</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1217;
      --panel: #142028;
      --panel-2: #1c2c36;
      --border: #2f4654;
      --text: #ecf4f3;
      --muted: #97adb7;
      --accent: #83d483;
      --accent-2: #54c6eb;
      --warn: #f5b85f;
      --danger: #ef6f6c;
    }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top right, rgba(84, 198, 235, 0.14), transparent 32%),
        radial-gradient(circle at top left, rgba(131, 212, 131, 0.10), transparent 28%),
        var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    main {
      width: min(100%, 1440px);
      padding: 1rem 1rem 1.25rem;
      box-sizing: border-box;
      margin: 0 auto;
    }
    .header {
      margin-bottom: 1rem;
      padding: 1rem 1.1rem;
      border: 1px solid var(--border);
      border-radius: 0.9rem;
      background: linear-gradient(145deg, rgba(28, 44, 54, 0.95), rgba(20, 32, 40, 0.92));
    }
    h1 {
      margin: 0 0 0.35rem;
      font-size: clamp(1.45rem, 2.2vw, 2rem);
    }
    .header p,
    .subtle {
      margin: 0 0 1rem;
      color: var(--muted);
      line-height: 1.45;
    }
    .subtle:last-child {
      margin-bottom: 0;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(320px, 0.95fr);
      gap: 1rem;
      align-items: start;
    }
    .stack {
      display: grid;
      gap: 1rem;
    }
    .panel {
      padding: 1rem;
      border: 1px solid var(--border);
      border-radius: 0.9rem;
      background: linear-gradient(160deg, rgba(20, 32, 40, 0.96), rgba(15, 24, 30, 0.98));
      box-shadow: 0 16px 36px rgba(0, 0, 0, 0.18);
    }
    .panel h2,
    .panel h3 {
      margin: 0 0 0.8rem;
      font-size: 1rem;
      letter-spacing: 0.01em;
    }
    .camera-shell {
      overflow: hidden;
    }
    .camera-head {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
      margin-bottom: 0.75rem;
    }
    .endpoint-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.45rem;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.28rem 0.55rem;
      border-radius: 999px;
      border: 1px solid rgba(131, 212, 131, 0.25);
      background: rgba(131, 212, 131, 0.08);
      color: #cfead0;
      font-size: 0.8rem;
      font-weight: 600;
    }
    .camera-frame {
      padding: 0.5rem;
      border-radius: 0.9rem;
      border: 1px solid rgba(84, 198, 235, 0.18);
      background: radial-gradient(circle at top, rgba(84, 198, 235, 0.06), rgba(0, 0, 0, 0));
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
      border-radius: 0.55rem;
      border: 1px solid var(--border);
      background: #12212a;
      color: var(--text);
    }
    button {
      width: fit-content;
      padding: 0.7rem 1rem;
      border: 0;
      border-radius: 0.55rem;
      background: linear-gradient(135deg, var(--accent), #b7ec87);
      color: #0b1710;
      font-weight: 700;
      cursor: pointer;
    }
    img {
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid var(--border);
      border-radius: 0.65rem;
      background: #000;
    }
    code {
      color: #d8f7ad;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.65rem;
      margin-bottom: 0.8rem;
    }
    .metric-card {
      padding: 0.75rem;
      border-radius: 0.75rem;
      border: 1px solid rgba(47, 70, 84, 0.85);
      background: linear-gradient(160deg, rgba(28, 44, 54, 0.75), rgba(13, 23, 29, 0.85));
    }
    .metric-label {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .metric-value {
      margin-top: 0.25rem;
      font-size: 1.35rem;
      font-weight: 700;
      line-height: 1.1;
    }
    .metric-detail {
      margin-top: 0.25rem;
      color: var(--muted);
      font-size: 0.82rem;
      min-height: 1.2em;
    }
    .meter {
      margin-top: 0.55rem;
      height: 0.35rem;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      overflow: hidden;
    }
    .meter > span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent-2), var(--accent));
    }
    .list-grid {
      display: grid;
      gap: 0.55rem;
    }
    .list-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0.75rem;
      align-items: center;
      padding: 0.5rem 0;
      border-bottom: 1px solid rgba(47, 70, 84, 0.45);
    }
    .list-row:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }
    .list-key {
      color: var(--muted);
      font-size: 0.9rem;
    }
    .list-value {
      text-align: right;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }
    .status-line {
      margin-top: 0.85rem;
      padding: 0.7rem 0.8rem;
      border-radius: 0.7rem;
      border: 1px solid rgba(84, 198, 235, 0.18);
      background: rgba(84, 198, 235, 0.06);
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.45;
    }
    .status-line strong {
      color: var(--text);
    }
    .error-banner {
      display: none;
      margin-bottom: 0.8rem;
      padding: 0.75rem 0.85rem;
      border-radius: 0.7rem;
      border: 1px solid rgba(239, 111, 108, 0.3);
      background: rgba(239, 111, 108, 0.08);
      color: #ffd4d2;
      font-size: 0.9rem;
    }
    .error-banner.visible {
      display: block;
    }
    @media (max-width: 980px) {
      .layout {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 620px) {
      .metrics-grid {
        grid-template-columns: 1fr;
      }
      .camera-head {
        flex-direction: column;
        align-items: stretch;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="header">
      <h1>Valenia Live Camera</h1>
      <p>
        Live face tracking, recognition, and telemetry. The stream is on the left;
        runtime metrics and enrollment stay pinned on the right.
      </p>
      <p class="subtle">Raw endpoints: <code>/stream.mjpg</code> and <code>/metrics.json</code></p>
    </section>
    <div class="layout">
      <section class="panel camera-shell">
        <div class="camera-head">
          <div>
            <h2>Camera Feed</h2>
            <p class="subtle">Track IDs are session-local tracker IDs. They are not person IDs.</p>
            <div class="endpoint-row">
              <span class="chip">MJPEG: /stream.mjpg</span>
              <span class="chip">Metrics: /metrics.json</span>
            </div>
          </div>
        </div>
        <div class="camera-frame">
          <img src="/stream.mjpg" alt="Live camera stream">
        </div>
      </section>
      <aside class="stack">
        <section class="panel">
          <h2>Live Metrics</h2>
          <div id="error-banner" class="error-banner"></div>
          <div id="hero-metrics" class="metrics-grid"></div>
          <h3>System</h3>
          <div id="system-metrics" class="list-grid"></div>
          <h3>Pipeline</h3>
          <div id="pipeline-metrics" class="list-grid"></div>
          <div id="runtime-status" class="status-line">Loading metrics...</div>
        </section>
        <section class="panel">
          <h2>Enroll Identity</h2>
          <p class="subtle">
            Upload one or more clear face photos to create or update a gallery identity.
          </p>
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
      </aside>
    </div>
  </main>
  <script>
    const heroMetrics = document.getElementById('hero-metrics');
    const systemMetrics = document.getElementById('system-metrics');
    const pipelineMetrics = document.getElementById('pipeline-metrics');
    const runtimeStatus = document.getElementById('runtime-status');
    const errorBanner = document.getElementById('error-banner');

    function fmt(value, digits = 1, suffix = '') {
      if (value === null || value === undefined) {
        return 'n/a';
      }
      const number = Number(value);
      if (!Number.isFinite(number)) {
        return 'n/a';
      }
      return number.toFixed(digits) + suffix;
    }

    function pct(value) {
      if (value === null || value === undefined) {
        return 'n/a';
      }
      return fmt(value, 1, '%');
    }

    function meter(value, max = 100) {
      if (value === null || value === undefined) {
        return '';
      }
      const clamped = Math.max(0, Math.min(max, Number(value)));
      const width = max > 0 ? (clamped / max) * 100 : 0;
      return '<div class="meter"><span style="width:' + width.toFixed(1) + '%"></span></div>';
    }

    function renderCards(metrics) {
      const cards = [
        {
          label: 'Current FPS',
          value: fmt(metrics.current_fps, 1),
          detail: 'Avg ' + fmt(metrics.avg_fps, 1),
        },
        {
          label: 'CPU Load',
          value: pct(metrics.cpu_usage_pct),
          detail: '1m load ' + fmt(metrics.loadavg_1m, 2),
          meterValue: metrics.cpu_usage_pct,
        },
        {
          label: 'CPU Temp',
          value: fmt(metrics.cpu_temp_c, 1, ' C'),
          detail: 'Peak RSS ' + fmt(metrics.peak_rss_mb, 1, ' MiB'),
          meterValue: metrics.cpu_temp_c,
          meterMax: 100,
        },
        {
          label: 'Recognized',
          value: String(metrics.last_recognized_faces ?? 0),
          detail: 'Gallery size ' + String(metrics.gallery_size ?? 0),
        },
      ];

      heroMetrics.innerHTML = cards.map((card) => {
        const detail = card.detail ?? '';
        const meterMarkup = Object.prototype.hasOwnProperty.call(card, 'meterValue')
          ? meter(card.meterValue, card.meterMax ?? 100)
          : '';
        return (
          '<div class="metric-card">' +
            '<div class="metric-label">' + card.label + '</div>' +
            '<div class="metric-value">' + card.value + '</div>' +
            '<div class="metric-detail">' + detail + '</div>' +
            meterMarkup +
          '</div>'
        );
      }).join('');
    }

    function renderRows(container, rows) {
      container.innerHTML = rows.map((row) => (
        '<div class="list-row">' +
          '<div class="list-key">' + row[0] + '</div>' +
          '<div class="list-value">' + row[1] + '</div>' +
        '</div>'
      )).join('');
    }

    function renderMetrics(metrics) {
      renderCards(metrics);

      renderRows(systemMetrics, [
        ['Accelerator', metrics.accelerator_mode || 'cpu-only'],
        [
          'GPU usage',
          metrics.gpu_usage_pct === null
            ? 'not used / unavailable'
            : pct(metrics.gpu_usage_pct)
        ],
        ['Current RSS', fmt(metrics.current_rss_mb, 1, ' MiB')],
        ['Peak RSS', fmt(metrics.peak_rss_mb, 1, ' MiB')],
        ['Load avg (1m / 5m / 15m)', [
          fmt(metrics.loadavg_1m, 2),
          fmt(metrics.loadavg_5m, 2),
          fmt(metrics.loadavg_15m, 2)
        ].join(' / ')],
      ]);

      renderRows(pipelineMetrics, [
        ['Detector cadence', 'every ' + String(metrics.det_every ?? 1) + ' frame(s)'],
        ['Last loop', fmt(metrics.last_loop_ms, 1, ' ms')],
        ['Last detect / track / embed', [
          fmt(metrics.last_detect_ms, 1, ' ms'),
          fmt(metrics.last_track_ms, 1, ' ms'),
          fmt(metrics.last_embed_ms, 1, ' ms')
        ].join(' / ')],
        ['Last faces / fresh tracks', [
          String(metrics.last_faces ?? 0),
          String(metrics.last_fresh_tracks ?? 0)
        ].join(' / ')],
        ['Refreshes / reuses', [
          String(metrics.last_refreshes ?? 0),
          String(metrics.last_reuses ?? 0)
        ].join(' / ')],
        ['Averages (loop / detect / track / embed)', [
          fmt(metrics.avg_loop_ms, 1, ' ms'),
          fmt(metrics.avg_detect_ms, 1, ' ms'),
          fmt(metrics.avg_track_ms, 1, ' ms'),
          fmt(metrics.avg_embed_ms, 1, ' ms')
        ].join(' / ')],
      ]);

      runtimeStatus.innerHTML =
        '<strong>Frames processed:</strong> ' + String(metrics.frames_processed ?? 0) +
        ' &nbsp;|&nbsp; <strong>Uptime:</strong> ' + fmt(metrics.uptime_seconds, 1, ' s') +
        ' &nbsp;|&nbsp; <strong>Embed refresh:</strong> ' +
        (metrics.embed_refresh_enabled ? 'enabled' : 'always recompute');

      if (metrics.last_error) {
        errorBanner.textContent = 'Latest runtime error: ' + metrics.last_error;
        errorBanner.classList.add('visible');
      } else {
        errorBanner.textContent = '';
        errorBanner.classList.remove('visible');
      }
    }

    async function refreshMetrics() {
      try {
        const response = await fetch('/metrics.json', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const metrics = await response.json();
        renderMetrics(metrics);
      } catch (error) {
        runtimeStatus.textContent = 'Metrics unavailable: ' + error;
        errorBanner.textContent = '';
        errorBanner.classList.remove('visible');
      }
    }

    refreshMetrics();
    window.setInterval(refreshMetrics, 1000);
  </script>
</body>
</html>
"""


@dataclass
class StreamSnapshot:
    frame_id: int
    jpeg_bytes: bytes | None
    error: str | None


class CameraStreamer:
    """Capture frames in one background loop and keep only the latest JPEG."""

    def __init__(self, args: argparse.Namespace, runtime: LiveRuntime) -> None:
        self.args = args
        self._runtime = runtime
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

    @property
    def runtime(self) -> LiveRuntime:
        return self._runtime

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

    def _capture_loop(self) -> None:
        camera = self._camera
        if camera is None:
            self._publish(None, error="Camera was not initialized")
            return

        next_frame_due = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                loop_t0 = time.perf_counter()
                frame_rgb = np.asarray(camera.capture_array(), dtype=np.uint8)
                frame_bgr = np.asarray(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
                frame_result = self._runtime.process_frame(frame_bgr)
                annotate_in_place(frame_bgr, frame_result.overlay)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("OpenCV failed to encode a JPEG frame")

                self._frames_processed += 1
                frame_memory = get_memory_stats()
                if self._frames_processed % 25 == 0:
                    frame_memory = enforce_memory_cap(
                        self.args.ram_cap_mb, f"live camera frame {self._frames_processed}"
                    )
                self._memory = frame_memory
                loop_ms = (time.perf_counter() - loop_t0) * 1000.0
                self._runtime.record_frame_metrics(
                    frame_result=frame_result,
                    loop_ms=loop_ms,
                    memory=frame_memory,
                )
                self._publish(encoded.tobytes())
            except Exception as exc:  # pragma: no cover - runtime hardware path
                self._memory = get_memory_stats()
                self._runtime.record_error(str(exc), memory=self._memory)
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


class LiveCameraHandler(BaseHTTPRequestHandler):
    server_version = "ValeniaLiveCamera/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._serve_index()
            return
        if self.path == "/stream.mjpg":
            self._serve_mjpeg()
            return
        if self.path == "/metrics.json":
            self._serve_metrics_json()
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
    def runtime(self) -> LiveRuntime:
        return self.streamer.runtime

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

    def _serve_metrics_json(self) -> None:
        body = json.dumps(self.runtime.metrics_snapshot, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_enroll(self) -> None:
        try:
            name, uploads = self._parse_enroll_request()
            result = self.runtime.enroll(name, uploads)
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


def build_pipeline(args: argparse.Namespace) -> PipelineLike:
    det_model = resolve_project_path(ROOT, args.det_model)
    rec_model = resolve_project_path(ROOT, args.rec_model)
    if not det_model.exists() or not rec_model.exists():
        raise FileNotFoundError("Missing model files. Run: ./scripts/download_models.sh")

    return build_face_pipeline(
        PipelineSpec(
            det_model=det_model,
            rec_model=rec_model,
            det_size=parse_size(args.det_size),
            det_thresh=args.det_thresh,
            max_faces=args.max_faces,
        )
    )


def build_runtime(args: argparse.Namespace, pipeline: PipelineLike) -> LiveRuntime:
    gallery = GalleryStore(resolve_project_path(ROOT, args.gallery_dir))
    metrics_json_path = resolve_project_path(ROOT, args.metrics_json) if args.metrics_json else None
    config = LiveRuntimeConfig(
        max_faces=args.max_faces,
        det_every=args.det_every,
        track_iou_thresh=args.track_iou_thresh,
        track_max_missed=args.track_max_missed,
        track_smoothing=args.track_smoothing,
        match_threshold=args.match_threshold,
        embed_refresh_frames=args.embed_refresh_frames,
        embed_refresh_iou=args.embed_refresh_iou,
        disable_embed_refresh=args.disable_embed_refresh,
        metrics_json_path=metrics_json_path,
        metrics_write_every=args.metrics_write_every,
    )
    return LiveRuntime(pipeline=pipeline, gallery=gallery, config=config)


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
        "--det-every",
        type=int,
        default=1,
        help="Run detector every N frames; values <1 are treated as 1",
    )
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
        "--embed-refresh-frames",
        type=int,
        default=5,
        help="Refresh a track's embedding after this many frames; <=0 disables periodic refresh",
    )
    parser.add_argument(
        "--embed-refresh-iou",
        type=float,
        default=0.85,
        help="Refresh a track's embedding when IoU drops below this value",
    )
    parser.add_argument(
        "--disable-embed-refresh",
        action="store_true",
        help=("Disable selective embedding refresh and recompute embeddings on every fresh track"),
    )
    parser.add_argument(
        "--metrics-json",
        default="data/metrics/live_camera_metrics.json",
        help=(
            "Output JSON path for rolling live metrics, relative to slop/valenia/; "
            "empty disables writes"
        ),
    )
    parser.add_argument(
        "--metrics-write-every",
        type=int,
        default=10,
        help="Write the rolling metrics JSON every N frames; values <1 are treated as 1",
    )
    parser.add_argument(
        "--ram-cap-mb",
        type=float,
        default=4096.0,
        help="Abort if current or peak RSS exceeds this limit; <=0 disables the check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        enforce_memory_cap(args.ram_cap_mb, "live camera startup")
        pipeline = build_pipeline(args)
        runtime = build_runtime(args, pipeline)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    except MemoryError as exc:
        print(exc)
        return 4

    streamer = CameraStreamer(args, runtime)
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
