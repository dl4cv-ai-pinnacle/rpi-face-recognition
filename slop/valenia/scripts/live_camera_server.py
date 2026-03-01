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
from urllib.parse import parse_qs, quote, urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from contracts import PipelineLike
from gallery import EnrollmentResult, GalleryStore, IdentityRecord, UnknownRecord
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
      --bg: #000;
      --panel: #111;
      --border: #333;
      --text: #eee;
      --muted: #999;
      --danger: #e55;
    }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    main {
      width: min(100%, 1440px);
      padding: 1rem;
      box-sizing: border-box;
      margin: 0 auto;
    }
    a {
      color: var(--text);
      text-decoration: underline;
      text-underline-offset: 0.2em;
    }
    a:hover {
      color: #fff;
    }
    nav {
      display: flex;
      align-items: baseline;
      gap: 1.5rem;
      padding: 0 0 1rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }
    nav strong {
      font-size: 1.1rem;
      margin-right: auto;
    }
    nav a {
      font-size: 0.9rem;
      color: var(--muted);
    }
    nav a:hover {
      color: var(--text);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(300px, 0.95fr);
      gap: 1rem;
      align-items: start;
    }
    .stack {
      display: grid;
      gap: 1rem;
    }
    h2, h3 {
      margin: 0 0 0.75rem;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      font-weight: 600;
    }
    .stream-wrap {
      position: relative;
      aspect-ratio: 4 / 3;
      background: #111;
      border-radius: 4px;
      overflow: hidden;
    }
    .stream-wrap img {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      border-radius: 4px;
      background: #000;
      opacity: 0;
      transition: opacity 0.3s;
    }
    .stream-wrap img.loaded {
      opacity: 1;
    }
    .stream-loading {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 0.9rem;
      gap: 0.5rem;
    }
    .stream-loading.hidden {
      display: none;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid var(--border);
      border-top-color: var(--muted);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.5rem;
      margin-bottom: 1rem;
    }
    .metric-card {
      padding: 0.75rem;
      border: 1px solid var(--border);
      border-radius: 4px;
    }
    .metric-label {
      color: var(--muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .metric-label[title],
    .list-key[title] {
      cursor: help;
      text-decoration: underline dotted var(--border);
      text-underline-offset: 0.18rem;
    }
    .metric-value {
      margin-top: 0.2rem;
      font-size: 1.5rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      line-height: 1.1;
    }
    .metric-detail {
      margin-top: 0.2rem;
      color: var(--muted);
      font-size: 0.8rem;
      min-height: 1.1em;
    }
    .meter {
      margin-top: 0.45rem;
      height: 3px;
      border-radius: 2px;
      background: var(--border);
      overflow: hidden;
    }
    .meter > span {
      display: block;
      height: 100%;
      border-radius: inherit;
      background: var(--text);
    }
    .list-grid {
      display: grid;
      gap: 0;
    }
    .list-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0.75rem;
      align-items: center;
      padding: 0.45rem 0;
      border-bottom: 1px solid #1a1a1a;
    }
    .list-row:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }
    .list-key {
      color: var(--muted);
      font-size: 0.85rem;
    }
    .list-value {
      text-align: right;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
      font-size: 0.9rem;
    }
    .status-line {
      margin-top: 0.75rem;
      padding-top: 0.75rem;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.85rem;
      line-height: 1.45;
    }
    .status-line strong {
      color: var(--text);
    }
    .section-sep {
      margin: 1rem 0 0.75rem;
      border-top: 1px solid var(--border);
      padding-top: 0.75rem;
    }
    .error-banner {
      display: none;
      margin-bottom: 0.75rem;
      padding: 0.6rem 0.75rem;
      border-radius: 4px;
      border: 1px solid rgba(238, 85, 85, 0.3);
      background: rgba(238, 85, 85, 0.08);
      color: #fcc;
      font-size: 0.85rem;
    }
    .error-banner.visible {
      display: block;
    }
    .disconnect-banner {
      display: none;
      padding: 0.6rem 0.75rem;
      margin-bottom: 1rem;
      border-radius: 4px;
      border: 1px solid #555;
      background: #1a1a1a;
      color: var(--muted);
      font-size: 0.85rem;
      text-align: center;
    }
    .disconnect-banner.visible {
      display: block;
    }
    .stream-wrap.disconnected {
      opacity: 0.4;
    }
    .toggle-btn {
      position: absolute;
      top: 0.5rem;
      right: 0.5rem;
      z-index: 10;
      background: rgba(0, 0, 0, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: #ccc;
      padding: 0.3rem 0.6rem;
      border-radius: 4px;
      font-size: 0.75rem;
      cursor: pointer;
      font-weight: 600;
      opacity: 0;
      transition: opacity 0.2s;
    }
    .stream-wrap:hover .toggle-btn,
    .toggle-btn:focus {
      opacity: 1;
    }
    .toggle-btn:hover {
      color: #fff;
      border-color: rgba(255, 255, 255, 0.4);
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
    }

    /* -- cinema / fullscreen mode -- */
    body.cinema main {
      width: 100%;
      max-width: 100%;
      padding: 0.5rem;
    }
    body.cinema nav {
      margin-bottom: 0.5rem;
      padding-bottom: 0.5rem;
    }
    body.cinema .default-view { display: none; }
    body.cinema .cinema-view { display: grid; }
    .cinema-view {
      display: none;
      grid-template-columns: minmax(180px, 0.7fr) minmax(0, 2fr) minmax(180px, 0.7fr);
      gap: 0.5rem;
      height: calc(100vh - 4.5rem);
    }
    .cinema-view .stream-wrap {
      aspect-ratio: unset;
      height: 100%;
    }
    .cinema-view #cinema-stream {
      min-height: 0;
    }
    .cinema-side {
      overflow-y: auto;
      max-height: calc(100vh - 4.5rem);
      align-self: start;
      font-size: 0.85rem;
    }
    .cinema-side h3 {
      margin: 0.5rem 0 0.4rem;
      font-size: 0.75rem;
    }
    .cinema-side .metric-card {
      padding: 0.5rem;
    }
    .cinema-side .metric-value {
      font-size: 1.1rem;
    }
    .cinema-side .metric-detail {
      font-size: 0.75rem;
    }
    .cinema-side .metrics-grid {
      grid-template-columns: 1fr;
      gap: 0.35rem;
      margin-bottom: 0.5rem;
    }
    .cinema-side .list-row {
      padding: 0.3rem 0;
    }
    .cinema-side .list-key {
      font-size: 0.8rem;
    }
    .cinema-side .list-value {
      font-size: 0.8rem;
    }
    .cinema-side .status-line {
      font-size: 0.8rem;
      margin-top: 0.5rem;
      padding-top: 0.5rem;
    }
    body.cinema .toggle-btn {
      opacity: 1;
    }
    body.cinema .disconnect-banner {
      margin-bottom: 0.5rem;
    }
    @media (max-width: 980px) {
      .cinema-view {
        grid-template-columns: 1fr;
        height: auto;
      }
      .cinema-side {
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <main>
    <nav>
      <strong>Valenia</strong>
      <a href="/gallery">Gallery</a>
      <a href="/stream.mjpg">Stream</a>
      <a href="/metrics.json">Metrics JSON</a>
    </nav>
    <div id="disconnect-banner" class="disconnect-banner">
      Server disconnected. The stream and metrics have stopped updating.
    </div>
    <!-- default two-column view -->
    <div class="layout default-view">
      <section>
        <h2>Live Feed</h2>
        <div id="stream-default" class="stream-wrap">
          <div id="stream-loading" class="stream-loading">
            <span class="spinner"></span> Connecting...
          </div>
          <img id="stream-img" src="/stream.mjpg"
               alt="Live camera stream">
          <button id="cinema-toggle" class="toggle-btn"
                  title="Toggle cinema mode (F)">Cinema</button>
        </div>
      </section>
      <aside class="stack">
        <section>
          <h2>Metrics</h2>
          <div id="error-banner" class="error-banner"></div>
          <div id="hero-metrics" class="metrics-grid"></div>
          <div class="section-sep">
            <h3>System</h3>
          </div>
          <div id="system-metrics" class="list-grid"></div>
          <div class="section-sep">
            <h3>Pipeline</h3>
          </div>
          <div id="pipeline-metrics" class="list-grid"></div>
          <div id="runtime-status"
               class="status-line">Loading metrics...</div>
        </section>
      </aside>
    </div>
    <!-- cinema three-column view -->
    <div class="cinema-view">
      <div id="cinema-left" class="cinema-side"></div>
      <div id="cinema-stream"></div>
      <div id="cinema-right" class="cinema-side"></div>
    </div>
  </main>
  <script>
    const streamImg = document.getElementById('stream-img');
    const streamLoading = document.getElementById('stream-loading');
    const streamWrap = streamImg.parentElement;
    const disconnectBanner = document.getElementById('disconnect-banner');
    const heroMetrics = document.getElementById('hero-metrics');
    const systemMetrics = document.getElementById('system-metrics');
    const pipelineMetrics = document.getElementById('pipeline-metrics');
    const runtimeStatus = document.getElementById('runtime-status');
    const errorBanner = document.getElementById('error-banner');

    let failCount = 0;
    let disconnected = false;
    let streamConnected = false;
    let streamEpoch = 0;

    // -- stream management --------------------------------------------------
    // MJPEG over <img> is a long-lived connection. The browser will NOT
    // reconnect it after the server restarts or the tab is backgrounded.
    // We don't trust load/error events (unreliable for MJPEG across browsers).
    // Instead we drive everything from the 1 Hz metrics poll + visibility API.

    function connectStream() {
      streamEpoch++;
      streamConnected = false;
      streamImg.classList.remove('loaded');
      streamLoading.classList.remove('hidden');
      streamLoading.innerHTML = '<span class="spinner"></span> Connecting...';
      // Cache-bust forces the browser to open a fresh MJPEG connection.
      streamImg.src = '/stream.mjpg?t=' + Date.now();
    }

    function checkStreamAlive() {
      // Once the first MJPEG frame arrives, naturalWidth becomes > 0.
      if (!streamConnected && streamImg.naturalWidth > 0) {
        streamConnected = true;
        streamImg.classList.add('loaded');
        streamLoading.classList.add('hidden');
      }
    }

    function setDisconnected(yes) {
      if (disconnected === yes) { return; }
      disconnected = yes;
      disconnectBanner.classList.toggle('visible', yes);
      streamWrap.classList.toggle('disconnected', yes);
      if (yes) {
        streamLoading.innerHTML = 'Stream stopped';
        streamLoading.classList.remove('hidden');
      }
    }

    // Reconnect whenever the tab becomes visible again.
    document.addEventListener('visibilitychange', function() {
      if (!document.hidden) { connectStream(); }
    });
    // Also handle bfcache restoration (back/forward navigation).
    window.addEventListener('pageshow', function(e) {
      if (e.persisted) { connectStream(); }
    });

    // Initial connection.
    connectStream();

    // -- formatting helpers --------------------------------------------------

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

    // -- metrics rendering ---------------------------------------------------

    function renderCards(metrics) {
      const cards = [
        {
          label: 'Output FPS',
          value: fmt(metrics.current_output_fps, 1),
          detail: 'avg ' + fmt(metrics.avg_output_fps, 1),
          tooltip:
            'Actual delivered stream cadence, including the enforced sleep and recognition work.',
        },
        {
          label: 'Target FPS',
          value: fmt(metrics.target_fps, 1),
          detail: 'configured cap',
          tooltip: 'The configured maximum loop cadence from --fps. Actual output can be lower.',
        },
        {
          label: 'CPU',
          value: pct(metrics.cpu_usage_pct),
          detail: 'load ' + fmt(metrics.loadavg_1m, 2),
          meterValue: metrics.cpu_usage_pct,
          tooltip: 'Whole-system CPU usage estimated from /proc/stat deltas.',
        },
        {
          label: 'Temp',
          value: fmt(metrics.cpu_temp_c, 1, '\u00b0C'),
          detail: 'RSS ' + fmt(metrics.peak_rss_mb, 0, ' MB'),
          meterValue: metrics.cpu_temp_c,
          meterMax: 100,
          tooltip: 'Raspberry Pi SoC temperature from the thermal sensor.',
        },
      ];

      heroMetrics.innerHTML = cards.map((card) => {
        const detail = card.detail ?? '';
        const meterMarkup = Object.prototype.hasOwnProperty.call(card, 'meterValue')
          ? meter(card.meterValue, card.meterMax ?? 100)
          : '';
        const tooltip = card.tooltip ? ' title="' + card.tooltip + '"' : '';
        return (
          '<div class="metric-card">' +
            '<div class="metric-label"' + tooltip + '>' + card.label + '</div>' +
            '<div class="metric-value">' + card.value + '</div>' +
            '<div class="metric-detail">' + detail + '</div>' +
            meterMarkup +
          '</div>'
        );
      }).join('');
    }

    function renderRows(container, rows) {
      container.innerHTML = rows.map((row) => (
        (function() {
          const tooltip = row[2] ? ' title="' + row[2] + '"' : '';
          return (
        '<div class="list-row">' +
          '<div class="list-key"' + tooltip + '>' + row[0] + '</div>' +
          '<div class="list-value">' + row[1] + '</div>' +
        '</div>'
          );
        })()
      )).join('');
    }

    function renderMetrics(metrics) {
      renderCards(metrics);

      renderRows(systemMetrics, [
        [
          'Accelerator',
          metrics.accelerator_mode || 'cpu-only',
          'Which inference provider is currently doing the recognition work.'
        ],
        [
          'GPU usage',
          metrics.gpu_usage_pct === null
            ? 'n/a'
            : pct(metrics.gpu_usage_pct),
          'Kernel-reported GPU busy percentage when available.'
        ],
        [
          'Current RSS',
          fmt(metrics.current_rss_mb, 1, ' MB'),
          'Current process resident memory.'
        ],
        [
          'Peak RSS',
          fmt(metrics.peak_rss_mb, 1, ' MB'),
          'Peak resident memory used by this process.'
        ],
        ['Load avg 1/5/15m', [
          fmt(metrics.loadavg_1m, 2),
          fmt(metrics.loadavg_5m, 2),
          fmt(metrics.loadavg_15m, 2)
        ].join(' / '), 'OS load average over 1, 5, and 15 minutes.'],
      ]);

      renderRows(pipelineMetrics, [
        [
          'Detect cadence',
          'every ' + String(metrics.det_every ?? 1) + ' frame(s)',
          'How often the face detector runs.'
        ],
        [
          'Processing FPS',
          fmt(metrics.current_processing_fps, 1),
          'Loop body speed excluding the intentional sleep.'
        ],
        [
          'Last loop',
          fmt(metrics.last_loop_ms, 1, ' ms'),
          'Processing time for the latest frame.'
        ],
        ['Detect / track / embed', [
          fmt(metrics.last_detect_ms, 1),
          fmt(metrics.last_track_ms, 1),
          fmt(metrics.last_embed_ms, 1)
        ].join(' / ') + ' ms', 'Per-stage timings for the latest frame.'],
        ['Faces / fresh tracks', [
          String(metrics.last_faces ?? 0),
          String(metrics.last_fresh_tracks ?? 0)
        ].join(' / '),
          'Visible tracked faces and how many were refreshed on the latest frame.'
        ],
        ['Refreshes / reuses', [
          String(metrics.last_refreshes ?? 0),
          String(metrics.last_reuses ?? 0)
        ].join(' / '), 'Embeddings recomputed vs reused from cache.'],
        [
          'Recognized / gallery',
          [
            String(metrics.last_recognized_faces ?? 0),
            String(metrics.gallery_size ?? 0)
          ].join(' / '),
          'Recognized faces versus confirmed gallery identities.'
        ],
        ['Avg loop / det / trk / emb', [
          fmt(metrics.avg_loop_ms, 1),
          fmt(metrics.avg_detect_ms, 1),
          fmt(metrics.avg_track_ms, 1),
          fmt(metrics.avg_embed_ms, 1)
        ].join(' / ') + ' ms', 'Rolling averages since start.'],
      ]);

      runtimeStatus.innerHTML =
        '<strong>Frames:</strong> ' + String(metrics.frames_processed ?? 0) +
        ' &nbsp;|&nbsp; <strong>Uptime:</strong> ' + fmt(metrics.uptime_seconds, 1, ' s') +
        ' &nbsp;|&nbsp; <strong>Embed refresh:</strong> ' +
        (metrics.embed_refresh_enabled ? 'on' : 'off');

      if (metrics.last_error) {
        errorBanner.textContent = 'Error: ' + metrics.last_error;
        errorBanner.classList.add('visible');
      } else {
        errorBanner.textContent = '';
        errorBanner.classList.remove('visible');
      }
    }

    // -- metrics poll (single source of truth for connectivity) ---------------

    async function refreshMetrics() {
      try {
        const response = await fetch('/metrics.json', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error('HTTP ' + response.status);
        }
        const metrics = await response.json();
        const wasDisconnected = disconnected;
        failCount = 0;
        setDisconnected(false);
        // Reconnect stream if we just recovered from a disconnect.
        if (wasDisconnected) { connectStream(); }
        // Check if the MJPEG connection has delivered a frame yet.
        checkStreamAlive();
        renderMetrics(metrics);
      } catch (error) {
        failCount++;
        if (failCount >= 3) {
          setDisconnected(true);
        }
        runtimeStatus.textContent = disconnected
          ? 'Server disconnected'
          : 'Retrying... (' + failCount + ')';
        errorBanner.textContent = '';
        errorBanner.classList.remove('visible');
      }
    }

    refreshMetrics();
    window.setInterval(refreshMetrics, 1000);

    // -- cinema mode ---------------------------------------------------------
    const cinemaToggle = document.getElementById('cinema-toggle');
    const cinemaLeft = document.getElementById('cinema-left');
    const cinemaCenter = document.getElementById('cinema-stream');
    const cinemaRight = document.getElementById('cinema-right');
    const streamDefault = document.getElementById('stream-default');
    let cinemaActive = false;

    function enterCinema() {
      cinemaActive = true;
      document.body.classList.add('cinema');
      cinemaToggle.textContent = 'Exit';
      // Move stream into cinema center.
      cinemaCenter.appendChild(streamWrap);
      // Reparenting kills the MJPEG connection; reconnect.
      connectStream();
      // Build side panels from live metric containers.
      cinemaLeft.innerHTML = '';
      cinemaRight.innerHTML = '';
      // Left side: hero cards + system metrics.
      const leftH = document.createElement('h3');
      leftH.textContent = 'Overview';
      cinemaLeft.appendChild(leftH);
      cinemaLeft.appendChild(heroMetrics);
      const sysH = document.createElement('h3');
      sysH.textContent = 'System';
      cinemaLeft.appendChild(sysH);
      cinemaLeft.appendChild(systemMetrics);
      // Right side: pipeline + status + errors.
      const pipH = document.createElement('h3');
      pipH.textContent = 'Pipeline';
      cinemaRight.appendChild(pipH);
      cinemaRight.appendChild(pipelineMetrics);
      cinemaRight.appendChild(runtimeStatus);
      cinemaRight.appendChild(errorBanner);
    }

    function exitCinema() {
      cinemaActive = false;
      document.body.classList.remove('cinema');
      cinemaToggle.textContent = 'Cinema';
      // Move stream back to default slot.
      const slot = document.querySelector(
        '.default-view section:first-child'
      );
      slot.innerHTML = '';
      const h2f = document.createElement('h2');
      h2f.textContent = 'Live Feed';
      slot.appendChild(h2f);
      slot.appendChild(streamWrap);
      // Reparenting kills the MJPEG connection; reconnect.
      connectStream();
      // Restore metrics sidebar.
      const aside = document.querySelector(
        '.default-view aside section'
      );
      aside.innerHTML = '';
      const h2m = document.createElement('h2');
      h2m.textContent = 'Metrics';
      aside.appendChild(h2m);
      aside.appendChild(errorBanner);
      aside.appendChild(heroMetrics);
      const sep1 = document.createElement('div');
      sep1.className = 'section-sep';
      sep1.innerHTML = '<h3>System</h3>';
      aside.appendChild(sep1);
      aside.appendChild(systemMetrics);
      const sep2 = document.createElement('div');
      sep2.className = 'section-sep';
      sep2.innerHTML = '<h3>Pipeline</h3>';
      aside.appendChild(sep2);
      aside.appendChild(pipelineMetrics);
      aside.appendChild(runtimeStatus);
    }

    function toggleCinema() {
      if (cinemaActive) { exitCinema(); } else { enterCinema(); }
    }

    cinemaToggle.addEventListener('click', toggleCinema);
    document.addEventListener('keydown', function(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
      }
      if (e.key === 'f' || e.key === 'F') { toggleCinema(); }
    });
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

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_index()
            return
        if parsed.path == "/gallery":
            self._serve_gallery()
            return
        if parsed.path == "/gallery/identity":
            self._serve_identity_detail(parsed.query)
            return
        if parsed.path == "/gallery/image":
            self._serve_gallery_image(parsed.query)
            return
        if parsed.path == "/stream.mjpg":
            self._serve_mjpeg()
            return
        if parsed.path == "/metrics.json":
            self._serve_metrics_json()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/enroll":
            self._handle_enroll()
            return
        if parsed.path == "/gallery/promote":
            self._handle_gallery_promote()
            return
        if parsed.path == "/gallery/rename":
            self._handle_gallery_rename()
            return
        if parsed.path == "/gallery/merge-unknowns":
            self._handle_gallery_merge_unknowns()
            return
        if parsed.path == "/gallery/delete-unknown":
            self._handle_gallery_delete_unknown()
            return
        if parsed.path == "/gallery/delete-identity":
            self._handle_gallery_delete_identity()
            return
        if parsed.path == "/gallery/delete-sample":
            self._handle_gallery_delete_sample()
            return
        if parsed.path == "/gallery/upload-samples":
            self._handle_gallery_upload_samples()
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

    def _serve_gallery(self) -> None:
        body = _render_gallery_page(
            identities=self.runtime.list_identities(),
            unknowns=self.runtime.list_unknowns(),
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_gallery_image(self, query_string: str) -> None:
        params = parse_qs(query_string, keep_blank_values=False)
        kind = params.get("kind", [""])[0]
        slug = params.get("slug", [""])[0]
        filename = params.get("file", [""])[0]
        try:
            body, content_type = self.runtime.read_gallery_image(kind, slug, filename)
        except ValueError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
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

    def _handle_gallery_promote(self) -> None:
        try:
            fields = self._parse_form_fields()
            unknown_slug = fields.get("unknown_slug", "")
            name = fields.get("name", "")
            self.runtime.promote_unknown(unknown_slug, name)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Promotion failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime path
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Promotion failed",
                str(exc),
            )
            return
        self._redirect("/gallery")

    def _handle_gallery_merge_unknowns(self) -> None:
        try:
            fields = self._parse_form_fields()
            target_slug = fields.get("target_slug", "")
            source_slug = fields.get("source_slug", "")
            self.runtime.merge_unknowns(target_slug, source_slug)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Merge failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime path
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Merge failed",
                str(exc),
            )
            return
        self._redirect("/gallery")

    def _handle_gallery_rename(self) -> None:
        try:
            fields = self._parse_form_fields()
            slug = fields.get("slug", "")
            name = fields.get("name", "")
            self.runtime.rename_identity(slug, name)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Rename failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime path
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Rename failed",
                str(exc),
            )
            return
        self._redirect("/gallery")

    def _handle_gallery_delete_unknown(self) -> None:
        try:
            fields = self._parse_form_fields()
            unknown_slug = fields.get("unknown_slug", "")
            self.runtime.delete_unknown(unknown_slug)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Delete failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime path
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Delete failed",
                str(exc),
            )
            return
        self._redirect("/gallery")

    def _handle_gallery_delete_identity(self) -> None:
        try:
            fields = self._parse_form_fields()
            slug = fields.get("slug", "")
            self.runtime.delete_identity(slug)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Delete failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime path
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Delete failed",
                str(exc),
            )
            return
        self._redirect("/gallery")

    def _handle_gallery_delete_sample(self) -> None:
        try:
            fields = self._parse_form_fields()
            slug = fields.get("slug", "")
            filename = fields.get("filename", "")
            self.runtime.delete_identity_sample(slug, filename)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Delete failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime path
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Delete failed",
                str(exc),
            )
            return
        self._redirect(f"/gallery/identity?slug={quote(slug)}")

    def _handle_gallery_upload_samples(self) -> None:
        try:
            slug, uploads = self._parse_upload_samples_request()
            result = self.runtime.upload_to_identity(slug, uploads)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Upload failed", str(exc))
            return
        except Exception as exc:  # pragma: no cover - runtime path
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "Upload failed",
                str(exc),
            )
            return
        self._redirect(f"/gallery/identity?slug={quote(result.slug)}")

    def _parse_upload_samples_request(self) -> tuple[str, list[tuple[str, bytes]]]:
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

        slug = ""
        uploads: list[tuple[str, bytes]] = []
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            field_name = part.get_param("name", header="Content-Disposition")
            if field_name is None:
                continue
            if field_name == "slug":
                text_value = part.get_content()
                if isinstance(text_value, str):
                    slug = text_value.strip()
                continue
            if field_name == "photos":
                body = part.get_payload(decode=True)
                if not isinstance(body, bytes):
                    continue
                filename = part.get_filename() or "upload"
                uploads.append((filename, body))

        if not slug:
            raise ValueError("Identity slug is required")
        if not uploads:
            raise ValueError("At least one photo is required")
        return slug, uploads

    def _serve_identity_detail(self, query_string: str) -> None:
        params = parse_qs(query_string, keep_blank_values=False)
        slug = params.get("slug", [""])[0]
        identities = self.runtime.list_identities()
        record = next((r for r in identities if r.slug == slug), None)
        if record is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Identity not found")
            return
        images = self.runtime.list_identity_images(slug)
        body = _render_identity_detail_page(record, images).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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

    def _parse_form_fields(self) -> dict[str, str]:
        content_type = self.headers.get("Content-Type", "")
        if "application/x-www-form-urlencoded" not in content_type:
            raise ValueError("Expected application/x-www-form-urlencoded form data")
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc
        payload = self.rfile.read(max(0, content_length))
        decoded = payload.decode("utf-8", errors="replace")
        raw_fields = parse_qs(decoded, keep_blank_values=True)
        return {key: values[0].strip() for key, values in raw_fields.items() if values}

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

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
    enrich_margin = 999.0 if args.disable_enrich else args.enrich_margin
    config = LiveRuntimeConfig(
        max_faces=args.max_faces,
        target_fps=args.fps,
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
        enrich_margin=enrich_margin,
        enrich_min_quality=args.enrich_min_quality,
        enrich_cooldown_seconds=args.enrich_cooldown,
        enrich_max_samples=args.enrich_max_samples,
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
    parser.add_argument(
        "--enrich-margin",
        type=float,
        default=0.10,
        help="Match score must exceed match_threshold + margin for enrichment",
    )
    parser.add_argument(
        "--enrich-min-quality",
        type=float,
        default=0.40,
        help="Minimum face quality score for enrichment",
    )
    parser.add_argument(
        "--enrich-cooldown",
        type=float,
        default=30.0,
        help="Per-identity cooldown in seconds between enrichments",
    )
    parser.add_argument(
        "--enrich-max-samples",
        type=int,
        default=48,
        help="Maximum number of embedding samples per identity",
    )
    parser.add_argument(
        "--disable-enrich",
        action="store_true",
        help="Disable auto-enrichment of identities during live recognition",
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
    body {{ margin: 0; font-family: \"IBM Plex Sans\", \"Segoe UI\", system-ui, sans-serif;
      background: #000; color: #eee; min-height: 100vh; display: grid; place-items: center; }}
    main {{ width: min(100%, 720px); padding: 1rem; box-sizing: border-box; }}
    h1 {{ font-size: 1.2rem; }}
    a {{ color: #eee; text-decoration: underline; text-underline-offset: 0.2em; }}
    a:hover {{ color: #fff; }}
    pre {{ white-space: pre-wrap; font-family: monospace; color: #999; }}
    .links {{ display: flex; gap: 1.5rem; margin-top: 1rem; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_title}</h1>
    <pre>{safe_message}</pre>
    <div class=\"links\">
      <a href=\"/\">Live feed</a>
      <a href=\"/gallery\">Gallery</a>
    </div>
  </main>
</body>
</html>
"""


def _render_gallery_page(
    *,
    identities: list[IdentityRecord],
    unknowns: list[UnknownRecord],
) -> str:
    identity_cards = "\n".join(_render_identity_card(record) for record in identities)
    unknown_cards = "\n".join(
        _render_unknown_card(record, unknowns=unknowns, identities=identities)
        for record in unknowns
    )
    if not identity_cards:
        identity_cards = '<p class="empty">No confirmed identities yet.</p>'
    if not unknown_cards:
        unknown_cards = '<p class="empty">No auto-captured unknowns yet.</p>'

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Valenia Gallery</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #000;
      --border: #333;
      --text: #eee;
      --muted: #999;
      --danger: #e55;
    }}
    body {{
      margin: 0;
      font-family: \"IBM Plex Sans\", \"Segoe UI\", system-ui, sans-serif;
      color: var(--text);
      background: var(--bg);
      min-height: 100vh;
    }}
    main {{
      width: min(100%, 1380px);
      margin: 0 auto;
      padding: 1rem;
      box-sizing: border-box;
    }}
    a {{
      color: var(--text);
      text-decoration: underline;
      text-underline-offset: 0.2em;
    }}
    a:hover {{
      color: #fff;
    }}
    nav {{
      display: flex;
      align-items: baseline;
      gap: 1.5rem;
      padding: 0 0 1rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }}
    nav strong {{
      font-size: 1.1rem;
      margin-right: auto;
    }}
    nav a {{
      font-size: 0.9rem;
      color: var(--muted);
    }}
    nav a:hover {{
      color: var(--text);
    }}
    h2, h3 {{
      margin: 0 0 0.75rem;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      font-weight: 600;
    }}
    .columns {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1rem;
      align-items: start;
    }}
    .card-list {{
      display: grid;
      gap: 0.75rem;
    }}
    .card {{
      display: grid;
      grid-template-columns: 100px minmax(0, 1fr);
      gap: 0.75rem;
      padding: 0.75rem 0;
      border-bottom: 1px solid #1a1a1a;
    }}
    .card:last-child {{
      border-bottom: 0;
    }}
    .thumb {{
      width: 100px;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 4px;
      background: #000;
    }}
    .thumb.empty {{
      display: grid;
      place-items: center;
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.8rem;
    }}
    .card-name {{
      margin: 0 0 0.2rem;
      font-size: 1rem;
      color: var(--text);
      text-transform: none;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.85rem;
      margin: 0.1rem 0 0.5rem;
    }}
    form {{
      display: grid;
      gap: 0.5rem;
      margin-top: 0.4rem;
    }}
    input, button {{
      font: inherit;
    }}
    input[type=\"text\"],
    input[type=\"file\"] {{
      padding: 0.5rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: #111;
      color: var(--text);
    }}
    .button-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    button {{
      width: fit-content;
      padding: 0.55rem 0.85rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: var(--text);
      color: #000;
      font-weight: 700;
      cursor: pointer;
    }}
    button.delete {{
      background: transparent;
      color: #fcc;
      border-color: rgba(238, 85, 85, 0.3);
    }}
    .empty {{
      color: var(--muted);
    }}
    .enroll-section {{
      margin-bottom: 1.5rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }}
    .enroll-section p {{
      color: var(--muted);
      font-size: 0.9rem;
      margin: 0 0 0.75rem;
      line-height: 1.4;
    }}
    .enroll-form {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: end;
    }}
    .enroll-form label {{
      display: grid;
      gap: 0.25rem;
      font-size: 0.85rem;
      color: var(--muted);
    }}
    @media (max-width: 980px) {{
      .columns {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 640px) {{
      .card {{
        grid-template-columns: 1fr;
      }}
      .thumb {{
        width: 100%;
        max-width: 200px;
      }}
      .enroll-form {{
        flex-direction: column;
        align-items: stretch;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <nav>
      <strong>Valenia</strong>
      <a href=\"/\">Live Feed</a>
      <a href=\"/stream.mjpg\">Stream</a>
      <a href=\"/metrics.json\">Metrics JSON</a>
    </nav>
    <section class=\"enroll-section\">
      <h2>Enroll Identity</h2>
      <p>Upload clear face photos to create or update a gallery identity.</p>
      <form class=\"enroll-form\" action=\"/enroll\" method=\"post\"
            enctype=\"multipart/form-data\">
        <label>
          Name
          <input type=\"text\" name=\"name\" required>
        </label>
        <label>
          Photos
          <input type=\"file\" name=\"photos\" accept=\"image/*\" multiple required>
        </label>
        <button type=\"submit\">Enroll</button>
      </form>
    </section>
    <div class=\"columns\">
      <section>
        <h2>Confirmed Identities</h2>
        <div class=\"card-list\">
          {identity_cards}
        </div>
      </section>
      <section>
        <h2>Unknown Review Inbox</h2>
        <div class=\"card-list\">
          {unknown_cards}
        </div>
      </section>
    </div>
  </main>
</body>
</html>
"""


def _render_identity_card(record: IdentityRecord) -> str:
    name = html.escape(record.name)
    slug = html.escape(record.slug)
    detail_url = "/gallery/identity?slug=" + quote(record.slug)
    sample_count = record.sample_count
    preview = _render_preview_image("identity", record.slug, record.preview_filename)
    return f"""
<div class=\"card\">
  {preview}
  <div>
    <h3 class=\"card-name\"><a href=\"{detail_url}\">{name}</a></h3>
    <p class=\"meta\">{slug} &middot; {sample_count} samples</p>
    <form action=\"/gallery/rename\" method=\"post\">
      <input type=\"hidden\" name=\"slug\" value=\"{slug}\">
      <input type=\"text\" name=\"name\" value=\"{name}\" required placeholder=\"New name\">
      <div class=\"button-row\">
        <button type=\"submit\">Rename</button>
      </div>
    </form>
    <form action=\"/gallery/delete-identity\" method=\"post\"
          onsubmit=\"return confirm('Delete identity {name}?')\" style=\"margin-top:0\">
      <input type=\"hidden\" name=\"slug\" value=\"{slug}\">
      <div class=\"button-row\">
        <button class=\"delete\" type=\"submit\">Delete</button>
      </div>
    </form>
  </div>
</div>
"""


def _render_unknown_card(
    record: UnknownRecord,
    *,
    unknowns: list[UnknownRecord],
    identities: list[IdentityRecord],
) -> str:
    slug_raw = record.slug
    slug = html.escape(slug_raw)
    sample_count = record.sample_count
    preview = _render_preview_image("unknown", slug_raw, record.preview_filename)

    merge_options: list[str] = []
    for other in unknowns:
        if other.slug == slug_raw:
            continue
        other_slug = html.escape(other.slug)
        cnt = other.sample_count
        merge_options.append(
            f'<option value="unknown:{other_slug}">{other_slug} ({cnt} captures)</option>'
        )
    for identity in identities:
        id_name = html.escape(identity.name)
        cnt = identity.sample_count
        merge_options.append(
            f'<option value="identity:{id_name}">{id_name} ({cnt} samples)</option>'
        )

    merge_form = ""
    if merge_options:
        options_html = "\n".join(merge_options)
        form_id = f"merge-form-{slug}"
        merge_form = f"""
    <form id=\"{form_id}\" method=\"post\" style=\"margin-top:0\">
      <input type=\"hidden\" name=\"source_slug\" value=\"{slug}\">
      <select name=\"merge_target\" required style=\"width:100%;margin-bottom:4px\">
        <option value=\"\" disabled selected>Merge into\u2026</option>
        {options_html}
      </select>
      <div class=\"button-row\">
        <button type=\"submit\">Merge</button>
      </div>
      <script>
        document.getElementById("{form_id}").addEventListener("submit", function(e) {{
          var sel = this.merge_target.value;
          if (!sel) return e.preventDefault();
          var parts = sel.split(":");
          if (parts[0] === "identity") {{
            this.action = "/gallery/promote";
            var nameInput = document.createElement("input");
            nameInput.type = "hidden"; nameInput.name = "name"; nameInput.value = parts[1];
            this.appendChild(nameInput);
            this.querySelector("[name=source_slug]").name = "unknown_slug";
          }} else {{
            this.action = "/gallery/merge-unknowns";
            var targetInput = document.createElement("input");
            targetInput.type = "hidden";
            targetInput.name = "target_slug";
            targetInput.value = parts[1];
            this.appendChild(targetInput);
          }}
        }});
      </script>
    </form>"""

    return f"""
<div class=\"card\">
  {preview}
  <div>
    <h3 class=\"card-name\">{slug}</h3>
    <p class=\"meta\">{sample_count} captures</p>
    <form action=\"/gallery/promote\" method=\"post\">
      <input type=\"hidden\" name=\"unknown_slug\" value=\"{slug}\">
      <input type=\"text\" name=\"name\" placeholder=\"Name to promote as\" required>
      <div class=\"button-row\">
        <button type=\"submit\">Promote</button>
        </div>
    </form>{merge_form}
    <form action=\"/gallery/delete-unknown\" method=\"post\" style=\"margin-top:0\">
      <input type=\"hidden\" name=\"unknown_slug\" value=\"{slug}\">
      <div class=\"button-row\">
        <button class=\"delete\" type=\"submit\">Discard</button>
      </div>
    </form>
  </div>
</div>
"""


def _render_identity_detail_page(record: IdentityRecord, images: list[str]) -> str:
    name = html.escape(record.name)
    slug = html.escape(record.slug)
    sample_count = record.sample_count

    image_cards: list[str] = []
    for img in images:
        safe_img = html.escape(img)
        img_url = "/gallery/image?kind=identity&slug=" + quote(record.slug) + "&file=" + quote(img)
        image_cards.append(f"""
<div class=\"sample-card\">
  <img class=\"sample-img\" src=\"{img_url}\" alt=\"{safe_img}\">
  <form action=\"/gallery/delete-sample\" method=\"post\"
        onsubmit=\"return confirm('Delete this sample?')\">
    <input type=\"hidden\" name=\"slug\" value=\"{slug}\">
    <input type=\"hidden\" name=\"filename\" value=\"{safe_img}\">
    <button class=\"delete\" type=\"submit\">Delete</button>
  </form>
</div>""")

    grid = "\n".join(image_cards) if image_cards else '<p class="empty">No sample images.</p>'

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{name} — Valenia Gallery</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #000;
      --border: #333;
      --text: #eee;
      --muted: #999;
      --danger: #e55;
    }}
    body {{
      margin: 0;
      font-family: \"IBM Plex Sans\", \"Segoe UI\", system-ui, sans-serif;
      color: var(--text);
      background: var(--bg);
      min-height: 100vh;
    }}
    main {{
      width: min(100%, 1380px);
      margin: 0 auto;
      padding: 1rem;
      box-sizing: border-box;
    }}
    a {{
      color: var(--text);
      text-decoration: underline;
      text-underline-offset: 0.2em;
    }}
    a:hover {{
      color: #fff;
    }}
    nav {{
      display: flex;
      align-items: baseline;
      gap: 1.5rem;
      padding: 0 0 1rem;
      border-bottom: 1px solid var(--border);
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }}
    nav strong {{
      font-size: 1.1rem;
      margin-right: auto;
    }}
    nav a {{
      font-size: 0.9rem;
      color: var(--muted);
    }}
    nav a:hover {{
      color: var(--text);
    }}
    h1 {{
      font-size: 1.2rem;
      margin: 0 0 0.25rem;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.85rem;
      margin: 0 0 1rem;
    }}
    form {{
      display: grid;
      gap: 0.5rem;
      margin-top: 0.4rem;
    }}
    input, button {{
      font: inherit;
    }}
    input[type=\"text\"] {{
      padding: 0.5rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: #111;
      color: var(--text);
    }}
    .button-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    button {{
      width: fit-content;
      padding: 0.55rem 0.85rem;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: var(--text);
      color: #000;
      font-weight: 700;
      cursor: pointer;
    }}
    button.delete {{
      background: transparent;
      color: #fcc;
      border-color: rgba(238, 85, 85, 0.3);
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: end;
      margin-bottom: 1.5rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }}
    .actions form {{
      margin-top: 0;
    }}
    .sample-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 0.75rem;
    }}
    .sample-card {{
      display: grid;
      gap: 0.3rem;
    }}
    .sample-img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: cover;
      border-radius: 4px;
      background: #111;
    }}
    .sample-card form {{
      margin-top: 0;
    }}
    .empty {{
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <nav>
      <strong>Valenia</strong>
      <a href=\"/gallery\">Gallery</a>
      <a href=\"/\">Live Feed</a>
    </nav>
    <h1>{name}</h1>
    <p class=\"meta\">{slug} &middot; {sample_count} samples</p>
    <div class=\"actions\">
      <form action=\"/gallery/rename\" method=\"post\"
            style=\"display:flex;gap:0.5rem;align-items:end\">
        <input type=\"hidden\" name=\"slug\" value=\"{slug}\">
        <input type=\"text\" name=\"name\" value=\"{name}\" required placeholder=\"New name\">
        <button type=\"submit\">Rename</button>
      </form>
      <form action=\"/gallery/delete-identity\" method=\"post\"
            onsubmit=\"return confirm('Delete identity {name} and all samples?')\">
        <input type=\"hidden\" name=\"slug\" value=\"{slug}\">
        <button class=\"delete\" type=\"submit\">Delete Identity</button>
      </form>
    </div>
    <div class=\"actions\">
      <form action=\"/gallery/upload-samples\" method=\"post\" enctype=\"multipart/form-data\"
            style=\"display:flex;gap:0.5rem;align-items:end;flex-wrap:wrap\">
        <input type=\"hidden\" name=\"slug\" value=\"{slug}\">
        <input type=\"file\" name=\"photos\" accept=\"image/*\" multiple required
               style=\"font-size:0.85rem;color:var(--muted)\">
        <button type=\"submit\">Upload Photos</button>
      </form>
    </div>
    <h2 style=\"font-size:0.85rem;text-transform:uppercase;
      letter-spacing:0.06em;color:var(--muted);
      font-weight:600;margin:0 0 0.75rem\">Samples</h2>
    <div class=\"sample-grid\">
      {grid}
    </div>
  </main>
</body>
</html>
"""


def _render_preview_image(kind: str, slug: str, filename: object) -> str:
    if not isinstance(filename, str) or not filename:
        return '<div class="thumb empty">No preview</div>'
    url = "/gallery/image?kind=" + quote(kind) + "&slug=" + quote(slug) + "&file=" + quote(filename)
    alt = html.escape(f"{kind} preview")
    return f'<img class="thumb" src="{url}" alt="{alt}">'


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
