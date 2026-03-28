"""HTTP request handlers for the live camera server.

Origin: Valenia scripts/live_camera_server.py — LiveCameraHandler + render functions.
"""

from __future__ import annotations

import html
import json
from email.parser import BytesParser
from email.policy import default as email_default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote, urlparse

if TYPE_CHECKING:
    from src.gallery import EnrollmentResult, IdentityRecord, UnknownRecord
    from src.live import LiveRuntime

    from server.streamer import CameraStreamer

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class LiveCameraHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    streamer: CameraStreamer


class LiveCameraHandler(BaseHTTPRequestHandler):
    server_version = "FaceRecognition/0.2"

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        routes: dict[str, object] = {
            "/": self._serve_index,
            "/gallery": self._serve_gallery,
            "/gallery/identity": lambda: self._serve_identity_detail(parsed.query),
            "/gallery/image": lambda: self._serve_gallery_image(parsed.query),
            "/stream.mjpg": self._serve_mjpeg,
            "/metrics.json": self._serve_metrics_json,
        }
        handler = routes.get(parsed.path)
        if handler is not None:
            handler()  # type: ignore[operator]
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        routes: dict[str, object] = {
            "/enroll": self._handle_enroll,
            "/gallery/promote": self._handle_gallery_promote,
            "/gallery/rename": self._handle_gallery_rename,
            "/gallery/merge-unknowns": self._handle_gallery_merge_unknowns,
            "/gallery/delete-unknown": self._handle_gallery_delete_unknown,
            "/gallery/delete-identity": self._handle_gallery_delete_identity,
            "/gallery/delete-sample": self._handle_gallery_delete_sample,
            "/gallery/upload-samples": self._handle_gallery_upload_samples,
        }
        handler = routes.get(parsed.path)
        if handler is not None:
            handler()  # type: ignore[operator]
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    @property
    def streamer(self) -> CameraStreamer:
        return self.server.streamer  # type: ignore[attr-defined]

    @property
    def runtime(self) -> LiveRuntime:
        return self.streamer.runtime

    # -- GET handlers --------------------------------------------------

    def _serve_index(self) -> None:
        index_path = _TEMPLATE_DIR / "index.html"
        body = index_path.read_bytes()
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
        self._send_html(body)

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
        self.send_header(
            "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
        )
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
                self.wfile.write(
                    f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(payload)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

    def _serve_metrics_json(self) -> None:
        body = json.dumps(
            self.runtime.metrics_snapshot, indent=2, sort_keys=True
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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
        self._send_html(body)

    # -- POST handlers -------------------------------------------------

    def _handle_enroll(self) -> None:
        try:
            name, uploads = self._parse_multipart_request("name")
            result = self.runtime.enroll(name, uploads)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Enrollment failed", str(exc))
            return
        except Exception as exc:
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR, "Enrollment failed", str(exc)
            )
            return
        self._serve_message_page(
            HTTPStatus.OK, "Enrollment saved", _format_enrollment_message(result)
        )

    def _handle_gallery_promote(self) -> None:
        self._gallery_action(
            lambda f: self.runtime.promote_unknown(f["unknown_slug"], f["name"]),
            "Promotion failed",
            "/gallery",
        )

    def _handle_gallery_rename(self) -> None:
        self._gallery_action(
            lambda f: self.runtime.rename_identity(f["slug"], f["name"]),
            "Rename failed",
            "/gallery",
        )

    def _handle_gallery_merge_unknowns(self) -> None:
        self._gallery_action(
            lambda f: self.runtime.merge_unknowns(f["target_slug"], f["source_slug"]),
            "Merge failed",
            "/gallery",
        )

    def _handle_gallery_delete_unknown(self) -> None:
        self._gallery_action(
            lambda f: self.runtime.delete_unknown(f["unknown_slug"]),
            "Delete failed",
            "/gallery",
        )

    def _handle_gallery_delete_identity(self) -> None:
        self._gallery_action(
            lambda f: self.runtime.delete_identity(f["slug"]),
            "Delete failed",
            "/gallery",
        )

    def _handle_gallery_delete_sample(self) -> None:
        fields = self._parse_form_fields()
        slug = fields.get("slug", "")
        try:
            self.runtime.delete_identity_sample(slug, fields.get("filename", ""))
        except (ValueError, Exception) as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Delete failed", str(exc))
            return
        self._redirect(f"/gallery/identity?slug={quote(slug)}")

    def _handle_gallery_upload_samples(self) -> None:
        try:
            slug, uploads = self._parse_multipart_request("slug")
            result = self.runtime.upload_to_identity(slug, uploads)
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, "Upload failed", str(exc))
            return
        except Exception as exc:
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR, "Upload failed", str(exc)
            )
            return
        self._redirect(f"/gallery/identity?slug={quote(result.slug)}")

    # -- Helpers -------------------------------------------------------

    def _gallery_action(
        self,
        action: object,
        error_title: str,
        redirect_to: str,
    ) -> None:
        try:
            fields = self._parse_form_fields()
            action(fields)  # type: ignore[operator]
        except ValueError as exc:
            self._serve_message_page(HTTPStatus.BAD_REQUEST, error_title, str(exc))
            return
        except Exception as exc:
            self._serve_message_page(
                HTTPStatus.INTERNAL_SERVER_ERROR, error_title, str(exc)
            )
            return
        self._redirect(redirect_to)

    def _parse_multipart_request(
        self, text_field_name: str
    ) -> tuple[str, list[tuple[str, bytes]]]:
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
        envelope = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
            + payload
        )
        message = BytesParser(policy=email_default_policy).parsebytes(envelope)
        if not message.is_multipart():
            raise ValueError("Expected multipart form fields")

        text_value = ""
        uploads: list[tuple[str, bytes]] = []
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue
            field_name = part.get_param("name", header="Content-Disposition")
            if field_name is None:
                continue
            if field_name == text_field_name:
                content = part.get_content()
                if isinstance(content, str):
                    text_value = content.strip()
                continue
            if field_name == "photos":
                body = part.get_payload(decode=True)
                if not isinstance(body, bytes):
                    continue
                filename = part.get_filename() or "upload"
                uploads.append((filename, body))

        if not text_value:
            raise ValueError(f"{text_field_name} is required")
        if not uploads:
            raise ValueError("At least one photo is required")
        return text_value, uploads

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
        return {
            key: values[0].strip() for key, values in raw_fields.items() if values
        }

    def _send_html(self, body: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _serve_message_page(
        self, status: HTTPStatus, title: str, message: str
    ) -> None:
        body = _render_message_page(title, message).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


# ---------------------------------------------------------------------------
# HTML render functions (extracted from Valenia's server monolith)
# ---------------------------------------------------------------------------


def _render_message_page(title: str, message: str) -> str:
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif;
      background: #000; color: #eee; min-height: 100vh;
      display: grid; place-items: center; }}
    main {{ width: min(100%, 720px); padding: 1rem; box-sizing: border-box; }}
    a {{ color: #eee; text-decoration: underline; }}
    pre {{ white-space: pre-wrap; color: #999; }}
    .links {{ display: flex; gap: 1.5rem; margin-top: 1rem; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <main>
    <h1>{safe_title}</h1>
    <pre>{safe_message}</pre>
    <div class="links">
      <a href="/">Live feed</a>
      <a href="/gallery">Gallery</a>
    </div>
  </main>
</body>
</html>"""


def _render_gallery_page(
    *,
    identities: list[IdentityRecord],
    unknowns: list[UnknownRecord],
) -> str:
    identity_cards = "\n".join(
        _render_identity_card(record) for record in identities
    )
    unknown_cards = "\n".join(
        _render_unknown_card(record, unknowns=unknowns, identities=identities)
        for record in unknowns
    )
    if not identity_cards:
        identity_cards = '<p class="empty">No confirmed identities yet.</p>'
    if not unknown_cards:
        unknown_cards = '<p class="empty">No auto-captured unknowns yet.</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gallery</title>
  <style>
    :root {{ color-scheme: dark; --bg: #000; --border: #333;
      --text: #eee; --muted: #999; --danger: #e55; }}
    body {{ margin: 0; font-family: system-ui, sans-serif;
      color: var(--text); background: var(--bg); }}
    main {{ width: min(100%, 1380px); margin: 0 auto; padding: 1rem; }}
    a {{ color: var(--text); text-decoration: underline; }}
    nav {{ display: flex; gap: 1.5rem; padding: 0 0 1rem;
      border-bottom: 1px solid var(--border); margin-bottom: 1rem; flex-wrap: wrap; }}
    nav strong {{ font-size: 1.1rem; margin-right: auto; }}
    nav a {{ font-size: 0.9rem; color: var(--muted); }}
    h2 {{ font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--muted); font-weight: 600; }}
    .columns {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }}
    .card {{ display: grid; grid-template-columns: 100px minmax(0, 1fr); gap: 0.75rem;
      padding: 0.75rem 0; border-bottom: 1px solid #1a1a1a; }}
    .thumb {{ width: 100px; aspect-ratio: 1; object-fit: cover; border-radius: 4px; }}
    .thumb.empty {{ display: grid; place-items: center; border: 1px solid var(--border);
      color: var(--muted); font-size: 0.8rem; }}
    .card-name {{ margin: 0; font-size: 1rem; text-transform: none; letter-spacing: 0; }}
    .meta {{ color: var(--muted); font-size: 0.85rem; }}
    form {{ display: grid; gap: 0.5rem; margin-top: 0.4rem; }}
    input, button {{ font: inherit; }}
    input[type="text"], input[type="file"] {{ padding: 0.5rem; border-radius: 4px;
      border: 1px solid var(--border); background: #111; color: var(--text); }}
    .button-row {{ display: flex; flex-wrap: wrap; gap: 0.5rem; }}
    button {{ width: fit-content; padding: 0.55rem 0.85rem; border-radius: 4px;
      border: 1px solid var(--border); background: var(--text); color: #000;
      font-weight: 700; cursor: pointer; }}
    button.delete {{ background: transparent; color: #fcc;
      border-color: rgba(238, 85, 85, 0.3); }}
    .empty {{ color: var(--muted); }}
    .enroll-section {{ margin-bottom: 1.5rem; padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border); }}
    .enroll-form {{ display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: end; }}
    .enroll-form label {{ display: grid; gap: 0.25rem; font-size: 0.85rem;
      color: var(--muted); }}
    @media (max-width: 980px) {{ .columns {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <nav>
      <strong>Face Recognition</strong>
      <a href="/">Live Feed</a>
      <a href="/stream.mjpg">Stream</a>
      <a href="/metrics.json">Metrics</a>
    </nav>
    <section class="enroll-section">
      <h2>Enroll Identity</h2>
      <form class="enroll-form" action="/enroll" method="post"
            enctype="multipart/form-data">
        <label>Name <input type="text" name="name" required></label>
        <label>Photos <input type="file" name="photos" accept="image/*"
          multiple required></label>
        <button type="submit">Enroll</button>
      </form>
    </section>
    <div class="columns">
      <section>
        <h2>Confirmed Identities</h2>
        <div>{identity_cards}</div>
      </section>
      <section>
        <h2>Unknown Review Inbox</h2>
        <div>{unknown_cards}</div>
      </section>
    </div>
  </main>
</body>
</html>"""


def _render_identity_card(record: IdentityRecord) -> str:
    name = html.escape(record.name)
    slug = html.escape(record.slug)
    detail_url = "/gallery/identity?slug=" + quote(record.slug)
    preview = _render_preview_image("identity", record.slug, record.preview_filename)
    return f"""<div class="card">
  {preview}
  <div>
    <h3 class="card-name"><a href="{detail_url}">{name}</a></h3>
    <p class="meta">{slug} &middot; {record.sample_count} samples</p>
    <form action="/gallery/rename" method="post">
      <input type="hidden" name="slug" value="{slug}">
      <input type="text" name="name" value="{name}" required>
      <div class="button-row"><button type="submit">Rename</button></div>
    </form>
    <form action="/gallery/delete-identity" method="post"
          onsubmit="return confirm('Delete {name}?')" style="margin-top:0">
      <input type="hidden" name="slug" value="{slug}">
      <div class="button-row"><button class="delete" type="submit">Delete</button></div>
    </form>
  </div>
</div>"""


def _render_unknown_card(
    record: UnknownRecord,
    *,
    unknowns: list[UnknownRecord],
    identities: list[IdentityRecord],
) -> str:
    slug = html.escape(record.slug)
    preview = _render_preview_image("unknown", record.slug, record.preview_filename)

    merge_options: list[str] = []
    for other in unknowns:
        if other.slug == record.slug:
            continue
        s = html.escape(other.slug)
        merge_options.append(f'<option value="unknown:{s}">{s} ({other.sample_count})</option>')
    for identity in identities:
        n = html.escape(identity.name)
        merge_options.append(
            f'<option value="identity:{n}">{n} ({identity.sample_count})</option>'
        )

    merge_form = ""
    if merge_options:
        options_html = "\n".join(merge_options)
        form_id = f"merge-{slug}"
        merge_form = f"""
    <form id="{form_id}" method="post" style="margin-top:0">
      <input type="hidden" name="source_slug" value="{slug}">
      <select name="merge_target" required style="width:100%;margin-bottom:4px">
        <option value="" disabled selected>Merge into\u2026</option>
        {options_html}
      </select>
      <div class="button-row"><button type="submit">Merge</button></div>
      <script>
        document.getElementById("{form_id}").addEventListener("submit", function(e) {{
          var sel = this.merge_target.value;
          if (!sel) return e.preventDefault();
          var p = sel.split(":");
          if (p[0] === "identity") {{
            this.action = "/gallery/promote";
            var ni = document.createElement("input");
            ni.type="hidden"; ni.name="name"; ni.value=p[1];
            this.appendChild(ni);
            this.querySelector("[name=source_slug]").name="unknown_slug";
          }} else {{
            this.action = "/gallery/merge-unknowns";
            var ti = document.createElement("input");
            ti.type="hidden"; ti.name="target_slug"; ti.value=p[1];
            this.appendChild(ti);
          }}
        }});
      </script>
    </form>"""

    return f"""<div class="card">
  {preview}
  <div>
    <h3 class="card-name">{slug}</h3>
    <p class="meta">{record.sample_count} captures</p>
    <form action="/gallery/promote" method="post">
      <input type="hidden" name="unknown_slug" value="{slug}">
      <input type="text" name="name" placeholder="Name to promote as" required>
      <div class="button-row"><button type="submit">Promote</button></div>
    </form>{merge_form}
    <form action="/gallery/delete-unknown" method="post" style="margin-top:0">
      <input type="hidden" name="unknown_slug" value="{slug}">
      <div class="button-row"><button class="delete" type="submit">Discard</button></div>
    </form>
  </div>
</div>"""


def _render_identity_detail_page(record: IdentityRecord, images: list[str]) -> str:
    name = html.escape(record.name)
    slug = html.escape(record.slug)

    image_cards: list[str] = []
    for img in images:
        safe_img = html.escape(img)
        img_url = (
            "/gallery/image?kind=identity&slug="
            + quote(record.slug)
            + "&file="
            + quote(img)
        )
        image_cards.append(f"""<div class="sample-card">
  <img class="sample-img" src="{img_url}" alt="{safe_img}">
  <form action="/gallery/delete-sample" method="post"
        onsubmit="return confirm('Delete this sample?')">
    <input type="hidden" name="slug" value="{slug}">
    <input type="hidden" name="filename" value="{safe_img}">
    <button class="delete" type="submit">Delete</button>
  </form>
</div>""")

    grid = "\n".join(image_cards) if image_cards else '<p class="empty">No samples.</p>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{name} — Gallery</title>
  <style>
    :root {{ color-scheme: dark; --bg: #000; --border: #333;
      --text: #eee; --muted: #999; }}
    body {{ margin: 0; font-family: system-ui, sans-serif;
      color: var(--text); background: var(--bg); }}
    main {{ width: min(100%, 1380px); margin: 0 auto; padding: 1rem; }}
    a {{ color: var(--text); text-decoration: underline; }}
    nav {{ display: flex; gap: 1.5rem; padding: 0 0 1rem;
      border-bottom: 1px solid var(--border); margin-bottom: 1rem; }}
    nav strong {{ font-size: 1.1rem; margin-right: auto; }}
    nav a {{ font-size: 0.9rem; color: var(--muted); }}
    .meta {{ color: var(--muted); font-size: 0.85rem; }}
    form {{ display: grid; gap: 0.5rem; margin-top: 0.4rem; }}
    input, button {{ font: inherit; }}
    input[type="text"] {{ padding: 0.5rem; border-radius: 4px;
      border: 1px solid var(--border); background: #111; color: var(--text); }}
    .button-row {{ display: flex; gap: 0.5rem; }}
    button {{ padding: 0.55rem 0.85rem; border-radius: 4px;
      border: 1px solid var(--border); background: var(--text); color: #000;
      font-weight: 700; cursor: pointer; }}
    button.delete {{ background: transparent; color: #fcc;
      border-color: rgba(238,85,85,0.3); }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: end;
      margin-bottom: 1.5rem; padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border); }}
    .actions form {{ margin-top: 0; }}
    .sample-grid {{ display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 0.75rem; }}
    .sample-img {{ width: 100%; aspect-ratio: 1; object-fit: cover;
      border-radius: 4px; background: #111; }}
    .sample-card form {{ margin-top: 0; }}
    .empty {{ color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <nav>
      <strong>Face Recognition</strong>
      <a href="/gallery">Gallery</a>
      <a href="/">Live Feed</a>
    </nav>
    <h1 style="font-size:1.2rem">{name}</h1>
    <p class="meta">{slug} &middot; {record.sample_count} samples</p>
    <div class="actions">
      <form action="/gallery/rename" method="post"
            style="display:flex;gap:0.5rem;align-items:end">
        <input type="hidden" name="slug" value="{slug}">
        <input type="text" name="name" value="{name}" required>
        <button type="submit">Rename</button>
      </form>
      <form action="/gallery/delete-identity" method="post"
            onsubmit="return confirm('Delete {name}?')">
        <input type="hidden" name="slug" value="{slug}">
        <button class="delete" type="submit">Delete Identity</button>
      </form>
    </div>
    <div class="actions">
      <form action="/gallery/upload-samples" method="post" enctype="multipart/form-data"
            style="display:flex;gap:0.5rem;align-items:end;flex-wrap:wrap">
        <input type="hidden" name="slug" value="{slug}">
        <input type="file" name="photos" accept="image/*" multiple required
               style="font-size:0.85rem;color:var(--muted)">
        <button type="submit">Upload Photos</button>
      </form>
    </div>
    <h2 style="font-size:0.85rem;text-transform:uppercase;letter-spacing:0.06em;
      color:var(--muted);font-weight:600">Samples</h2>
    <div class="sample-grid">{grid}</div>
  </main>
</body>
</html>"""


def _render_preview_image(kind: str, slug: str, filename: object) -> str:
    if not isinstance(filename, str) or not filename:
        return '<div class="thumb empty">No preview</div>'
    url = (
        "/gallery/image?kind=" + quote(kind) + "&slug=" + quote(slug) + "&file=" + quote(filename)
    )
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
