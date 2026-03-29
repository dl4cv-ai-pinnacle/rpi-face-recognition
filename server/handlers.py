"""HTTP request handlers for the live camera server.

Origin: Valenia scripts/live_camera_server.py — LiveCameraHandler + render functions.
"""

from __future__ import annotations

import base64
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
    from src.config import AppConfig
    from src.contracts import Float32Array, UInt8Array
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
            "/settings": self._serve_settings,
            "/gallery": self._serve_gallery,
            "/gallery/identity": lambda: self._serve_identity_detail(parsed.query),
            "/gallery/image": lambda: self._serve_gallery_image(parsed.query),
            "/enroll-wizard": lambda: self._serve_enroll_wizard(parsed.query),
            "/stream.mjpg": self._serve_mjpeg,
            "/metrics.json": self._serve_metrics_json,
            "/style.css": self._serve_static_css,
            "/api/config": self._serve_api_config,
            "/api/config/backends": self._serve_api_backends,
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
            "/api/capture-frame": self._handle_api_capture_frame,
            "/api/enroll-captures": self._handle_api_enroll_captures,
            "/api/config": self._handle_api_config_update,
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

    def _serve_static_css(self) -> None:
        css_path = _TEMPLATE_DIR / "style.css"
        body = css_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/css; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600")
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

    def _serve_settings(self) -> None:
        body = _render_settings_page(self.runtime.config).encode("utf-8")
        self._send_html(body)

    def _serve_api_config(self) -> None:
        """GET /api/config — return current active config as JSON."""
        config = self.runtime.config
        data = {
            "detection": {"backend": config.detection.backend},
            "alignment": {"method": config.alignment.method},
            "embedding": {"quantize_int8": config.embedding.quantize_int8},
            "matching": {"threshold": config.matching.threshold},
            "tracking": {
                "max_missed": config.tracking.max_missed,
                "smoothing": config.tracking.smoothing,
            },
            "live": {
                "det_every": config.live.det_every,
                "match_threshold": config.live.match_threshold,
            },
            "gallery": {
                "enrich_min_quality": config.gallery.enrich_min_quality,
                "enrich_cooldown_seconds": config.gallery.enrich_cooldown_seconds,
            },
        }
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_api_backends(self) -> None:
        """GET /api/config/backends — available backends per stage."""
        data = {
            "detection": ["insightface", "ultraface"],
            "alignment": ["cv2", "skimage"],
        }
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_api_config_update(self) -> None:
        """POST /api/config — update config and rebuild affected components."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
            return
        payload = self.rfile.read(max(0, content_length))
        try:
            updates = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return

        from dataclasses import replace

        from src.alignment import create_aligner
        from src.detection import create_detector
        from src.embedding import create_embedder
        from src.pipeline import FacePipeline

        config = self.runtime.config
        needs_rebuild = False

        # Detection backend change
        det_config = config.detection
        if "detection" in updates and "backend" in updates["detection"]:
            new_backend = updates["detection"]["backend"]
            if new_backend != det_config.backend:
                det_config = replace(det_config, backend=new_backend)
                needs_rebuild = True

        # Alignment method change
        align_config = config.alignment
        if "alignment" in updates and "method" in updates["alignment"]:
            new_method = updates["alignment"]["method"]
            if new_method != align_config.method:
                align_config = replace(align_config, method=new_method)
                needs_rebuild = True

        # Embedding INT8 toggle
        embed_config = config.embedding
        if "embedding" in updates and "quantize_int8" in updates["embedding"]:
            new_int8 = bool(updates["embedding"]["quantize_int8"])
            if new_int8 != embed_config.quantize_int8:
                embed_config = replace(embed_config, quantize_int8=new_int8)
                needs_rebuild = True

        # Tunable params (no rebuild needed)
        live_config = config.live
        if "live" in updates:
            live_updates = updates["live"]
            if "det_every" in live_updates:
                live_config = replace(live_config, det_every=int(live_updates["det_every"]))
            if "match_threshold" in live_updates:
                live_config = replace(
                    live_config, match_threshold=float(live_updates["match_threshold"])
                )

        matching_config = config.matching
        if "matching" in updates and "threshold" in updates["matching"]:
            matching_config = replace(
                matching_config, threshold=float(updates["matching"]["threshold"])
            )

        new_config = replace(
            config,
            detection=det_config,
            alignment=align_config,
            embedding=embed_config,
            live=live_config,
            matching=matching_config,
        )
        self.runtime.config = new_config  # type: ignore[misc]

        if needs_rebuild:
            try:
                detector = create_detector(det_config)
                aligner = create_aligner(align_config)
                embedder = create_embedder(embed_config)
                new_pipeline = FacePipeline(
                    detector=detector,
                    aligner=aligner,
                    embedder=embedder,
                    max_faces=live_config.max_faces,
                )
                self.runtime.swap_pipeline(new_pipeline)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

        body = json.dumps({"ok": True, "rebuilt": needs_rebuild}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_enroll_wizard(self, query_string: str) -> None:
        params = parse_qs(query_string, keep_blank_values=False)
        slug = params.get("slug", [""])[0]
        target_record: IdentityRecord | None = None
        if slug:
            identities = self.runtime.list_identities()
            target_record = next((r for r in identities if r.slug == slug), None)
            if target_record is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Identity not found")
                return
        body = _render_enroll_wizard_page(target_record=target_record).encode("utf-8")
        self._send_html(body)

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
            self._serve_message_page(HTTPStatus.INTERNAL_SERVER_ERROR, "Upload failed", str(exc))
            return
        self._redirect(f"/gallery/identity?slug={quote(result.slug)}")

    def _handle_api_capture_frame(self) -> None:
        """POST /api/capture-frame — grab the latest frame, run detection, return JSON."""
        no_face: dict[str, object] = {
            "face_detected": False,
            "face_count": 0,
            "pose": "unknown",
            "jpeg_base64": "",
            "face_jpeg_base64": None,
        }
        try:
            self._handle_api_capture_frame_inner(no_face)
        except Exception as exc:
            no_face["error"] = str(exc)
            self._send_json(no_face)

    def _handle_api_capture_frame_inner(self, no_face: dict[str, object]) -> None:
        from src.quality import estimate_face_pose

        snapshot = self.streamer.wait_for_frame(last_seen=-1, timeout=3.0)
        source_jpeg = snapshot.raw_jpeg_bytes or snapshot.jpeg_bytes
        if snapshot.error is not None or source_jpeg is None:
            self._send_json(no_face)
            return

        import cv2
        import numpy as np

        jpeg_arr = np.frombuffer(source_jpeg, dtype=np.uint8)
        decoded = cv2.imdecode(jpeg_arr, cv2.IMREAD_COLOR)
        if decoded is None:
            self._send_json(no_face)
            return
        frame_bgr = np.asarray(decoded, dtype=np.uint8)

        det_result, _ = self.runtime.pipeline.detect(frame_bgr)
        face_count = len(det_result.boxes)

        pose = "unknown"
        landmarks: list[list[float]] | None = None
        if face_count == 1 and det_result.kps is not None and len(det_result.kps) > 0:
            pose = estimate_face_pose(det_result.kps[0])
            landmarks = det_result.kps[0].astype(float).tolist()

        full_b64 = base64.b64encode(source_jpeg).decode("ascii")

        face_b64: str | None = None
        if face_count > 0:
            box = det_result.boxes[0]
            h, w = frame_bgr.shape[:2]
            x1 = max(0, int(float(box[0])))
            y1 = max(0, int(float(box[1])))
            x2 = min(w, int(float(box[2])))
            y2 = min(h, int(float(box[3])))
            if x2 > x1 and y2 > y1:
                face_crop = frame_bgr[y1:y2, x1:x2]
                ok, enc = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    face_b64 = base64.b64encode(enc.tobytes()).decode("ascii")

        self._send_json(
            {
                "face_detected": face_count > 0,
                "face_count": face_count,
                "pose": pose,
                "jpeg_base64": full_b64,
                "face_jpeg_base64": face_b64,
                "landmarks": landmarks,
            }
        )

    def _handle_api_enroll_captures(self) -> None:
        """POST /api/enroll-captures — enroll from base64-encoded captured frames."""
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, status=HTTPStatus.BAD_REQUEST)
            return

        payload = self.rfile.read(max(0, content_length))
        try:
            data = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
            return

        name = data.get("name", "").strip()
        raw_slug = data.get("slug", "")
        captures = data.get("captures", [])
        if raw_slug is None:
            slug = ""
        elif isinstance(raw_slug, str):
            slug = raw_slug.strip()
        else:
            self._send_json({"error": "Invalid identity slug"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not slug and not name:
            self._send_json({"error": "Name is required"}, status=HTTPStatus.BAD_REQUEST)
            return
        if not captures or not isinstance(captures, list):
            self._send_json(
                {"error": "At least one capture is required"}, status=HTTPStatus.BAD_REQUEST
            )
            return

        try:
            embeddings, uploads = self._prepare_enrollment_captures(captures)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if slug:
                result = self.runtime.upload_captured_to_identity(slug, embeddings, uploads)
                mode = "update"
            else:
                result = self.runtime.enroll_captured(name, embeddings, uploads)
                mode = "create"
        except ValueError as exc:
            message = str(exc)
            status = (
                HTTPStatus.NOT_FOUND
                if slug and message.startswith("Unknown identity:")
                else HTTPStatus.BAD_REQUEST
            )
            self._send_json({"error": message}, status=status)
            return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self._send_json(
            {
                "ok": True,
                "mode": mode,
                "name": result.name,
                "slug": result.slug,
                "sample_count": result.sample_count,
                "accepted_count": len(result.accepted_files),
                "accepted_files": list(result.accepted_files),
                "rejected_files": list(result.rejected_files),
            }
        )

    def _prepare_enrollment_captures(
        self, captures: list[object]
    ) -> tuple[list[Float32Array], list[tuple[str, bytes]]]:
        import numpy as np

        embeddings: list[Float32Array] = []
        uploads: list[tuple[str, bytes]] = []

        for idx, capture in enumerate(captures, start=1):
            label = f"capture_{idx}.jpg"
            if isinstance(capture, str):
                jpeg_bytes = self._decode_capture_base64(capture, label)
                frame_bgr = self._decode_capture_image(jpeg_bytes, label)
                det_result, _ = self.runtime.pipeline.detect(frame_bgr)
                if det_result.kps is None or len(det_result.boxes) != 1:
                    raise ValueError(f"Capture {idx} must contain exactly one clear face")
                emb, _ = self.runtime.pipeline.embed_from_kps(frame_bgr, det_result.kps[0])
                embeddings.append(np.asarray(emb, dtype=np.float32))
                uploads.append((label, jpeg_bytes))
                continue

            if not isinstance(capture, dict):
                raise ValueError(f"Capture {idx} has an unsupported payload shape")

            frame_b64 = capture.get("frame_jpeg_base64", "")
            face_b64 = capture.get("face_jpeg_base64", "")
            landmarks = capture.get("landmarks")
            if not isinstance(frame_b64, str) or not frame_b64:
                raise ValueError(f"Capture {idx} is missing frame_jpeg_base64")
            if landmarks is None:
                raise ValueError(f"Capture {idx} is missing landmarks")

            frame_bytes = self._decode_capture_base64(frame_b64, f"Capture {idx} frame")
            frame_bgr = self._decode_capture_image(frame_bytes, f"Capture {idx} frame")
            try:
                kps = np.asarray(landmarks, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Capture {idx} has invalid landmarks") from exc
            if kps.shape != (5, 2):
                raise ValueError(f"Capture {idx} landmarks must have shape 5x2")

            emb, _ = self.runtime.pipeline.embed_from_kps(frame_bgr, kps)
            embeddings.append(np.asarray(emb, dtype=np.float32))
            if isinstance(face_b64, str) and face_b64:
                face_bytes = self._decode_capture_base64(face_b64, f"Capture {idx} face")
            else:
                face_bytes = frame_bytes
            uploads.append((label, face_bytes))

        return embeddings, uploads

    def _decode_capture_base64(self, value: str, label: str) -> bytes:
        try:
            return base64.b64decode(value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{label} has invalid base64 encoding") from exc

    def _decode_capture_image(self, payload: bytes, label: str) -> UInt8Array:
        import cv2
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"{label} is not a readable image")
        return np.asarray(frame, dtype=np.uint8)

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
            self._serve_message_page(HTTPStatus.INTERNAL_SERVER_ERROR, error_title, str(exc))
            return
        self._redirect(redirect_to)

    def _parse_multipart_request(self, text_field_name: str) -> tuple[str, list[tuple[str, bytes]]]:
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
        return {key: values[0].strip() for key, values in raw_fields.items() if values}

    def _send_html(self, body: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


# ---------------------------------------------------------------------------
# HTML render functions — all use _page_shell for consistent layout
# ---------------------------------------------------------------------------


def _page_shell(title: str, content: str, *, current: str = "") -> str:
    """Wrap content in shared HTML boilerplate with consistent nav."""

    def _nav_link(href: str, label: str) -> str:
        aria = ' aria-current="page"' if href == current else ""
        return f'<a href="{href}"{aria}>{label}</a>'

    nav = "\n      ".join(
        [
            _nav_link("/", "Live Feed"),
            _nav_link("/gallery", "Gallery"),
            _nav_link("/enroll-wizard", "Enroll"),
            _nav_link("/settings", "Settings"),
            _nav_link("/metrics.json", "Metrics"),
        ]
    )

    repo_link = (
        '<a href="https://github.com/dl4cv-ai-pinnacle/rpi-face-recognition"'
        ' target="_blank" style="margin-left:auto;font-size:0.8rem;color:var(--muted)">'
        "GitHub</a>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} — Face Recognition</title>
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <main>
    <nav>
      <strong>Face Recognition</strong>
      {nav}
      {repo_link}
    </nav>
    {content}
  </main>
</body>
</html>"""


def _render_message_page(title: str, message: str) -> str:
    safe_title = html.escape(title)
    safe_message = html.escape(message)
    content = f"""<h1>{safe_title}</h1>
    <pre>{safe_message}</pre>
    <div class="links">
      <a href="/">Live feed</a>
      <a href="/gallery">Gallery</a>
    </div>"""
    return _page_shell(title, content)


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

    content = f"""<section class="enroll-section">
      <h2>Enroll Identity</h2>
      <p style="color:var(--muted);font-size:0.9rem;margin:0 0 0.75rem;line-height:1.5">
        For best recognition across different angles, upload <strong>3-5 photos</strong>
        from different poses: frontal, slight left, slight right. Each photo must
        contain exactly one clearly visible face.</p>
      <form class="enroll-form" action="/enroll" method="post"
            enctype="multipart/form-data">
        <label>Name <input type="text" name="name" required></label>
        <label>Photos <input type="file" name="photos" accept="image/*"
          multiple required></label>
        <button type="submit">Enroll</button>
      </form>
      <p style="color:var(--muted);font-size:0.85rem;margin:0.75rem 0 0">
        Or use the camera: <a href="/enroll-wizard">Enroll with Camera</a>
        &mdash; guided multi-pose capture for best recognition quality.</p>
    </section>
    <div class="columns">
      <section class="section-stack">
        <h2>Confirmed Identities</h2>
        <p class="section-summary">Known people you can rename, expand, or review in detail.</p>
        <div class="section-card">{identity_cards}</div>
      </section>
      <section class="section-stack">
        <h2>Unknowns</h2>
        <p class="section-summary">
          Recent captures that still need a name or should be discarded.
        </p>
        <div class="section-card">{unknown_cards}</div>
      </section>
    </div>"""
    return _page_shell("Gallery", content, current="/gallery")


def _render_identity_card(record: IdentityRecord) -> str:
    name = html.escape(record.name)
    slug = html.escape(record.slug)
    detail_url = "/gallery/identity?slug=" + quote(record.slug)
    camera_url = "/enroll-wizard?slug=" + quote(record.slug)
    preview = _render_preview_image("identity", record.slug, record.preview_filename)
    return f"""<div class="card">
  {preview}
  <div class="card-body">
    <h3 class="card-name"><a href="{detail_url}">{name}</a></h3>
    <p class="meta">{slug} &middot; {record.sample_count} samples</p>
    <div class="card-actions">
      <a class="text-button" href="{camera_url}">Add Samples with Camera</a>
      <a class="text-button subtle" href="{detail_url}">Open Details</a>
    </div>
    <form class="inline-form" action="/gallery/rename" method="post">
      <input type="hidden" name="slug" value="{slug}">
      <input type="text" name="name" value="{name}" required>
      <button type="submit">Rename</button>
    </form>
    <form action="/gallery/delete-identity" method="post"
          onsubmit="return confirm('Delete {name}?')" style="margin-top:0">
      <input type="hidden" name="slug" value="{slug}">
      <div class="button-row compact"><button class="delete" type="submit">Delete</button></div>
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
        merge_options.append(f'<option value="identity:{n}">{n} ({identity.sample_count})</option>')

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
  <div class="card-body">
    <h3 class="card-name">{slug}</h3>
    <p class="meta">{record.sample_count} captures</p>
    <form class="inline-form" action="/gallery/promote" method="post">
      <input type="hidden" name="unknown_slug" value="{slug}">
      <input type="text" name="name" placeholder="Name to promote as" required>
      <button type="submit">Promote</button>
    </form>{merge_form}
    <div class="card-actions">
      {merge_form}
      <form action="/gallery/delete-unknown" method="post" style="margin-top:0">
        <input type="hidden" name="unknown_slug" value="{slug}">
        <div class="button-row compact"><button class="delete" type="submit">Discard</button></div>
      </form>
    </div>
  </div>
</div>"""


def _render_identity_detail_page(record: IdentityRecord, images: list[str]) -> str:
    name = html.escape(record.name)
    slug = html.escape(record.slug)
    camera_url = "/enroll-wizard?slug=" + quote(record.slug)

    image_cards: list[str] = []
    for img in images:
        safe_img = html.escape(img)
        img_url = "/gallery/image?kind=identity&slug=" + quote(record.slug) + "&file=" + quote(img)
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

    content = f"""<h1>{name}</h1>
    <p class="meta">{slug} &middot; {record.sample_count} samples</p>
    <div class="actions">
      <form class="inline-form" action="/gallery/rename" method="post">
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
      <form class="inline-form" action="/gallery/upload-samples" method="post"
            enctype="multipart/form-data">
        <input type="hidden" name="slug" value="{slug}">
        <input type="file" name="photos" accept="image/*" multiple required
               style="font-size:0.85rem;color:var(--muted)">
        <button type="submit">Upload Photos</button>
      </form>
      <a class="text-button" href="{camera_url}">Add Samples with Camera</a>
      <p style="color:var(--muted);font-size:0.8rem;margin:0">
        Add photos from different angles to improve recognition.</p>
    </div>
    <h2>Samples</h2>
    <div class="sample-grid">{grid}</div>"""
    return _page_shell(name, content, current="/gallery")


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


def _render_settings_page(config: AppConfig) -> str:
    det_sel = lambda v: " selected" if config.detection.backend == v else ""  # noqa: E731
    align_sel = lambda v: " selected" if config.alignment.method == v else ""  # noqa: E731
    int8_checked = " checked" if config.embedding.quantize_int8 else ""

    content = f"""<h1>Pipeline Settings</h1>
    <form id="settings-form" onsubmit="return applySettings(event)">

      <h2>Detection</h2>
      <div class="field">
        <label for="det-backend">Backend</label>
        <select id="det-backend" name="detection.backend">
          <option value="insightface"{det_sel("insightface")}>SCRFD (insightface)</option>
          <option value="ultraface"{det_sel("ultraface")}>UltraFace (fast)</option>
        </select>
      </div>
      <p id="det-warning" style="color:var(--danger);font-size:0.85rem;margin:0.25rem 0 0.5rem;
        display:none">UltraFace has no landmarks — recognition uses center-crop fallback.
        Quality is lower, especially for non-frontal faces. Re-enroll if switching detectors.</p>
      <div class="field">
        <label for="det-every">Det every N frames</label>
        <input type="number" id="det-every" name="live.det_every"
               value="{config.live.det_every}" min="1" max="10">
      </div>

      <h2>Alignment</h2>
      <div class="field">
        <label for="align-method">Method</label>
        <select id="align-method" name="alignment.method">
          <option value="cv2"{align_sel("cv2")}>cv2 (LMEDS)</option>
          <option value="skimage"{align_sel("skimage")}>skimage</option>
        </select>
      </div>

      <h2>Embedding</h2>
      <div class="field">
        <label for="int8">INT8 quantize</label>
        <input type="checkbox" id="int8" name="embedding.quantize_int8"{int8_checked}>
      </div>

      <h2>Matching</h2>
      <div class="field">
        <label for="threshold">Threshold</label>
        <input type="number" id="threshold" name="matching.threshold"
               value="{config.matching.threshold}" min="0.1" max="0.9" step="0.05">
      </div>

      <h2>Tracking</h2>
      <div class="field">
        <label for="max-missed">Max missed frames</label>
        <input type="number" id="max-missed" name="tracking.max_missed"
               value="{config.tracking.max_missed}" min="1" max="10">
      </div>
      <div class="field">
        <label for="smoothing">Smoothing</label>
        <input type="number" id="smoothing" name="tracking.smoothing"
               value="{config.tracking.smoothing}" min="0" max="1" step="0.05">
      </div>

      <h2>Enrichment</h2>
      <div class="field">
        <label for="enrich-quality">Min quality</label>
        <input type="number" id="enrich-quality" name="gallery.enrich_min_quality"
               value="{config.gallery.enrich_min_quality}" min="0" max="1" step="0.05">
      </div>
      <div class="field">
        <label for="enrich-cooldown">Cooldown (s)</label>
        <input type="number" id="enrich-cooldown" name="gallery.enrich_cooldown_seconds"
               value="{config.gallery.enrich_cooldown_seconds}" min="1" max="120" step="1">
      </div>

      <button type="submit">Apply Changes</button>
      <div id="status" class="status"></div>
    </form>
  </main>
  <script>
    // Show warning when UltraFace is selected.
    (function() {{
      const sel = document.getElementById("det-backend");
      const warn = document.getElementById("det-warning");
      function toggle() {{ warn.style.display = sel.value === "ultraface" ? "block" : "none"; }}
      sel.addEventListener("change", toggle);
      toggle();
    }})();

    async function applySettings(e) {{
      e.preventDefault();
      const status = document.getElementById("status");
      status.textContent = "Applying...";
      status.className = "status";

      const payload = {{
        detection: {{
          backend: document.getElementById("det-backend").value,
        }},
        alignment: {{
          method: document.getElementById("align-method").value,
        }},
        embedding: {{
          quantize_int8: document.getElementById("int8").checked,
        }},
        matching: {{
          threshold: parseFloat(document.getElementById("threshold").value),
        }},
        live: {{
          det_every: parseInt(document.getElementById("det-every").value),
          match_threshold: parseFloat(document.getElementById("threshold").value),
        }},
      }};

      try {{
        const resp = await fetch("/api/config", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});
        const data = await resp.json();
        if (resp.ok) {{
          status.textContent = data.rebuilt
            ? "Applied — pipeline rebuilt"
            : "Applied — parameters updated";
          status.className = "status ok";
        }} else {{
          status.textContent = "Error: " + (data.error || resp.statusText);
          status.className = "status err";
        }}
      }} catch (err) {{
        status.textContent = "Network error: " + err.message;
        status.className = "status err";
      }}
    }}
  </script>"""
    return _page_shell("Settings", content, current="/settings")


def _render_enroll_wizard_page(*, target_record: IdentityRecord | None = None) -> str:
    target_name = target_record.name if target_record is not None else ""
    target_slug = target_record.slug if target_record is not None else ""
    escaped_target_name = html.escape(target_name)
    escaped_target_slug = html.escape(target_slug)
    target_name_json = json.dumps(target_name)
    target_slug_json = json.dumps(target_slug)
    title = "Add Camera Samples" if target_record is not None else "Enroll with Camera"
    intro = (
        f"""<h1>Add Camera Samples</h1>
    <p style="color:var(--muted);font-size:0.9rem;margin:0 0 0.35rem;line-height:1.5">
      Capture 3 photos for <strong>{escaped_target_name}</strong> from different angles.</p>
    <p style="color:var(--muted);font-size:0.85rem;margin:0 0 1rem;line-height:1.5">
      New samples will be added to <code>{escaped_target_slug}</code> without
      creating a new identity.</p>"""
        if target_record is not None
        else """<h1>Enroll with Camera</h1>
    <p style="color:var(--muted);font-size:0.9rem;margin:0 0 1rem;line-height:1.5">
      Capture 3 photos from different angles for robust recognition.
      Position yourself in front of the camera and follow the instructions.</p>"""
    )
    enroll_form = (
        f"""<div id="wizard-enroll" class="wizard-enroll" style="display:none">
            <p style="margin:0;color:var(--muted);font-size:0.85rem;line-height:1.45">
              Ready to add these samples to <strong>{escaped_target_name}</strong>.</p>
            <button onclick="submitEnrollment()">Add Samples</button>
          </div>"""
        if target_record is not None
        else """<div id="wizard-enroll" class="wizard-enroll" style="display:none">
            <label style="display:grid;gap:0.25rem;font-size:0.85rem;color:var(--muted)">
              Name
              <input type="text" id="enroll-name" required
                     placeholder="Enter name for enrollment">
            </label>
            <button onclick="submitEnrollment()">Enroll</button>
          </div>"""
    )

    content = (
        intro
        + """

    <div class="wizard">
      <div class="wizard-steps">
        <div class="wizard-step active" data-step="1">1. Frontal</div>
        <div class="wizard-step" data-step="2">2. Left</div>
        <div class="wizard-step" data-step="3">3. Right</div>
      </div>

      <div class="wizard-body">
        <div class="wizard-stream">
          <div id="wizard-stream-loading" class="wizard-stream-loading">
            Connecting camera...
          </div>
          <img id="wizard-feed" src="/stream.mjpg" alt="Live camera feed" class="wizard-feed">
          <div id="wizard-instruction" class="wizard-instruction">
            Look straight at the camera</div>
        </div>

        <div class="wizard-controls">
          <div id="wizard-captures" class="wizard-captures">
            <div class="wizard-thumb-slot" data-slot="1">
              <div class="wizard-thumb-empty">1</div>
            </div>
            <div class="wizard-thumb-slot" data-slot="2">
              <div class="wizard-thumb-empty">2</div>
            </div>
            <div class="wizard-thumb-slot" data-slot="3">
              <div class="wizard-thumb-empty">3</div>
            </div>
          </div>

          <div id="wizard-actions" class="wizard-actions">
            <button id="btn-capture" onclick="captureFrame()">Capture</button>
          </div>

          <div id="wizard-review" class="wizard-review" style="display:none">
            <img id="review-img" class="wizard-review-img" alt="Captured face">
            <div class="button-row">
              <button onclick="retakeCapture()">Retake</button>
              <button id="btn-next" onclick="nextStep()">Next</button>
            </div>
          </div>

    """
        + enroll_form
        + """

          <div id="wizard-status" class="status"></div>
        </div>
      </div>
    </div>

    <script>
    (function() {
      const TARGET_SLUG = """
        + target_slug_json
        + """;
      const TARGET_NAME = """
        + target_name_json
        + """;
      const STEPS = [
        "Look straight at the camera",
        "Turn your head slightly to the left",
        "Turn your head slightly to the right",
      ];
      const EXPECTED_POSES = ["frontal", "left", "right"];
      const POLL_MS = 500;
      const HOLD_POLLS = 2;  // consecutive matching polls (~1s at 500ms)

      let currentStep = 0;  // 0-indexed
      let captures = [null, null, null];  // base64 JPEG strings
      let faceThumbs = [null, null, null];  // base64 face crops for preview
      let capturing = false;
      let streamConnected = false;

      // Auto-capture state
      let poseMatchCount = 0;   // consecutive polls with matching pose
      let pollTimer = null;     // setInterval id
      let autoCaptureData = null;  // last poll data when pose matched

      const streamImg = document.getElementById("wizard-feed");
      const streamLoading = document.getElementById("wizard-stream-loading");

      function connectWizardStream() {
        streamConnected = false;
        streamImg.classList.remove("loaded");
        streamLoading.classList.remove("hidden");
        streamLoading.textContent = "Connecting camera...";
        streamImg.src = "/stream.mjpg?t=" + Date.now();
      }

      function checkWizardStreamAlive() {
        if (!streamConnected && streamImg.naturalWidth > 0) {
          streamConnected = true;
          streamImg.classList.add("loaded");
          streamLoading.classList.add("hidden");
        }
      }

      streamImg.addEventListener("load", checkWizardStreamAlive);
      document.addEventListener("visibilitychange", function() {
        if (!document.hidden) {
          connectWizardStream();
        }
      });
      window.addEventListener("pageshow", function(event) {
        if (event.persisted) {
          connectWizardStream();
        }
      });
      connectWizardStream();
      window.setInterval(checkWizardStreamAlive, 500);

      function startPolling() {
        stopPolling();
        poseMatchCount = 0;
        autoCaptureData = null;
        pollTimer = setInterval(pollForPose, POLL_MS);
      }

      function stopPolling() {
        if (pollTimer !== null) {
          clearInterval(pollTimer);
          pollTimer = null;
        }
        poseMatchCount = 0;
        autoCaptureData = null;
      }

      async function pollForPose() {
        if (capturing) return;
        if (captures[currentStep] !== null) return;

        var instrEl = document.getElementById("wizard-instruction");
        try {
          var resp = await fetch("/api/capture-frame", { method: "POST" });
          var data = await resp.json();
        } catch (err) {
          return;  // silently skip network errors during polling
        }

        // No pose detection available (UltraFace) — stay on manual mode
        if (data.pose === "unknown" && data.face_detected) {
          instrEl.classList.remove("pose-ok");
          return;
        }

        var expected = EXPECTED_POSES[currentStep];
        if (data.face_detected && data.face_count === 1 && data.pose === expected) {
          poseMatchCount++;
          autoCaptureData = data;
          instrEl.textContent = STEPS[currentStep] + " - hold pose...";
          instrEl.classList.add("pose-ok");

          if (poseMatchCount >= HOLD_POLLS) {
            // Auto-capture!
            const capturedData = autoCaptureData;
            stopPolling();
            if (capturedData) {
              applyCapture(capturedData);
            }
          }
        } else {
          poseMatchCount = 0;
          autoCaptureData = null;
          instrEl.textContent = STEPS[currentStep];
          instrEl.classList.remove("pose-ok");
        }
      }

      function applyCapture(data) {
        if (!data) {
          showStatus("Capture failed. Please try again.", "err");
          capturing = false;
          startPolling();
          return;
        }
        captures[currentStep] = buildCapturePayload(data);
        faceThumbs[currentStep] = data.face_jpeg_base64;
        updateThumb(currentStep);

        var instrEl = document.getElementById("wizard-instruction");
        instrEl.textContent = "Captured!";
        instrEl.classList.add("pose-ok");

        showStatus("", "");

        // Brief flash then auto-advance
        setTimeout(function() {
          instrEl.classList.remove("pose-ok");
          if (currentStep < 2) {
            currentStep++;
            updateStepUI();
            startPolling();
          } else {
            // All 3 captured — show save form
            document.getElementById("wizard-actions").style.display = "none";
            document.getElementById("wizard-enroll").style.display = "";
            if (!TARGET_SLUG) {
              document.getElementById("enroll-name").focus();
            }
          }
        }, 600);
      }

      window.captureFrame = async function() {
        if (capturing) return;
        capturing = true;
        stopPolling();
        const btn = document.getElementById("btn-capture");
        btn.textContent = "Capturing...";
        btn.disabled = true;

        try {
          const resp = await fetch("/api/capture-frame", { method: "POST" });
          const data = await resp.json();
          if (!data.face_detected) {
            showStatus("No face detected. Adjust position and try again.", "err");
            btn.textContent = "Capture";
            btn.disabled = false;
            capturing = false;
            startPolling();
            return;
          }
          if (data.face_count > 1) {
            showStatus("Multiple faces detected. Only one person should be visible.", "err");
            btn.textContent = "Capture";
            btn.disabled = false;
            capturing = false;
            startPolling();
            return;
          }

          captures[currentStep] = buildCapturePayload(data);
          faceThumbs[currentStep] = data.face_jpeg_base64;

          // Show review
          const reviewImg = document.getElementById("review-img");
          reviewImg.src = "data:image/jpeg;base64," + (data.face_jpeg_base64 || data.jpeg_base64);
          document.getElementById("wizard-actions").style.display = "none";
          document.getElementById("wizard-review").style.display = "";

          // Update thumbnail
          updateThumb(currentStep);

          // Change "Next" to "Finish" on last step
          document.getElementById("btn-next").textContent =
            currentStep === 2 ? "Finish" : "Next";

          showStatus("", "");
        } catch (err) {
          showStatus("Network error: " + err.message, "err");
        }
        btn.textContent = "Capture";
        btn.disabled = false;
        capturing = false;
      };

      window.retakeCapture = function() {
        captures[currentStep] = null;
        faceThumbs[currentStep] = null;
        resetThumb(currentStep);
        document.getElementById("wizard-review").style.display = "none";
        document.getElementById("wizard-actions").style.display = "";
        showStatus("", "");
        startPolling();
      };

      window.nextStep = function() {
        document.getElementById("wizard-review").style.display = "none";

        if (currentStep < 2) {
          // Advance to next step
          currentStep++;
          updateStepUI();
          document.getElementById("wizard-actions").style.display = "";
          startPolling();
        } else {
          // All 3 captured — show save form
          document.getElementById("wizard-actions").style.display = "none";
          document.getElementById("wizard-enroll").style.display = "";
          if (!TARGET_SLUG) {
            document.getElementById("enroll-name").focus();
          }
        }
      };

      window.submitEnrollment = async function() {
        stopPolling();
        const nameInput = document.getElementById("enroll-name");
        const name = TARGET_SLUG ? TARGET_NAME : (nameInput ? nameInput.value.trim() : "");
        if (!TARGET_SLUG && !name) {
          showStatus("Name is required.", "err");
          return;
        }
        const validCaptures = captures.filter(function(c) { return c !== null; });
        if (validCaptures.length === 0) {
          showStatus("No captures available.", "err");
          return;
        }

        showStatus("Enrolling...", "");
        try {
          const resp = await fetch("/api/enroll-captures", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: name,
              slug: TARGET_SLUG || "",
              captures: validCaptures,
            }),
          });
          const data = await resp.json();
          if (resp.ok && data.ok) {
            if (data.mode === "update") {
              showStatus(
                "Added " + data.accepted_count + " sample(s) to " + data.name + ".", "ok"
              );
              setTimeout(function() {
                window.location.href = "/gallery/identity?slug=" + encodeURIComponent(data.slug);
              }, 1500);
            } else {
              showStatus(
                "Enrolled " + data.name + " with " + data.sample_count + " samples.", "ok"
              );
              setTimeout(function() { window.location.href = "/gallery"; }, 1500);
            }
          } else {
            showStatus("Error: " + (data.error || "Unknown error"), "err");
          }
        } catch (err) {
          showStatus("Network error: " + err.message, "err");
        }
      };

      function updateStepUI() {
        document.getElementById("wizard-instruction").textContent = STEPS[currentStep];
        document.getElementById("wizard-instruction").classList.remove("pose-ok");
        var stepEls = document.querySelectorAll(".wizard-step");
        for (var i = 0; i < stepEls.length; i++) {
          stepEls[i].classList.toggle("active", i === currentStep);
          stepEls[i].classList.toggle("done", captures[i] !== null && i !== currentStep);
        }
      }

      function updateThumb(idx) {
        var slot = document.querySelector('.wizard-thumb-slot[data-slot="' + (idx + 1) + '"]');
        if (!slot) return;
        var src = faceThumbs[idx] || captures[idx];
        if (src) {
          slot.innerHTML = '<img class="wizard-thumb-img" src="data:image/jpeg;base64,' +
            src + '" alt="Capture ' + (idx + 1) + '">';
        }
      }

      function resetThumb(idx) {
        var slot = document.querySelector('.wizard-thumb-slot[data-slot="' + (idx + 1) + '"]');
        if (!slot) return;
        slot.innerHTML = '<div class="wizard-thumb-empty">' + (idx + 1) + '</div>';
      }

      function buildCapturePayload(data) {
        return {
          frame_jpeg_base64: data.jpeg_base64,
          face_jpeg_base64: data.face_jpeg_base64 || data.jpeg_base64,
          landmarks: data.landmarks || null
        };
      }

      function showStatus(msg, cls) {
        var el = document.getElementById("wizard-status");
        el.textContent = msg;
        el.className = "status" + (cls ? " " + cls : "");
      }

      // Start polling after a short delay to let the stream connect.
      setTimeout(startPolling, 2000);
    })();
    </script>"""
    )
    return _page_shell(title, content, current="/enroll-wizard")
