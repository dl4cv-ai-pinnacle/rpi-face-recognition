from __future__ import annotations

import base64
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from email.message import EmailMessage
from http import HTTPStatus
from socketserver import BaseServer

import cv2
import numpy as np
from server.handlers import LiveCameraHandler
from src.gallery import EnrollmentResult


@dataclass
class _DetectionResultStub:
    boxes: np.ndarray
    kps: np.ndarray


class _PipelineStub:
    def __init__(self) -> None:
        self.detect_calls = 0
        self.embed_calls = 0

    def detect(self, frame_bgr: np.ndarray) -> tuple[_DetectionResultStub, float]:
        del frame_bgr
        self.detect_calls += 1
        return (
            _DetectionResultStub(
                boxes=np.asarray([[0.0, 0.0, 16.0, 16.0, 0.99]], dtype=np.float32),
                kps=np.asarray(
                    [[[4.0, 5.0], [11.0, 5.0], [7.5, 8.0], [5.0, 11.0], [10.0, 11.0]]],
                    dtype=np.float32,
                ),
            ),
            0.0,
        )

    def embed_from_kps(
        self, frame_bgr: np.ndarray, landmarks: np.ndarray
    ) -> tuple[np.ndarray, float]:
        del frame_bgr, landmarks
        self.embed_calls += 1
        return np.ones(512, dtype=np.float32), 0.0

    @property
    def detector(self) -> _PipelineStub:
        return self

    @property
    def embedder(self) -> _PipelineStub:
        return self

    @property
    def provider_name(self) -> str:
        return "CPUExecutionProvider"


class _RuntimeStub:
    def __init__(
        self,
        *,
        enroll_result: EnrollmentResult | None = None,
        upload_result: EnrollmentResult | None = None,
        upload_error: Exception | None = None,
    ) -> None:
        self.enroll_result = enroll_result
        self.upload_result = upload_result
        self.upload_error = upload_error
        self.pipeline = _PipelineStub()

    def enroll_captured(
        self,
        name: str,
        embeddings: list[np.ndarray],
        uploads: list[tuple[str, bytes]],
    ) -> EnrollmentResult:
        assert name
        assert embeddings
        assert uploads
        assert self.enroll_result is not None
        return self.enroll_result

    def upload_captured_to_identity(
        self,
        slug: str,
        embeddings: list[np.ndarray],
        uploads: list[tuple[str, bytes]],
    ) -> EnrollmentResult:
        assert slug
        assert embeddings
        assert uploads
        if self.upload_error is not None:
            raise self.upload_error
        assert self.upload_result is not None
        return self.upload_result


@dataclass
class _StreamerStub:
    runtime: _RuntimeStub


class _ServerStub(BaseServer):
    streamer: _StreamerStub


class _TestLiveCameraHandler(LiveCameraHandler):
    def __init__(self) -> None:
        self.json_responses: list[tuple[dict[str, object], HTTPStatus]] = []

    def _send_json(self, data: object, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        assert isinstance(data, dict)
        self.json_responses.append((data, status))

    def handle_api_enroll_captures(self) -> None:
        self._handle_api_enroll_captures()


def _make_handler(
    payload: Mapping[str, object], runtime: _RuntimeStub
) -> tuple[_TestLiveCameraHandler, list[tuple[dict[str, object], HTTPStatus]]]:
    body = json.dumps(payload).encode("utf-8")
    handler = object.__new__(_TestLiveCameraHandler)
    _TestLiveCameraHandler.__init__(handler)
    headers = EmailMessage()
    headers["Content-Length"] = str(len(body))
    handler.headers = headers
    handler.rfile = io.BytesIO(body)
    server = object.__new__(_ServerStub)
    server.streamer = _StreamerStub(runtime=runtime)
    handler.server = server
    return handler, handler.json_responses


def _capture_payload(**extra: object) -> dict[str, object]:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload: dict[str, object] = {"captures": [base64.b64encode(encoded.tobytes()).decode("ascii")]}
    payload.update(extra)
    return payload


def test_enroll_captures_returns_not_found_for_unknown_identity() -> None:
    runtime = _RuntimeStub(upload_error=ValueError("Unknown identity: missing"))
    handler, responses = _make_handler(
        _capture_payload(name="Ignored", slug="missing"),
        runtime,
    )

    handler.handle_api_enroll_captures()

    assert responses == [({"error": "Unknown identity: missing"}, HTTPStatus.NOT_FOUND)]


def test_enroll_captures_update_returns_mode_and_accepted_count() -> None:
    runtime = _RuntimeStub(
        upload_result=EnrollmentResult(
            name="Andrii",
            slug="andrii",
            accepted_files=("capture_1.jpg", "capture_2.jpg"),
            rejected_files=(),
            sample_count=7,
        )
    )
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    capture_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    handler, responses = _make_handler(
        _capture_payload(
            slug="andrii",
            captures=[
                capture_b64,
                capture_b64,
            ],
        ),
        runtime,
    )

    handler.handle_api_enroll_captures()

    assert responses == [
        (
            {
                "ok": True,
                "mode": "update",
                "name": "Andrii",
                "slug": "andrii",
                "sample_count": 7,
                "accepted_count": 2,
                "accepted_files": ["capture_1.jpg", "capture_2.jpg"],
                "rejected_files": [],
            },
            HTTPStatus.OK,
        )
    ]


def test_enroll_captures_uses_capture_landmarks_when_available() -> None:
    runtime = _RuntimeStub(
        enroll_result=EnrollmentResult(
            name="Andrii",
            slug="andrii",
            accepted_files=("capture_1.jpg",),
            rejected_files=(),
            sample_count=1,
        )
    )
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = {
        "name": "Andrii",
        "captures": [
            {
                "frame_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "face_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "landmarks": [
                    [4.0, 5.0],
                    [11.0, 5.0],
                    [7.5, 8.0],
                    [5.0, 11.0],
                    [10.0, 11.0],
                ],
            }
        ],
    }
    handler, responses = _make_handler(payload, runtime)

    handler.handle_api_enroll_captures()

    assert responses == [
        (
            {
                "ok": True,
                "mode": "create",
                "name": "Andrii",
                "slug": "andrii",
                "sample_count": 1,
                "accepted_count": 1,
                "accepted_files": ["capture_1.jpg"],
                "rejected_files": [],
            },
            HTTPStatus.OK,
        )
    ]
    assert runtime.pipeline.detect_calls == 0
    assert runtime.pipeline.embed_calls == 1


def test_enroll_captures_rejects_capture_object_without_landmarks() -> None:
    runtime = _RuntimeStub(
        enroll_result=EnrollmentResult(
            name="Andrii",
            slug="andrii",
            accepted_files=("capture_1.jpg",),
            rejected_files=(),
            sample_count=1,
        )
    )
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = {
        "name": "Andrii",
        "captures": [
            {
                "frame_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "face_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
        ],
    }
    handler, responses = _make_handler(payload, runtime)

    handler.handle_api_enroll_captures()

    assert responses == [
        (
            {"error": "Capture 1 is missing landmarks"},
            HTTPStatus.BAD_REQUEST,
        )
    ]


def test_enroll_captures_create_accepts_null_slug() -> None:
    runtime = _RuntimeStub(
        enroll_result=EnrollmentResult(
            name="Andrii",
            slug="andrii",
            accepted_files=("capture_1.jpg",),
            rejected_files=(),
            sample_count=1,
        )
    )
    handler, responses = _make_handler(
        _capture_payload(name="Andrii", slug=None),
        runtime,
    )

    handler.handle_api_enroll_captures()

    assert responses == [
        (
            {
                "ok": True,
                "mode": "create",
                "name": "Andrii",
                "slug": "andrii",
                "sample_count": 1,
                "accepted_count": 1,
                "accepted_files": ["capture_1.jpg"],
                "rejected_files": [],
            },
            HTTPStatus.OK,
        )
    ]
