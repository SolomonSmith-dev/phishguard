"""BodySizeLimit rejects oversized bodies with 413 before the app reads them."""

from __future__ import annotations

from collections.abc import Iterator

from phishguard.serving.limits import BodySizeLimit
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

LIMIT = 1024


async def echo_len(request: Request) -> JSONResponse:
    body = await request.body()
    return JSONResponse({"n": len(body)})


def _client() -> TestClient:
    app = Starlette(routes=[Route("/", echo_len, methods=["POST"])])
    return TestClient(BodySizeLimit(app, max_bytes=LIMIT))


def test_body_under_limit_passes() -> None:
    r = _client().post("/", content=b"a" * LIMIT)
    assert r.status_code == 200
    assert r.json() == {"n": LIMIT}


def test_declared_length_over_limit_rejected() -> None:
    r = _client().post("/", content=b"a" * (LIMIT + 1))
    assert r.status_code == 413


def test_chunked_body_over_limit_rejected() -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(4):
            yield b"a" * 400  # 1,600 bytes, no Content-Length

    r = _client().post("/", content=chunks())
    assert r.status_code == 413


def test_get_without_body_unaffected() -> None:
    app = Starlette(routes=[Route("/", lambda r: JSONResponse({"ok": True}))])
    r = TestClient(BodySizeLimit(app, max_bytes=LIMIT)).get("/")
    assert r.status_code == 200
