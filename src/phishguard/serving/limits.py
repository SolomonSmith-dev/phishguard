"""Request body size limit for the public API.

Field-level ``max_length`` checks run only after the whole body has been read
and parsed, so on their own they do not stop a caller from sending a 100 MB
JSON body. This ASGI middleware rejects oversized bodies with 413 before they
are buffered: up front when ``Content-Length`` says so, and while streaming
for chunked bodies that carry no length.
"""

from __future__ import annotations

from typing import Any

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

Scope = dict[str, Any]
Message = dict[str, Any]


class _BodyTooLargeError(HTTPException):  # type: ignore[misc]
    """Raised from receive() mid-stream. Being an HTTPException, the app's own
    exception layer turns it into a 413 response; otherwise it would surface as
    a 500 from the app's server-error layer before reaching this middleware."""

    def __init__(self) -> None:
        super().__init__(status_code=413, detail="request body too large")


class BodySizeLimit:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, scope: Scope, receive: Any, send: Any) -> None:
        response = JSONResponse({"detail": "request body too large"}, status_code=413)
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = self.max_bytes + 1
                if declared > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message: Message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLargeError:
            if not started:
                await self._reject(scope, receive, send)
