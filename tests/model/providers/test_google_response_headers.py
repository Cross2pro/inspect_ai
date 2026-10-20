"""The Google provider keeps the response headers of a model call on both backends.

On asyncio google-genai sends requests over aiohttp, which Inspect's httpx hooks
can't observe, so the provider records the headers the SDK hands back (#5842).
"""

import json
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import inspect_ai.model._model as model_module
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.model._response_headers import (
    ResponseHeaders,
    track_response_headers,
)

_REPLY = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "ok"}]},
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 1,
        "candidatesTokenCount": 1,
        "totalTokenCount": 2,
    },
    "modelVersion": "gemini-test",
    "responseId": "resp-1",
}


class _Server:
    """A local Gemini API that answers each request with the next scripted status."""

    def __init__(self, *statuses: int) -> None:
        self.statuses = list(statuses)
        self.request_ids: list[str | None] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", 0))
                self.rfile.read(length)
                server.request_ids.append(self.headers.get("x-irid"))
                attempt = len(server.request_ids)
                status = server.statuses.pop(0) if server.statuses else 200
                if status != 200:
                    body = json.dumps(
                        {"error": {"code": status, "message": "no", "status": "X"}}
                    ).encode()
                    content_type = "application/json"
                elif "alt=sse" in self.path:
                    body = f"data: {json.dumps(_REPLY)}\r\n\r\n".encode()
                    content_type = "text/event-stream"
                else:
                    body = json.dumps(_REPLY).encode()
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("X-Served-By", f"gateway-{attempt}")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def attempts(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[ResponseHeaders]]:
    """What the model layer keeps for each attempt, in order."""
    kept: list[ResponseHeaders] = []

    def tracking(headers: ResponseHeaders) -> AbstractContextManager[None]:
        kept.append(headers)
        return track_response_headers(headers)

    monkeypatch.setattr(model_module, "track_response_headers", tracking)
    yield kept


def _served_by(attempts: list[ResponseHeaders]) -> list[str | None]:
    return [(headers.latest or {}).get("x-served-by") for headers in attempts]


def _model(server: _Server, streaming: bool = False) -> Any:
    return get_model(
        "google/gemini-2.5-flash",
        api_key="key",
        base_url=server.base_url,
        memoize=False,
        streaming=streaming,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("streaming", [False, True], ids=["unary", "streaming"])
async def test_headers_of_a_reply_are_kept(
    attempts: list[ResponseHeaders], streaming: bool
) -> None:
    with _Server() as server:
        output = await _model(server, streaming).generate(
            "hello", config=GenerateConfig(max_retries=0)
        )

    assert output.completion == "ok"
    # the request carried Inspect's request id, so the headers are its own
    assert len(server.request_ids) == 1 and server.request_ids[0]
    assert _served_by(attempts) == ["gateway-1"]


@pytest.mark.anyio
@pytest.mark.parametrize("status", [400, 503])
async def test_headers_of_a_failed_call_are_kept(
    attempts: list[ResponseHeaders], status: int
) -> None:
    with _Server(*[status] * 10) as server:
        with pytest.raises(Exception):
            await _model(server).generate("hello", config=GenerateConfig(max_retries=0))

    # the latest response the call got (the SDK may retry a 503 itself)
    assert _served_by(attempts) == [f"gateway-{len(server.request_ids)}"]
