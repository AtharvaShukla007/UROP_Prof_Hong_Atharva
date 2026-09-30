"""A minimal, real HTTP server implementing enough of the OpenAI
chat-completions shape to integration-test the REAL
external/autotracegt_upstream OpenCodingAgent against it -- not a mock of
our own code, an actual server a real `openai.OpenAI(base_url=...)` client
talks to over a real socket.

Not a pytest fixture module (kept plain so it can be imported without a
conftest.py); tests construct `FakeLocalChatServer` directly.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLocalChatServer:
    """Usage:

        server = FakeLocalChatServer(responses=["===CODES===\\n[]\\n===END==="])
        with server:
            register_local_endpoint(base_url=server.base_url)
            ...
        server.requests  # list of received {"messages": [...], ...} bodies
    """

    def __init__(
        self,
        *,
        responses: list[str] | None = None,
        response_fn=None,
        delay_seconds: float = 0.0,
        finish_reason: str = "stop",
    ):
        """responses: content strings returned in order, one per request
        (the last one repeats if there are more requests than responses).
        response_fn(request_index, body) -> str: takes priority over `responses`
        if given, for tests that need to inspect the request to decide the reply.
        delay_seconds: sleep before responding, to exercise client-side timeouts.
        finish_reason: the finish_reason every response reports (default
        "stop"); set to "length" to simulate a max-tokens cutoff.
        """
        self.responses = responses or ['===CODES===\n[]\n===SKIPPED===\n===SEGMENT_MEMO===\n===END===']
        self.response_fn = response_fn
        self.finish_reason = finish_reason
        self.delay_seconds = delay_seconds
        self.requests: list[dict] = []
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://127.0.0.1:{port}/v1"

    def __enter__(self) -> "FakeLocalChatServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: D401 -- silence default stderr logging
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    body = {"_raw": raw.decode("utf-8", errors="replace")}
                idx = len(server.requests)
                server.requests.append(body)

                if server.delay_seconds:
                    time.sleep(server.delay_seconds)

                if server.response_fn is not None:
                    content = server.response_fn(idx, body)
                else:
                    content = server.responses[min(idx, len(server.responses) - 1)]

                payload = {
                    "id": f"fake-{idx}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", "fake-model"),
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": server.finish_reason}
                    ],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
