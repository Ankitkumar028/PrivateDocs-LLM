"""Lightweight, standard-library OpenAI-compatible mock LLM HTTP server for tests."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockLLMHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible /v1/chat/completions and /v1/models requests."""

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        # Suppress standard logging during tests
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models" or self.path == "/models":
            response = {
                "object": "list",
                "data": [
                    {
                        "id": "mock-model",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "mock",
                    }
                ],
            }
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health" or self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            body = b'{"status": "healthy"}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/chat/completions" or self.path == "/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(request_body) if request_body else {}
            except json.JSONDecodeError:
                data = {}

            stream = data.get("stream", False)
            model = data.get("model", "mock-model")
            messages = data.get("messages", [])

            # Extract user query
            user_prompt = ""
            for m in messages:
                if m.get("role") == "user":
                    user_prompt = m.get("content", "")

            # Generate mock reply
            mock_reply = "Based on [test_doc, p.1], the system is fully operational and meets all requirements."
            if "fail" in user_prompt.lower():
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                body = b'{"error": {"message": "Simulated LLM error"}}'
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                tokens = mock_reply.split(" ")
                for i, tok in enumerate(tokens):
                    chunk_text = tok + (" " if i < len(tokens) - 1 else "")
                    chunk_data = {
                        "id": "chatcmpl-mock-stream",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk_text},
                                "finish_reason": None if i < len(tokens) - 1 else "stop",
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk_data)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                response = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": mock_reply,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(user_prompt.split()) + 10,
                        "completion_tokens": len(mock_reply.split()),
                        "total_tokens": len(user_prompt.split()) + len(mock_reply.split()) + 10,
                    },
                }
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def start_mock_server_thread() -> tuple[ThreadingHTTPServer, int]:
    """Start mock LLM server on an ephemeral port in a daemon thread."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockLLMHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


if __name__ == "__main__":
    server, port = start_mock_server_thread()
    print(f"Mock LLM server running on http://127.0.0.1:{port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.shutdown()
