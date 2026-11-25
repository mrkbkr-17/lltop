"""The tap: watches one inspected request and turns it into dashboard events.

The proxy hands it the request body, then every response chunk exactly as it was sent
to the caller, then tells it how the request ended. The tap emits:

    started    id, model, endpoint, client, messages/prompt, tools, options
    token      new content / thinking / tool calls (one event per network chunk)
    finished   Ollama's final stats + computed speed (stats.summarize)
    error      HTTP error from Ollama, error object in the stream, or upstream failure
    cancelled  the caller disconnected before the end

Fail open: every public method catches its own exceptions. After the first failure
the tap logs it once and goes quiet; the request itself carries on untouched.
"""

import functools
import json
import logging
import time
import uuid

from . import stats
from .streams import LineSplitter, merge_final, ollama_delta, openai_delta, sse_data

log = logging.getLogger("lltop")

MAX_FIELD = 100_000      # characters kept per prompt/message in the started event
MAX_BODY = 16 * 1024 * 1024

ENDPOINTS = {"/api/chat": "chat", "/api/generate": "generate",
             "/v1/chat/completions": "openai"}

OPENAI_OPTIONS = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed",
                  "stop", "frequency_penalty", "presence_penalty", "response_format",
                  "reasoning_effort", "stream_options")


def safe(method):
    """Run a tap method; on any error, log once and disable the tap."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.broken:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            self.broken = True
            log.warning("tap %s: stopped watching this request (%s: %s)",
                        self.id, type(e).__name__, e, exc_info=True)
            if not self.done:           # so the dashboard doesn't show it running forever
                self.done = True
                try:
                    self._emit("error", error=f"lltop stopped following this request "
                                              f"({type(e).__name__}); the caller was not affected")
                except Exception:
                    pass
            return None
    return wrapper


def _clip(text, limit=MAX_FIELD):
    if not isinstance(text, str):
        return text
    return text if len(text) <= limit else text[:limit] + f"… [{len(text) - limit:,} more chars]"


def _content_text(content):
    """OpenAI content may be a list of parts ({type: text|image_url, ...})."""
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        images = sum(1 for p in content if isinstance(p, dict) and p.get("type") == "image_url")
        return "\n".join(texts), images
    return content, 0


def describe_request(kind: str, body: dict) -> dict:
    """The parts of a request body the dashboard shows. Images become a count."""
    out = {"model": body.get("model")}
    if kind == "generate":
        out["prompt"] = _clip(body.get("prompt"))
        out["system"] = _clip(body.get("system"))
        out["images"] = len(body.get("images") or [])
    else:
        msgs = []
        for m in body.get("messages") or []:
            if not isinstance(m, dict):
                continue
            text, images = _content_text(m.get("content"))
            msg = {"role": m.get("role"), "content": _clip(text)}
            images += len(m.get("images") or [])
            if images:
                msg["images"] = images
            for k in ("thinking", "tool_calls", "tool_name", "tool_call_id", "name"):
                if m.get(k):
                    msg[k] = m[k]
            msgs.append(msg)
        out["messages"] = msgs
    tools = []
    for t in body.get("tools") or []:
        f = t.get("function", t) if isinstance(t, dict) else {}
        tools.append({"name": f.get("name"), "description": _clip(f.get("description"), 500)})
    out["tools"] = tools
    if kind == "openai":
        out["options"] = {k: body[k] for k in OPENAI_OPTIONS if k in body}
        out["stream"] = bool(body.get("stream", False))
    else:
        out["options"] = body.get("options") or {}
        out["stream"] = body.get("stream", True) is not False
        for k in ("think", "format", "keep_alive"):
            if k in body:
                out["options"][k] = body[k]
    return out


class Tap:
    def __init__(self, hub, path: str, client: str | None, clock=time.monotonic,
                 app: str | None = None):
        self.hub = hub
        self.app = app
        self.kind = ENDPOINTS[path]
        self.path = path
        self.client = client
        self.clock = clock
        self.id = uuid.uuid4().hex[:12]
        self.broken = False
        self.done = False           # an end event (finished/error/cancelled) was sent
        self.started = clock()
        self.first_token = None
        self.chunks = 0
        self.final = None
        self.model = None
        self.was_loaded = None
        self.mode = None            # 'ndjson' | 'sse' | 'json' | 'error' | 'skip'
        self.status = None
        self.lines = LineSplitter()
        self.body = bytearray()
        self.stream_error = None

    def _emit(self, type_, **fields):
        self.hub.publish({"type": type_, "id": self.id, **fields})

    # ------------------------------------------------------------- request side

    @safe
    def request(self, body: bytes) -> None:
        """The request body, before it is sent to Ollama: emits 'started'."""
        try:
            data = json.loads(body) if body else {}
        except ValueError:
            data = {}
        info = describe_request(self.kind, data if isinstance(data, dict) else {})
        self.model = info["model"]
        self.was_loaded = self.hub.model_loaded(self.model)
        self._emit("started", endpoint=self.path, api=self.kind, client=self.client, app=self.app,
                   was_loaded=self.was_loaded, **info)

    # ------------------------------------------------------------ response side

    @safe
    def response(self, status: int, headers) -> None:
        """Ollama's status and headers: decides how to read the body."""
        self.status = status
        ctype = headers.get("content-type", "")
        if headers.get("content-encoding", "identity") not in ("", "identity"):
            self.mode = "skip"              # compressed: pass through, don't parse
        elif status >= 400:
            self.mode = "error"
        elif "text/event-stream" in ctype:
            self.mode = "sse"
        elif "ndjson" in ctype:
            self.mode = "ndjson"
        else:
            self.mode = "json"

    @safe
    def feed(self, data: bytes) -> None:
        """One response chunk, as the caller received it."""
        if self.mode in ("json", "error"):
            if len(self.body) + len(data) > MAX_BODY:
                self.mode = "skip"
                self.body.clear()
                return
            self.body += data
        elif self.mode in ("ndjson", "sse"):
            self._lines(self.lines.feed(data))

    def _lines(self, lines) -> None:
        content, thinking, calls = [], [], []
        for line in lines:
            if self.mode == "sse":
                obj = sse_data(line)
                if obj is None or obj == "DONE":
                    continue
                d = openai_delta(obj)
            else:
                d = ollama_delta(json.loads(line), self.kind)
            content.append(d.content)
            thinking.append(d.thinking)
            calls.extend(d.tool_calls)
            self.final = merge_final(self.final, d.final)
            if d.error:
                self.stream_error = d.error
        self._token("".join(content), "".join(thinking), calls)

    def _token(self, content: str, thinking: str, calls: list) -> None:
        if not (content or thinking or calls):
            return
        if self.first_token is None:
            self.first_token = self.clock()
        self.chunks += 1
        ev = {"content": content, "thinking": thinking}
        if calls:
            ev["tool_calls"] = calls
        self._emit("token", **ev)

    # ---------------------------------------------------------------- endings

    @safe
    def finish(self) -> None:
        """Ollama sent the whole response."""
        if self.done:
            return
        if self.mode in ("ndjson", "sse"):
            self._lines(self.lines.flush())
        elif self.mode in ("json", "error") and self.body:
            obj = json.loads(bytes(self.body))
            d = openai_delta(obj) if self.kind == "openai" else ollama_delta(obj, self.kind)
            self.final = merge_final(self.final, d.final)
            if d.error:
                self.stream_error = d.error
            if self.status < 400:
                self._token(d.content, d.thinking, d.tool_calls)
        self.done = True
        summary = stats.summarize(self.final, started=self.started, first_token=self.first_token,
                                  ended=self.clock(), chunks=self.chunks,
                                  was_loaded=self.was_loaded)
        summary["done_reason"] = (self.final or {}).get("done_reason")
        if self.status and self.status >= 400 or self.stream_error:
            self._emit("error", http_status=self.status,
                       error=self.stream_error or f"HTTP {self.status}", stats=summary)
        else:
            self._emit("finished", http_status=self.status, stats=summary,
                       raw_stats=self.final or {})

    @safe
    def cancel(self) -> None:
        """The caller went away before the end."""
        if self.done:
            return
        self.done = True
        self._emit("cancelled", stats=self._partial())

    @safe
    def fail(self, message: str) -> None:
        """Ollama couldn't be reached, or the connection broke mid-response."""
        if self.done:
            return
        self.done = True
        self._emit("error", http_status=self.status, error=message, stats=self._partial())

    def _partial(self) -> dict:
        return stats.summarize(self.final, started=self.started, first_token=self.first_token,
                               ended=self.clock(), chunks=self.chunks, was_loaded=self.was_loaded)
