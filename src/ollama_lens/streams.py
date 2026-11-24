"""Understanding the bytes that come back from Ollama.

Network chunks don't line up with messages: one chunk can hold half a line, or three
lines. LineSplitter reassembles whole lines. Then each line is decoded per API:

  /api/chat, /api/generate   NDJSON: one JSON object per line, the last has "done": true
                             with the stats. Non-streaming: one JSON object.
  /v1/chat/completions       Server-Sent Events: lines "data: {json}", ending with
                             "data: [DONE]". Non-streaming: one JSON object.

Every decoder returns a Delta: the new text, thinking and tool calls in that piece,
plus the final stats or an error if it carried them.
"""

import json
from dataclasses import dataclass, field

from .stats import STAT_KEYS

MAX_LINE = 16 * 1024 * 1024  # a line longer than this isn't a chat chunk; stop parsing


class LineTooLong(Exception):
    pass


class LineSplitter:
    """Feed it bytes in whatever pieces they arrive; get back complete lines."""

    def __init__(self, max_line: int = MAX_LINE):
        self.buf = b""
        self.max_line = max_line

    def feed(self, data: bytes) -> list[bytes]:
        self.buf += data
        *lines, self.buf = self.buf.split(b"\n")
        if len(self.buf) > self.max_line:
            raise LineTooLong(f"line over {self.max_line} bytes")
        return [line.rstrip(b"\r") for line in lines if line.strip()]

    def flush(self) -> list[bytes]:
        """The last line, if the stream didn't end with a newline."""
        rest, self.buf = self.buf.strip(), b""
        return [rest] if rest else []


@dataclass
class Delta:
    content: str = ""
    thinking: str = ""
    tool_calls: list = field(default_factory=list)
    final: dict | None = None      # Ollama stats (STAT_KEYS) + done_reason, when done
    error: str | None = None

    @property
    def has_output(self) -> bool:
        return bool(self.content or self.thinking or self.tool_calls)


def _error_text(err) -> str:
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err)


def ollama_delta(obj: dict, kind: str) -> Delta:
    """One NDJSON object from /api/chat (kind 'chat') or /api/generate ('generate')."""
    d = Delta()
    if not isinstance(obj, dict):
        return d
    if "error" in obj:
        d.error = _error_text(obj["error"])
    if kind == "chat":
        msg = obj.get("message") or {}
        d.content = msg.get("content") or ""
        d.thinking = msg.get("thinking") or ""
        d.tool_calls = msg.get("tool_calls") or []
    else:
        d.content = obj.get("response") or ""
        d.thinking = obj.get("thinking") or ""
    if obj.get("done"):
        d.final = {k: obj.get(k) for k in STAT_KEYS if obj.get(k) is not None}
        d.final["done_reason"] = obj.get("done_reason")
    return d


def openai_delta(obj: dict) -> Delta:
    """One /v1/chat/completions object: a streamed chunk ("delta") or a whole reply
    ("message"). Ollama puts thinking in "reasoning". Token counts come from "usage"
    (streaming only sends it with stream_options.include_usage)."""
    d = Delta()
    if not isinstance(obj, dict):
        return d
    if "error" in obj:
        d.error = _error_text(obj["error"])
    choices = obj.get("choices") or []
    finish = None
    if choices and isinstance(choices[0], dict):
        c = choices[0]
        part = c.get("delta") or c.get("message") or {}
        d.content = part.get("content") or ""
        d.thinking = part.get("reasoning") or part.get("reasoning_content") or ""
        d.tool_calls = part.get("tool_calls") or []
        finish = c.get("finish_reason")
    usage = obj.get("usage")
    if usage or finish:
        d.final = {"done_reason": finish}
        if usage:
            d.final["prompt_eval_count"] = usage.get("prompt_tokens")
            d.final["eval_count"] = usage.get("completion_tokens")
    return d


def sse_data(line: bytes) -> dict | None | str:
    """The JSON payload of an SSE line; 'DONE' for the end marker; None for other
    lines (comments, event: fields)."""
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return "DONE"
    return json.loads(payload)


def merge_final(a: dict | None, b: dict | None) -> dict | None:
    """Combine stats from several pieces (OpenAI sends finish_reason and usage in
    separate chunks); later non-empty values win."""
    if not b:
        return a
    out = dict(a or {})
    out.update({k: v for k, v in b.items() if v is not None})
    return out
