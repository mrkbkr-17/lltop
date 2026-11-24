"""The hub: live state for the dashboard, and fan-out of events to browsers.

Everything runs on one asyncio event loop, so no locks are needed. publish() never
blocks and never raises: each browser has its own bounded queue, and a browser that
falls too far behind is dropped (it reconnects and gets a fresh snapshot). A slow or
broken dashboard can therefore never slow down a request going to Ollama.

State is in memory only (milestone 1): loaded models from /api/ps, requests in
progress (with the text generated so far, so a browser that opens mid-request sees
it all), and the last RECENT_MAX finished requests.
"""

import asyncio
import logging
import time
from collections import deque

log = logging.getLogger("ollama_lens")

RECENT_MAX = 50
QUEUE_MAX = 5000            # events buffered per browser before it is dropped
MAX_TEXT = 1_000_000        # characters of response/thinking kept per request


def model_key(name: str | None) -> str:
    """'qwen3.5' and 'qwen3.5:latest' are the same model to Ollama."""
    if not name:
        return ""
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


class Subscriber:
    def __init__(self, maxsize: int = QUEUE_MAX):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize)
        self.dropped = False


class Hub:
    def __init__(self, recent_max: int = RECENT_MAX, queue_max: int = QUEUE_MAX):
        self.models: list[dict] = []
        self.ps_ok: bool | None = None       # None until the first /api/ps poll
        self.ps_error: str | None = None
        self.active: dict[str, dict] = {}
        self.recent: deque[dict] = deque(maxlen=recent_max)
        self.subscribers: set[Subscriber] = set()
        self.queue_max = queue_max

    # ------------------------------------------------------------------ browsers

    def subscribe(self) -> Subscriber:
        sub = Subscriber(self.queue_max)
        self.subscribers.add(sub)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        self.subscribers.discard(sub)

    def snapshot(self) -> dict:
        return {"type": "snapshot", "ts": time.time(), "models": self.models,
                "ps_ok": self.ps_ok, "ps_error": self.ps_error,
                "active": list(self.active.values()), "recent": list(self.recent)}

    # ------------------------------------------------------------------ events

    def publish(self, event: dict) -> None:
        """Record an event and queue it for every browser. Never raises."""
        try:
            event.setdefault("ts", time.time())
            self._apply(event)
        except Exception:
            log.exception("hub: could not apply %s event", event.get("type"))
        for sub in list(self.subscribers):
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                sub.dropped = True           # its websocket loop closes it
                self.subscribers.discard(sub)
            except Exception:
                log.exception("hub: could not queue event")

    def _apply(self, ev: dict) -> None:
        kind = ev["type"]
        if kind == "started":
            self.active[ev["id"]] = {**ev, "content": "", "thinking": "", "tool_calls": [],
                                     "chunks": 0, "first_token_ts": None, "status": "running"}
        elif kind == "token":
            req = self.active.get(ev["id"])
            if req is None:
                return
            for field in ("content", "thinking"):
                if ev.get(field) and len(req[field]) < MAX_TEXT:
                    req[field] += ev[field]
            req["tool_calls"].extend(ev.get("tool_calls") or [])
            req["chunks"] += 1
            if req["first_token_ts"] is None:
                req["first_token_ts"] = ev["ts"]
        elif kind in ("finished", "error", "cancelled"):
            req = self.active.pop(ev["id"], None)
            if req is None:
                return
            req.update({k: v for k, v in ev.items() if k not in ("type", "ts")})
            req["status"] = kind
            req["ended_ts"] = ev["ts"]
            self.recent.appendleft(req)
        elif kind == "models":
            self.models = ev["models"]
            self.ps_ok = ev.get("ok")
            self.ps_error = ev.get("error")

    # ------------------------------------------------------------------ helpers

    def model_loaded(self, name: str | None) -> bool | None:
        """Is the model in the last /api/ps answer? None if we don't know yet."""
        if not self.ps_ok:
            return None
        key = model_key(name)
        return any(model_key(m.get("name")) == key for m in self.models)
