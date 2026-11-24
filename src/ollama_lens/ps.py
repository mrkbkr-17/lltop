"""Polls Ollama's /api/ps (the models currently in memory) every few seconds.

For each loaded model Ollama reports its total size and how much of it is in GPU
memory (size_vram); the rest is in system RAM, run by the CPU. expires_at is when
Ollama will unload it (keep_alive, reset by every request). A 'models' event goes to
the hub whenever anything changes (or polling starts or stops failing).
"""

import asyncio
import json
import logging

log = logging.getLogger("ollama_lens")


def model_info(m: dict) -> dict:
    size = m.get("size") or 0
    vram = m.get("size_vram") or 0
    details = m.get("details") or {}
    return {
        "name": m.get("name") or m.get("model"),
        "size": size,
        "size_vram": vram,
        "size_cpu": max(size - vram, 0),
        "vram_pct": round(100 * vram / size) if size else None,
        "expires_at": m.get("expires_at"),
        "context_length": m.get("context_length"),
        "parameter_size": details.get("parameter_size"),
        "quantization": details.get("quantization_level"),
        "family": details.get("family"),
    }


async def poll_once(client, hub, last: str | None) -> str:
    """One poll; publishes a 'models' event if anything changed. Returns the new state key."""
    try:
        r = await client.get("/api/ps", timeout=5.0)
        r.raise_for_status()
        models = sorted((model_info(m) for m in r.json().get("models") or []),
                        key=lambda m: m["name"] or "")
        event = {"type": "models", "ok": True, "error": None, "models": models}
    except Exception as e:
        event = {"type": "models", "ok": False, "error": f"{type(e).__name__}: {e}",
                 "models": hub.models}
    key = json.dumps(event, sort_keys=True, default=str)
    if key != last:
        if not event["ok"]:
            log.warning("/api/ps poll failed: %s", event["error"])
        hub.publish(event)
    return key


async def poll_forever(client, hub, interval: float) -> None:
    last = None
    while True:
        try:
            last = await poll_once(client, hub, last)
        except Exception:
            log.exception("/api/ps poller error")
        await asyncio.sleep(interval)
