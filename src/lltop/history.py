"""Keeps the last finished requests in a JSON file, so a restart doesn't lose them.

The file holds prompts and responses, so it is created readable by its owner only
(directory 0700, file 0600). It is rewritten whole (tmp file + rename, so a crash can't
leave half a file) at most every SAVE_EVERY seconds after something finished, and once
more on shutdown. A missing or unreadable file just means an empty history.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("lltop")

SAVE_EVERY = 2.0
VERSION = 1


def default_path() -> Path:
    state = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(state) / "lltop" / "history.json"


class History:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.failing = False

    def load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            records = data.get("recent", []) if isinstance(data, dict) else []
            return [r for r in records if isinstance(r, dict) and r.get("id")]
        except FileNotFoundError:
            return []
        except Exception as e:
            log.warning("history: could not read %s (%s); starting empty", self.path, e)
            return []

    def save(self, records: list[dict]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"version": VERSION, "recent": records}, f, ensure_ascii=False)
        os.replace(tmp, self.path)


async def save_forever(history: History, hub, every: float = SAVE_EVERY) -> None:
    """Background task: save hub.recent whenever it changed. Never raises."""
    while True:
        await asyncio.sleep(every)
        await save_now(history, hub)


async def save_now(history: History, hub) -> None:
    if not hub.dirty:
        return
    hub.dirty = False
    try:
        await asyncio.to_thread(history.save, list(hub.recent))
        if history.failing:
            log.info("history: saving to %s works again", history.path)
        history.failing = False
    except Exception:
        hub.dirty = True                      # try again next time
        if not history.failing:               # log once, not every few seconds
            log.exception("history: could not save to %s", history.path)
        history.failing = True
