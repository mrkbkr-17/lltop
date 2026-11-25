import os
import stat

import httpx
import pytest

from lltop.app import create_app
from lltop.history import History, save_now
from lltop.hub import Hub

pytestmark = pytest.mark.anyio


def finished(hub, i):
    hub.publish({"type": "started", "id": f"r{i}", "model": "m", "messages": []})
    hub.publish({"type": "token", "id": f"r{i}", "content": f"answer {i}"})
    hub.publish({"type": "finished", "id": f"r{i}", "stats": {"output_tokens": i}})


async def test_saved_history_comes_back(tmp_path):
    path = tmp_path / "state" / "history.json"
    hub, history = Hub(), History(path)
    for i in range(3):
        finished(hub, i)
    assert hub.dirty
    await save_now(history, hub)
    assert not hub.dirty
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600          # holds prompts
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700

    records = History(path).load()
    assert [r["id"] for r in records] == ["r2", "r1", "r0"]       # newest first
    assert records[0]["content"] == "answer 2" and records[0]["status"] == "finished"


async def test_app_loads_history_and_keeps_running_requests_on_shutdown(tmp_path):
    path = tmp_path / "history.json"
    hub = Hub()
    finished(hub, 1)
    History(path).save(list(hub.recent))

    hub2 = Hub()
    app = create_app("http://ollama", hub=hub2, start_poller=False, history_path=path)
    assert [r["id"] for r in hub2.recent] == ["r1"]
    hub2.publish({"type": "started", "id": "busy", "model": "m", "messages": []})
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x") as c:
            state = (await c.get("/lltop/api/state")).json()
            assert [r["id"] for r in state["recent"]] == ["r1"]
    # lifespan ended: the running request was saved as interrupted
    records = History(path).load()
    assert [(r["id"], r["status"]) for r in records] == [("busy", "interrupted"), ("r1", "finished")]


def test_missing_or_broken_file_means_empty(tmp_path):
    assert History(tmp_path / "nope.json").load() == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert History(bad).load() == []


def test_recent_is_capped_when_loading(tmp_path):
    path = tmp_path / "h.json"
    History(path).save([{"id": f"r{i}", "status": "finished"} for i in range(80)])
    hub = Hub()
    create_app("http://ollama", hub=hub, start_poller=False, history_path=path)
    assert len(hub.recent) == 50 and hub.recent[0]["id"] == "r0"
