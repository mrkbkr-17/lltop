"""Postgres history. The round-trip tests need a throwaway database:

    LLTOP_TEST_PG_DSN=postgresql://user@host:5432/lltop_test uv run pytest

(they drop and recreate its `requests` table). Without it they are skipped; the
fail-open tests run anyway, against a Postgres that isn't there."""

import asyncio
import json
import os
import time

import httpx
import psycopg
import pytest

from lltop.app import create_app, merge_from_postgres
from lltop.hub import Hub
from lltop.store import PgStore, from_row, preview_of, to_row

pytestmark = pytest.mark.anyio

TEST_DSN = os.environ.get("LLTOP_TEST_PG_DSN")
needs_pg = pytest.mark.skipif(not TEST_DSN, reason="set LLTOP_TEST_PG_DSN to a throwaway database")
NOWHERE = "postgresql://nobody@127.0.0.1:1/none?connect_timeout=1"


def record(i, **over):
    t = 1_790_000_000.0 + i * 10
    rec = {"id": f"req{i:03d}", "ts": t, "first_token_ts": t + 1.5, "ended_ts": t + 4,
           "status": "finished", "model": "qwen3.5:9b", "endpoint": "/api/chat", "api": "chat",
           "client": "192.168.4.210:5000", "stream": True, "was_loaded": True,
           "messages": [{"role": "system", "content": "be brief"},
                        {"role": "user", "content": f"question number {i}"}],
           "tools": [], "options": {"temperature": 0}, "content": f"answer {i}",
           "thinking": "", "tool_calls": [], "chunks": 5,
           "stats": {"prompt_tokens": 10, "output_tokens": 5, "output_tokens_per_s": 3.0,
                     "ttft_s": 1.5, "wall_s": 4.0, "cold_load": False, "rate_source": "ollama"},
           "raw_stats": {"eval_count": 5, "done_reason": "stop"}}
    rec.update(over)
    return rec


def test_row_mapping_round_trip_without_a_database():
    rec = record(1, content="bad \x00 byte", messages=[{"role": "user", "content": "nul\x00"}])
    row = to_row(rec)
    assert row["content"] == "bad  byte"                       # Postgres can't store NUL
    assert row["messages"].obj == [{"role": "user", "content": "nul"}]
    assert row["preview"] == "nul" and row["done_reason"] == "stop"
    assert row["started_at"].timestamp() == rec["ts"]
    back = from_row({**row, "messages": row["messages"].obj, "tools": row["tools"].obj,
                     "options": row["options"].obj, "tool_calls": row["tool_calls"].obj,
                     "raw_stats": row["raw_stats"].obj})
    assert back["ts"] == rec["ts"] and back["ended_ts"] == rec["ended_ts"]
    assert back["stats"]["output_tokens_per_s"] == 3.0 and back["status"] == "finished"


def test_preview_is_the_last_user_message_or_the_prompt():
    assert preview_of(record(2)) == "question number 2"
    assert preview_of({"prompt": "x" * 1000})[:3] == "xxx" and len(preview_of({"prompt": "x" * 1000})) == 300
    assert preview_of({"messages": []}) == ""


# ------------------------------------------------------------------- fail open

async def test_record_never_blocks_and_counts_drops():
    store = PgStore(NOWHERE, queue_max=2)
    t = time.monotonic()
    for i in range(5):
        store.record(record(i))
    assert time.monotonic() - t < 0.05
    assert store.dropped == 3 and store.queue.qsize() == 2


async def test_postgres_down_doesnt_touch_requests():
    """Proxy a request with Postgres unreachable: the caller is unaffected and the row
    waits in the queue."""
    async def ollama(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body",
                    "body": json.dumps({"message": {"content": "hi"}, "done": True}).encode()})
    hub = Hub()
    app = create_app("http://ollama", transport=httpx.ASGITransport(app=ollama), hub=hub,
                     start_poller=False, pg_dsn=NOWHERE)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x") as c:
            r = await c.post("/api/chat", json={"model": "m", "messages": [], "stream": False})
            assert r.json()["message"]["content"] == "hi"
            await asyncio.sleep(1.5)                      # the writer tried and failed
            state = (await c.get("/lltop/api/state")).json()
            assert state["history"]["ok"] is False and state["history"]["queued"] == 1
            r = await c.get("/lltop/api/history")
            assert r.status_code == 503 and "error" in r.json()


async def test_history_endpoints_without_postgres_configured():
    app = create_app("http://ollama", start_poller=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://x") as c:
        r = await c.get("/lltop/api/history")
        assert r.status_code == 404 and r.json()["enabled"] is False
        assert (await c.get("/lltop/api/state")).json()["history"] == {"enabled": False}


# ----------------------------------------------------------- against Postgres

@pytest.fixture
async def store():
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS requests")
    s = PgStore(TEST_DSN)
    task = asyncio.create_task(s.run())
    yield s
    task.cancel()
    await s.close()


async def written(store, n, timeout=5):
    async with asyncio.timeout(timeout):
        while store.written < n:
            await asyncio.sleep(0.02)


@needs_pg
async def test_write_and_read_back(store):
    rec = record(1, content="Paris is the capital.\x00", thinking="hmm",
                 tool_calls=[{"function": {"name": "f", "arguments": {}}}])
    store.record(rec)
    await written(store, 1)
    got = await store.get("req001")
    assert got["content"] == "Paris is the capital." and got["thinking"] == "hmm"
    assert got["messages"][1]["content"] == "question number 1"
    assert got["tool_calls"][0]["function"]["name"] == "f"
    assert got["ts"] == rec["ts"] and got["first_token_ts"] == rec["first_token_ts"]
    assert got["stats"]["ttft_s"] == 1.5 and got["stats"]["cold_load"] is False
    # recording the same id again updates it
    store.record(record(1, status="error", error="boom"))
    await written(store, 2)
    assert (await store.get("req001"))["status"] == "error"
    assert store.status()["ok"] is True


@needs_pg
async def test_search_filters_and_paging(store):
    for i in range(7):
        store.record(record(i, model="fast" if i % 2 else "smart",
                            status="cancelled" if i == 3 else "finished",
                            content="the weather is sunny" if i == 5 else f"answer {i}"))
    await written(store, 7)
    page = await store.search(limit=3)
    assert [r["id"] for r in page["rows"]] == ["req006", "req005", "req004"]   # newest first
    assert "content" not in page["rows"][0] and page["rows"][0]["content_preview"]
    page2 = await store.search(limit=3, before=page["next"])
    assert [r["id"] for r in page2["rows"]] == ["req003", "req002", "req001"]
    assert [r["id"] for r in (await store.search(model="fast"))["rows"]] == ["req005", "req003", "req001"]
    assert [r["id"] for r in (await store.search(status="cancelled"))["rows"]] == ["req003"]
    assert [r["id"] for r in (await store.search(q="sunny"))["rows"]] == ["req005"]      # word
    assert [r["id"] for r in (await store.search(q="eath"))["rows"]] == ["req005"]       # substring
    assert [r["id"] for r in (await store.search(q="number 4"))["rows"]] == ["req004"]   # preview
    assert (await store.search(q="100%_"))["rows"] == []                                 # escaped
    facets = await store.facets()
    assert {f["value"]: f["count"] for f in facets["model"]} == {"smart": 4, "fast": 3}


@needs_pg
async def test_startup_merges_postgres_and_file(store):
    store.record(record(1))
    await written(store, 1)
    hub = Hub()
    hub.recent.extend([record(2), record(0)])        # from the history file, newest first
    await merge_from_postgres(store, hub)
    assert [r["id"] for r in hub.recent] == ["req002", "req001", "req000"]
    await written(store, 3)                          # the file-only ones were saved too
    assert (await store.get("req000")) is not None


@needs_pg
async def test_stats_for_the_charts(store):
    now = time.time()
    store.record(record(1, ts=now - 3600 * 2, model="smart", stats={"output_tokens": 10, "output_tokens_per_s": 2.0, "ttft_s": 30.0}))
    store.record(record(2, ts=now - 3600 * 2, model="smart", stats={"output_tokens": 20, "output_tokens_per_s": 4.0, "ttft_s": 10.0}))
    store.record(record(3, ts=now - 600, model="fast", status="cancelled", stats={"output_tokens": 0}))
    store.record(record(4, ts=now - 86400 * 3, model="old"))             # outside 24h
    await written(store, 4)
    d = await store.stats("24h", None, "America/New_York")
    assert d["unit"] == "hour" and len(d["buckets"]) == 24
    assert d["model_order"] == ["old", "smart", "fast"]                  # by first seen
    assert d["models"] == ["smart", "fast"]
    assert sum(d["series"]["smart"]["requests"]) == 2
    assert max(v for v in d["series"]["smart"]["tps"] if v is not None) == pytest.approx(3.0)   # median
    k = d["kpis"]
    assert (k["requests"], k["failed"], k["output_tokens"]) == (3, 1, 30)
    assert k["ttft"] == pytest.approx(20.0)                              # finished only
    assert d["clients"][0] == {"client": "192.168.4.210", "is_app": False, "requests": 3, "output_tokens": 30}
    store.record(record(5, ts=now - 60, model="smart", app="agent"))
    await written(store, 5)
    who = {c["client"]: c["is_app"] for c in (await store.stats("24h", None, "UTC"))["clients"]}
    assert who == {"192.168.4.210": False, "agent": True}
    assert [r["app"] for r in (await store.search(client="age"))["rows"]] == ["agent"]
    only = await store.stats("7d", "fast", "Not/AZone")                  # bad tz -> UTC
    assert only["tz"] == "UTC" and only["unit"] == "6h" and only["models"] == ["fast"]
    assert only["kpis"]["requests"] == 1
