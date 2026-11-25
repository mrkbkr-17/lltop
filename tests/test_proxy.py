"""The proxy against a fake Ollama, in process (httpx's ASGI transport, no sockets).

Checks that callers get exactly the bytes Ollama sent, which events the dashboard
gets, and that the proxy fails open."""

import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from conftest import drain
from lltop.app import create_app
from lltop.hub import Hub
from lltop.tap import Tap

pytestmark = pytest.mark.anyio

FINAL = {"done": True, "done_reason": "stop", "prompt_eval_count": 12,
         "prompt_eval_duration": 2_000_000_000, "eval_count": 3,
         "eval_duration": 1_500_000_000, "load_duration": 5_000_000,
         "total_duration": 3_600_000_000}

CHAT = [{"model": "m:1b", "message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False},
        {"model": "m:1b", "message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"model": "m:1b", "message": {"role": "assistant", "content": "lo"}, "done": False},
        {"model": "m:1b", "message": {"role": "assistant", "content": ""}, **FINAL}]
CHAT_BYTES = b"".join(json.dumps(o).encode() + b"\n" for o in CHAT)

SSE = [{"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]},
       {"choices": [{"delta": {"content": " there"}, "finish_reason": None}]},
       {"choices": [{"delta": {}, "finish_reason": "stop"}]},
       {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}]
SSE_BYTES = b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in SSE) + b"data: [DONE]\n\n"


def fake_ollama() -> FastAPI:
    app = FastAPI()
    app.state.seen = []

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        app.state.seen.append(body)
        if body["model"] == "missing":
            return JSONResponse({"error": "model 'missing' not found"}, status_code=404)
        if body.get("stream") is False:
            whole = {"model": "m:1b", "message": {"role": "assistant", "content": "Hello"}, **FINAL}
            return JSONResponse(whole)

        async def gen():                 # several lines per network chunk, split mid-line
            yield CHAT_BYTES[:10]
            yield CHAT_BYTES[10:150]
            yield CHAT_BYTES[150:]
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/api/generate")
    async def generate(request: Request):
        await request.body()
        lines = [{"response": "4", "done": False}, {"response": "", **FINAL}]
        return Response(b"".join(json.dumps(o).encode() + b"\n" for o in lines),
                        media_type="application/x-ndjson")

    @app.post("/v1/chat/completions")
    async def openai(request: Request):
        await request.body()
        return Response(SSE_BYTES, media_type="text/event-stream")

    @app.get("/api/tags")
    async def tags(request: Request):
        return {"models": [{"name": "m:1b"}], "query": str(request.url.query),
                "header": request.headers.get("x-test")}

    @app.get("/")
    async def root():
        return Response("Ollama is running", media_type="text/plain")

    return app


@pytest.fixture
def setup():
    upstream = fake_ollama()
    hub = Hub()
    proxy_app = create_app("http://ollama", transport=httpx.ASGITransport(app=upstream),
                      hub=hub, start_poller=False)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_app, client=("10.0.0.7", 5555)),
                               base_url="http://lltop")
    return client, hub, hub.subscribe(), upstream


async def test_streaming_chat_bytes_and_events(setup):
    client, hub, sub, upstream = setup
    req = {"model": "m:1b", "messages": [{"role": "user", "content": "Say hello"}],
           "tools": [{"type": "function", "function": {"name": "get_time", "description": "Time"}}],
           "options": {"temperature": 0}}
    r = await client.post("/api/chat", json=req)
    assert r.status_code == 200
    assert r.content == CHAT_BYTES                       # byte for byte
    assert r.headers["content-type"] == "application/x-ndjson"
    assert upstream.state.seen[-1] == req                # Ollama got the request unchanged

    events = drain(sub)
    assert [e["type"] for e in events][0] == "started" and events[-1]["type"] == "finished"
    started, finished = events[0], events[-1]
    assert started["model"] == "m:1b" and started["client"] == "10.0.0.7:5555"
    assert started["messages"] == [{"role": "user", "content": "Say hello"}]
    assert started["tools"] == [{"name": "get_time", "description": "Time"}]
    assert started["options"] == {"temperature": 0} and started["stream"] is True
    tokens = [e for e in events if e["type"] == "token"]
    assert "".join(e["content"] for e in tokens) == "Hello"
    assert "".join(e["thinking"] for e in tokens) == "hmm"
    s = finished["stats"]
    assert s["output_tokens_per_s"] == pytest.approx(2.0)     # 3 tokens / 1.5 s
    assert s["prompt_tokens_per_s"] == pytest.approx(6.0)     # 12 / 2 s
    assert s["cold_load"] is False and s["done_reason"] == "stop"
    assert s["ttft_s"] is not None

    # the hub moved it from active to recent, with the whole text
    assert not hub.active
    assert hub.recent[0]["content"] == "Hello" and hub.recent[0]["status"] == "finished"


async def test_non_streaming_chat(setup):
    client, hub, sub, _ = setup
    r = await client.post("/api/chat", json={"model": "m:1b", "stream": False,
                                             "messages": [{"role": "user", "content": "hi"}]})
    assert r.json()["message"]["content"] == "Hello"
    events = drain(sub)
    assert [e["type"] for e in events] == ["started", "token", "finished"]
    assert events[0]["stream"] is False
    assert events[1]["content"] == "Hello"
    assert events[2]["stats"]["output_tokens"] == 3


async def test_generate_and_openai(setup):
    client, hub, sub, _ = setup
    r = await client.post("/api/generate", json={"model": "m:1b", "prompt": "2+2?", "images": ["AAAA"]})
    events = drain(sub)
    assert events[0]["prompt"] == "2+2?" and events[0]["images"] == 1
    assert events[-1]["type"] == "finished"

    r = await client.post("/v1/chat/completions", json={
        "model": "m:1b", "stream": True, "temperature": 0.2,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                                  {"type": "image_url", "image_url": {"url": "x"}}]}]})
    assert r.content == SSE_BYTES
    events = drain(sub)
    assert events[0]["messages"] == [{"role": "user", "content": "hi", "images": 1}]
    assert events[0]["options"] == {"temperature": 0.2}
    assert "".join(e.get("content", "") for e in events if e["type"] == "token") == "Hi there"
    s = events[-1]["stats"]
    assert (s["prompt_tokens"], s["output_tokens"], s["done_reason"]) == (9, 2, "stop")
    assert s["rate_source"] in ("chunks", None)


async def test_other_paths_pass_through_untouched(setup):
    client, hub, sub, _ = setup
    r = await client.get("/api/tags?x=1", headers={"x-test": "yes"})
    assert r.json() == {"models": [{"name": "m:1b"}], "query": "x=1", "header": "yes"}
    assert (await client.get("/")).text == "Ollama is running"
    assert drain(sub) == []                               # not inspected


async def test_ollama_error_is_returned_and_reported(setup):
    client, hub, sub, _ = setup
    r = await client.post("/api/chat", json={"model": "missing", "messages": []})
    assert r.status_code == 404
    assert r.json() == {"error": "model 'missing' not found"}
    events = drain(sub)
    assert events[-1]["type"] == "error"
    assert events[-1]["http_status"] == 404 and "not found" in events[-1]["error"]


async def test_fail_open_when_the_tap_breaks(setup, monkeypatch):
    client, hub, sub, _ = setup

    def broken(self, data):
        raise RuntimeError("tap bug")
    monkeypatch.setattr(Tap, "feed", broken)
    r = await client.post("/api/chat", json={"model": "m:1b", "messages": []})
    assert r.content == CHAT_BYTES                       # caller unaffected


async def test_tap_bug_ends_the_request_on_the_dashboard(setup, monkeypatch):
    client, hub, sub, _ = setup

    def broken(self, lines):                             # a bug inside the tap's parsing
        raise ValueError("parser bug")
    monkeypatch.setattr(Tap, "_lines", broken)
    r = await client.post("/api/chat", json={"model": "m:1b", "messages": []})
    assert r.content == CHAT_BYTES
    events = drain(sub)
    assert events[-1]["type"] == "error" and "not affected" in events[-1]["error"]
    assert not hub.active                                # not stuck as "running"


async def test_fail_open_when_the_hub_breaks(setup, monkeypatch):
    client, hub, sub, _ = setup

    def broken(event):
        raise RuntimeError("hub bug")
    monkeypatch.setattr(hub, "publish", broken)
    r = await client.post("/api/chat", json={"model": "m:1b", "messages": []})
    assert r.content == CHAT_BYTES


async def test_slow_browser_is_dropped_not_waited_for():
    hub = Hub(queue_max=2)
    slow = hub.subscribe()
    for i in range(5):
        hub.publish({"type": "token", "id": "x", "content": str(i)})
    assert slow.dropped and slow not in hub.subscribers


async def test_ollama_unreachable():
    def refuse(request):
        raise httpx.ConnectError("connection refused")
    hub = Hub()
    proxy_app = create_app("http://ollama", transport=httpx.MockTransport(refuse), hub=hub,
                      start_poller=False)
    sub = hub.subscribe()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy_app), base_url="http://lltop") as c:
        r = await c.post("/api/chat", json={"model": "m:1b", "messages": []})
        assert r.status_code == 502 and "could not reach Ollama" in r.json()["error"]
        r = await c.get("/api/tags")
        assert r.status_code == 502
    events = drain(sub)
    assert [e["type"] for e in events] == ["started", "error"]
