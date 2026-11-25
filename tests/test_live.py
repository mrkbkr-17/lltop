"""The proxy on real sockets: uvicorn in front, a fake Ollama (also uvicorn) behind.

The fake Ollama sends one chunk, then holds the rest until the test releases it.
That shows the proxy passes chunks on immediately (no buffering), and that a caller
hanging up closes the connection to Ollama (so Ollama would stop generating)."""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn

from conftest import drain
from lltop.app import create_app
from lltop.hub import Hub

pytestmark = pytest.mark.anyio

LINE1 = json.dumps({"message": {"content": "first"}, "done": False}).encode() + b"\n"
LINE2 = json.dumps({"message": {"content": ""}, "done": True, "eval_count": 1,
                    "eval_duration": 1}).encode() + b"\n"


class FakeOllama:
    """Raw ASGI app, so it controls exactly when headers and chunks go out.

    POST /api/chat      headers + LINE1 at once, then waits for `release`, then LINE2
    POST /api/generate  sends nothing (not even headers) until `release`, like a
                        non-streaming request while the model works
    Either one records 'disconnected' if the client goes away while it waits."""

    def __init__(self):
        self.release = asyncio.Event()
        self.disconnected = asyncio.Event()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        while True:                                  # read the whole request body
            msg = await receive()
            if not msg.get("more_body"):
                break
        headers = [(b"content-type", b"application/x-ndjson")]
        if scope["path"] == "/api/chat":
            await send({"type": "http.response.start", "status": 200, "headers": headers})
            await send({"type": "http.response.body", "body": LINE1, "more_body": True})
        if not await self._wait(receive):
            return
        if scope["path"] != "/api/chat":
            await send({"type": "http.response.start", "status": 200, "headers": headers})
            await send({"type": "http.response.body", "body": LINE1, "more_body": True})
        await send({"type": "http.response.body", "body": LINE2})

    async def _wait(self, receive) -> bool:
        released = asyncio.ensure_future(self.release.wait())
        gone = asyncio.ensure_future(receive())
        done, _ = await asyncio.wait({released, gone}, return_when=asyncio.FIRST_COMPLETED)
        if gone in done:
            released.cancel()
            self.disconnected.set()
            return False
        gone.cancel()
        return True


@asynccontextmanager
async def serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                           lifespan="auto"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


@asynccontextmanager
async def stack():
    fake, hub = FakeOllama(), Hub()
    async with serve(fake) as ollama_url:
        async with serve(create_app(ollama_url, hub=hub, start_poller=False)) as proxy_url:
            yield fake, hub, proxy_url


async def wait_for(event_type, sub, timeout=3.0) -> list[dict]:
    seen = []
    async with asyncio.timeout(timeout):
        while not any(e["type"] == event_type for e in seen):
            seen += drain(sub)
            await asyncio.sleep(0.02)
    return seen


async def test_chunks_are_passed_on_immediately():
    async with stack() as (fake, hub, url):
        sub = hub.subscribe()
        async with httpx.AsyncClient(base_url=url) as client:
            async with client.stream("POST", "/api/chat", json={"model": "m", "messages": []}) as r:
                chunks = r.aiter_raw()
                async with asyncio.timeout(3):        # LINE2 is still held back by Ollama
                    first = await anext(chunks)
                assert first == LINE1
                # the dashboard already has the token too
                events = await wait_for("token", sub)
                assert any(e.get("content") == "first" for e in events)
                fake.release.set()
                rest = b"".join([c async for c in chunks])
                assert rest == LINE2
        events = await wait_for("finished", sub)
        assert events[-1]["stats"]["ttft_s"] is not None


async def test_caller_hanging_up_mid_stream_stops_ollama():
    async with stack() as (fake, hub, url):
        sub = hub.subscribe()
        async with httpx.AsyncClient(base_url=url) as client:
            async with client.stream("POST", "/api/chat", json={"model": "m", "messages": []}) as r:
                assert await anext(r.aiter_raw()) == LINE1
            # leaving the block closes the connection mid-stream
        async with asyncio.timeout(3):
            await fake.disconnected.wait()            # Ollama saw its connection close
        events = await wait_for("cancelled", sub)
        assert events[-1]["stats"]["chunks"] == 1
        assert not hub.active


async def test_caller_hanging_up_before_the_first_byte_stops_ollama():
    async with stack() as (fake, hub, url):
        sub = hub.subscribe()
        async with httpx.AsyncClient(base_url=url, timeout=0.5) as client:
            with pytest.raises(httpx.ReadTimeout):    # gives up while Ollama "thinks"
                await client.post("/api/generate", json={"model": "m", "prompt": "hi", "stream": False})
        async with asyncio.timeout(3):
            await fake.disconnected.wait()
        await wait_for("cancelled", sub)
