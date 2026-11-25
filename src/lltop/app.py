"""The FastAPI app: the dashboard under /lltop/, everything else forwarded to Ollama.

    /lltop/            the dashboard page
    /lltop/ws          WebSocket: a snapshot on connect, then every event live
    /lltop/api/state   the same snapshot as JSON (handy with curl)
    anything else     forwarded to Ollama unchanged (proxy.forward)
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from .hub import Hub
from .proxy import forward
from .ps import poll_forever

log = logging.getLogger("lltop")

STATIC = Path(__file__).parent / "static"
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def create_app(upstream: str = "http://127.0.0.1:11434", *, poll_interval: float = 3.0,
               transport: httpx.AsyncBaseTransport | None = None, hub: Hub | None = None,
               start_poller: bool = True) -> FastAPI:
    hub = hub or Hub()
    # No read timeout: on a slow CPU a model can think for minutes between chunks.
    client = httpx.AsyncClient(
        base_url=upstream, transport=transport,
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None))

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(poll_forever(client, hub, poll_interval)) if start_poller else None
        log.info("lltop: forwarding to %s; dashboard at /lltop/", upstream)
        yield
        if task:
            task.cancel()
        await client.aclose()

    # No /docs or /openapi.json: those paths belong to Ollama (forwarded like the rest).
    app = FastAPI(title="lltop", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.hub = hub
    app.state.client = client

    @app.get("/lltop", include_in_schema=False)
    async def dashboard_redirect():
        return RedirectResponse("/lltop/")

    @app.get("/lltop/", include_in_schema=False)
    async def dashboard():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/lltop/api/state", include_in_schema=False)
    async def state():
        return JSONResponse({**hub.snapshot(), "upstream": upstream})

    @app.websocket("/lltop/ws")
    async def events(ws: WebSocket):
        await ws.accept()
        sub = hub.subscribe()

        async def send():
            await ws.send_json({**hub.snapshot(), "upstream": upstream})
            while True:
                event = await sub.queue.get()
                if sub.dropped:      # we fell behind and missed events: start over
                    await ws.close(code=1013, reason="too slow; reconnect for a fresh snapshot")
                    return
                await ws.send_json(event)

        async def receive():         # notices the browser closing the page
            while True:
                await ws.receive_text()

        tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except Exception:
            pass
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    pass
                except Exception:
                    log.debug("websocket task ended with an error", exc_info=True)
            hub.unsubscribe(sub)

    @app.api_route("/{path:path}", methods=METHODS, include_in_schema=False)
    async def proxy(request: Request):
        return await forward(request, client, hub)

    return app
