"""The FastAPI app: the dashboard under /lltop/, everything else forwarded to Ollama.

    /lltop/            the dashboard page
    /lltop/ws          WebSocket: a snapshot on connect, then every event live
    /lltop/api/state   the same snapshot as JSON (handy with curl)
    /lltop/api/history         Postgres history: ?model= &status= &endpoint= &client= &q=
                               &before= (the `next` of the previous page) &limit=
    /lltop/api/history/{id}    one request in full
    /lltop/api/history-facets  models, endpoints, statuses with counts (for the filters)
    anything else     forwarded to Ollama unchanged (proxy.forward)
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from .history import History, save_forever, save_now
from .store import PgStore
from .hub import Hub
from .proxy import forward
from .ps import poll_forever

log = logging.getLogger("lltop")

STATIC = Path(__file__).parent / "static"
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


async def merge_from_postgres(store: PgStore, hub: Hub) -> None:
    """Start with the newest requests from Postgres and the history file together; save
    file-only requests (e.g. from before Postgres was set up) to Postgres. If Postgres
    doesn't answer within a few seconds, keep the file's and carry on."""
    try:
        async with asyncio.timeout(8):
            rows = await store.recent(hub.recent.maxlen)
    except Exception as e:
        log.warning("postgres history: could not load recent requests (%s: %s)", type(e).__name__, e)
        return
    in_pg = {r["id"] for r in rows}
    file_only = [r for r in hub.recent if r.get("id") not in in_pg]
    for rec in file_only:
        store.record(rec)
    merged = sorted(rows + file_only, key=lambda r: r.get("ts") or 0, reverse=True)
    hub.recent.clear()
    hub.recent.extend(merged[:hub.recent.maxlen])
    log.info("postgres history: %d recent requests loaded, %d added from the history file",
             len(rows), len(file_only))


def create_app(upstream: str = "http://127.0.0.1:11434", *, poll_interval: float = 3.0,
               transport: httpx.AsyncBaseTransport | None = None, hub: Hub | None = None,
               start_poller: bool = True, history_path: str | Path | None = None,
               pg_dsn: str | None = None) -> FastAPI:
    hub = hub or Hub()
    history = History(history_path) if history_path else None
    if history:
        # newest first, like hub.recent; keep the newest if the file holds more than fits
        hub.recent.extend(history.load()[:hub.recent.maxlen])
    store = PgStore(pg_dsn) if pg_dsn else None
    if store:
        hub.sink = store.record
    # No read timeout: on a slow CPU a model can think for minutes between chunks.
    client = httpx.AsyncClient(
        base_url=upstream, transport=transport,
        timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None))

    @asynccontextmanager
    async def lifespan(app):
        tasks = []
        if start_poller:
            tasks.append(asyncio.create_task(poll_forever(client, hub, poll_interval)))
        if history:
            tasks.append(asyncio.create_task(save_forever(history, hub)))
            log.info("history in %s (%d requests loaded)", history.path, len(hub.recent))
        if store:
            tasks.append(asyncio.create_task(store.run()))
            await merge_from_postgres(store, hub)
        log.info("forwarding to %s; dashboard at /lltop/", upstream)
        yield
        for task in tasks:
            task.cancel()
        hub.interrupt_active()             # running requests are kept as 'interrupted'
        if history:
            await save_now(history, hub)
        if store:
            await store.flush()
            await store.close()
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

    def no_store():
        return JSONResponse({"enabled": False, "error": "Postgres history is not configured "
                             "(set LLTOP_PG_DSN)"}, status_code=404)

    async def from_store(read):
        try:
            return JSONResponse(await read())
        except Exception as e:
            return JSONResponse({"enabled": True, "error": f"{type(e).__name__}: {e}"},
                                status_code=503)

    @app.get("/lltop/api/history", include_in_schema=False)
    async def history_list(model: str | None = None, status: str | None = None,
                           endpoint: str | None = None, client: str | None = None,
                           q: str | None = None, before: str | None = None,
                           limit: int = Query(50, ge=1, le=200)):
        if not store:
            return no_store()
        return await from_store(lambda: store.search(model=model, status=status, endpoint=endpoint,
                                                     client=client, q=q, before=before, limit=limit))

    @app.get("/lltop/api/history-facets", include_in_schema=False)
    async def history_facets():
        if not store:
            return no_store()
        return await from_store(store.facets)

    @app.get("/lltop/api/history/{request_id}", include_in_schema=False)
    async def history_get(request_id: str):
        if not store:
            return no_store()

        async def read():
            rec = await store.get(request_id)
            return rec if rec else {"error": "not found"}
        return await from_store(read)

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
