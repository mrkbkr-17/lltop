"""The FastAPI app: the dashboard under /lltop/, everything else forwarded to Ollama.

    /lltop/            the dashboard page
    /lltop/ws          WebSocket: a snapshot on connect, then every event live
    /lltop/api/state   the same snapshot as JSON (handy with curl)
    /lltop/api/history         Postgres history: ?model= &status= &endpoint= &client= &q=
                               &before= (the `next` of the previous page) &limit=
    /lltop/api/history/{id}    one request in full
    /lltop/api/history-facets  models, endpoints, statuses with counts (for the filters)
    /lltop/api/stats           aggregates for the Charts tab: ?range=24h|7d|30d|90d &model= &tz=
    /lltop/api/machine         machine readings for the Machine tab: ?from= &to= (epoch seconds)
                               &machine= (a name; default: this machine)
    /lltop/api/meters          this machine's last reading, for another lltop's --machines
    anything else     forwarded to Ollama unchanged (proxy.forward)
"""

import asyncio
import logging
import socket
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from .history import History, save_forever, save_now
from .store import PgStore
from .hub import Hub
from .machine import METERS_PATH, Machine, MinuteLog, Remote, collect_forever
from .machine import poll_forever as poll_machine
from .proxy import APP_HEADERS, forward
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


def last_reading(hub: Hub, name: str) -> JSONResponse:
    """This machine's last reading and its age, for another lltop that collects it."""
    ev = hub.machines.get(name)
    if not ev or ev.get("remote") or "ts" not in ev:
        return JSONResponse({"error": "no machine reading (the meters are off, or not "
                             "supported on this system)"}, status_code=404)
    return JSONResponse({**ev, "age_s": round(time.time() - ev["ts"], 1)})


def create_meters_app(*, poll_interval: float = 3.0, machine=True) -> FastAPI:
    """`lltop --meters-only`: nothing but this machine's meters at /lltop/api/meters. No
    proxy, no dashboard, no prompts, nothing stored; for a machine whose readings another
    lltop shows (--machines)."""
    hub = Hub()
    reader = Machine() if machine is True else machine
    name = getattr(reader, "name", None) or socket.gethostname()

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(poll_machine(reader, hub, poll_interval, name=name))
        log.info("meters only: %s answers with this machine's readings", METERS_PATH)
        yield
        task.cancel()

    app = FastAPI(title="lltop meters", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.hub = hub

    @app.get(METERS_PATH, include_in_schema=False)
    async def meters():
        return last_reading(hub, name)

    return app


def create_app(upstream: str = "http://127.0.0.1:11434", *, poll_interval: float = 3.0,
               transport: httpx.AsyncBaseTransport | None = None, hub: Hub | None = None,
               start_poller: bool = True, history_path: str | Path | None = None,
               pg_dsn: str | None = None, app_headers: tuple[str, ...] | None = None,
               machine=None, machines: list[tuple[str, str]] | None = None,
               machines_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """machine: True reads this machine's meters (machine.Machine), or pass a reader with
    an async read(); None or False leaves the machine card out.
    machines: [(name, url)] of other machines whose lltop answers /lltop/api/meters."""
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

    reader = Machine() if machine is True else machine
    local_name = (getattr(reader, "name", None) or socket.gethostname()) if reader else ""
    logs = {local_name: MinuteLog(local_name)} if reader else {}
    if store:
        store.local_machine = local_name
    remotes_client = httpx.AsyncClient(transport=machines_transport, timeout=4.0, follow_redirects=False)
    remotes = [Remote(name, url, remotes_client) for name, url in machines or []]
    save_minute = store.record_machine if store else None

    @asynccontextmanager
    async def lifespan(app):
        tasks = []
        if start_poller:
            tasks.append(asyncio.create_task(poll_forever(client, hub, poll_interval)))
        if reader:
            tasks.append(asyncio.create_task(poll_machine(
                reader, hub, poll_interval, logs[local_name], save_minute, local_name)))
        for remote in remotes:
            hub.publish({"type": "machine", "name": remote.name, "remote": True, "waiting": True})
            logs[remote.name] = MinuteLog(remote.name)
            tasks.append(asyncio.create_task(collect_forever(
                remote, hub, poll_interval, logs[remote.name], save_minute)))
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
        for minutes in logs.values():
            last_minute = minutes.flush()
            if store and last_minute:
                store.record_machine(last_minute)
        await remotes_client.aclose()
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

    @app.get(METERS_PATH, include_in_schema=False)
    async def meters():
        return last_reading(hub, local_name)

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
                           limit: int = Query(50, ge=1, le=200),
                           running_from: float | None = None, running_to: float | None = None):
        if not store:
            return no_store()
        async def read():
            page = await store.search(model=model, status=status, endpoint=endpoint,
                                      client=client, q=q, before=before, limit=limit,
                                      running_from=running_from, running_to=running_to)
            return {**page, "status": store.status()}
        return await from_store(read)

    @app.get("/lltop/api/history-facets", include_in_schema=False)
    async def history_facets():
        if not store:
            return no_store()
        return await from_store(store.facets)

    @app.get("/lltop/api/stats", include_in_schema=False)
    async def history_stats(range: str = "24h", model: str | None = None, tz: str | None = None):
        if not store:
            return no_store()
        return await from_store(lambda: store.stats(range, model, tz))

    @app.get("/lltop/api/machine", include_in_schema=False)
    async def machine_history(start: float = Query(..., alias="from", ge=0, le=4e9),
                              end: float = Query(..., alias="to", ge=0, le=4e9),
                              machine: str | None = Query(None, max_length=80)):
        if not store:
            return no_store()
        return await from_store(lambda: store.machine_history(start, end, machine))

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
        return await forward(request, client, hub, app_headers or APP_HEADERS)

    return app
