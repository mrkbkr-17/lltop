"""Postgres history: every finished request becomes a row in table `requests`.

Optional: used when LLTOP_PG_DSN is set. The database holds every prompt and response,
so give lltop its own database and role, and don't grant other roles access.

Fail open, like the rest of lltop:
  - record() only puts the row on a bounded queue; it never waits and never raises.
  - A background task writes the queue to Postgres. If Postgres is down, it keeps the
    row and retries with backoff; rows that don't fit in the queue meanwhile are
    dropped (counted in `dropped`). Requests to Ollama never wait for the database.
  - Reads (the History tab) use their own connection; if Postgres is down they fail
    with an error the dashboard shows, nothing else is affected.
"""

import asyncio
import logging
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

log = logging.getLogger("lltop")

QUEUE_MAX = 10_000
PREVIEW_CHARS = 300

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                  text PRIMARY KEY,
    started_at          timestamptz NOT NULL,
    first_token_at      timestamptz,
    ended_at            timestamptz,
    status              text NOT NULL,      -- finished | error | cancelled | interrupted
    model               text,
    endpoint            text,
    api                 text,               -- chat | generate | openai
    client              text,
    stream              boolean,
    was_loaded          boolean,            -- per /api/ps when the request started
    preview             text,               -- last user message / prompt, shortened
    prompt              text,               -- /api/generate
    system              text,
    messages            jsonb,              -- chat endpoints (images counted, not stored)
    images              integer,
    tools               jsonb,
    options             jsonb,
    content             text,
    thinking            text,
    tool_calls          jsonb,
    error               text,
    http_status         integer,
    prompt_tokens       integer,
    output_tokens       integer,
    prompt_tokens_per_s double precision,
    output_tokens_per_s double precision,
    rate_source         text,               -- ollama | chunks (estimated)
    ttft_s              double precision,
    load_s              double precision,
    prompt_eval_s       double precision,
    eval_s              double precision,
    total_s             double precision,   -- Ollama's total_duration
    wall_s              double precision,   -- as measured by lltop
    cold_load           boolean,
    done_reason         text,
    chunks              integer,
    raw_stats           jsonb,
    search              tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(preview, '') || ' ' || coalesce(system, '') || ' ' ||
                              left(coalesce(content, ''), 100000))) STORED
);
CREATE INDEX IF NOT EXISTS requests_started_idx ON requests (started_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS requests_model_idx ON requests (model, started_at DESC);
CREATE INDEX IF NOT EXISTS requests_search_idx ON requests USING gin (search);
"""

STAT_COLUMNS = ("prompt_tokens", "output_tokens", "prompt_tokens_per_s", "output_tokens_per_s",
                "rate_source", "ttft_s", "load_s", "prompt_eval_s", "eval_s", "total_s", "wall_s",
                "cold_load", "done_reason")
TEXT_COLUMNS = ("model", "endpoint", "api", "client", "preview", "prompt", "system", "content",
                "thinking", "error")
JSON_COLUMNS = ("messages", "tools", "options", "tool_calls", "raw_stats")
COLUMNS = (("id", "started_at", "first_token_at", "ended_at", "status", "stream", "was_loaded",
            "images", "http_status", "chunks") + TEXT_COLUMNS + JSON_COLUMNS + STAT_COLUMNS)

UPSERT = (f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES "
          f"({', '.join(f'%({c})s' for c in COLUMNS)}) ON CONFLICT (id) DO UPDATE SET "
          + ", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS if c != "id"))

# What the History list needs (no full texts); the detail view loads one row in full.
LIST_COLUMNS = ("id, started_at, ended_at, status, model, endpoint, client, preview, error, "
                "left(content, 200) AS content_preview, " + ", ".join(STAT_COLUMNS))


# ---------------------------------------------------------------- record <-> row

def _clean(value):
    """Postgres text and jsonb can't hold NUL characters (PDF text sometimes has them)."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {_clean(k): _clean(v) for k, v in value.items()}
    return value


def _ts(epoch):
    return None if epoch is None else datetime.fromtimestamp(epoch, timezone.utc)


def _epoch(dt):
    return None if dt is None else dt.timestamp()


def preview_of(rec: dict) -> str:
    """The last user message (chat) or the prompt (generate), shortened."""
    text = rec.get("prompt")
    if text is None:
        msgs = rec.get("messages") or []
        users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        pick = users[-1] if users else (msgs[-1] if msgs else {})
        text = pick.get("content") if isinstance(pick, dict) else ""
    text = text if isinstance(text, str) else ""
    return text[:PREVIEW_CHARS]


def to_row(rec: dict) -> dict:
    """A finished hub record (see hub.Hub._apply) as a `requests` row."""
    stats = rec.get("stats") or {}
    row = {
        "id": rec["id"],
        "started_at": _ts(rec.get("ts")),
        "first_token_at": _ts(rec.get("first_token_ts")),
        "ended_at": _ts(rec.get("ended_ts")),
        "status": rec.get("status") or "finished",
        "stream": rec.get("stream"),
        "was_loaded": rec.get("was_loaded"),
        "images": rec.get("images"),
        "http_status": rec.get("http_status"),
        "chunks": rec.get("chunks"),
        "preview": preview_of(rec),
    }
    for c in TEXT_COLUMNS:
        row.setdefault(c, rec.get(c))
    for c in JSON_COLUMNS:
        v = rec.get(c)
        row[c] = None if v is None else Jsonb(_clean(v))
    for c in STAT_COLUMNS:
        row[c] = stats.get(c)
    if row["done_reason"] is None:
        row["done_reason"] = (rec.get("raw_stats") or {}).get("done_reason")
    return {k: _clean(v) if isinstance(v, str) else v for k, v in row.items()}


def from_row(row: dict) -> dict:
    """A `requests` row in the shape the dashboard uses for finished requests."""
    rec = {k: v for k, v in row.items() if k not in STAT_COLUMNS and k != "search"}
    rec["stats"] = {c: row[c] for c in STAT_COLUMNS if c in row}
    rec["ts"] = _epoch(rec.pop("started_at", None))
    rec["ended_ts"] = _epoch(rec.pop("ended_at", None))
    if "first_token_at" in rec:
        rec["first_token_ts"] = _epoch(rec.pop("first_token_at"))
    return rec


def _like(text: str) -> str:
    return "%" + text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


# ---------------------------------------------------------------- the store

class PgStore:
    def __init__(self, dsn: str, queue_max: int = QUEUE_MAX):
        self.dsn = dsn
        self.queue: asyncio.Queue = asyncio.Queue(queue_max)
        self.ok: bool | None = None          # None until the first write or read
        self.error: str | None = None
        self.dropped = 0
        self.written = 0
        self._write_conn = None
        self._read_conn = None
        self._read_lock = asyncio.Lock()
        self._schema_ready = False
        self._holding = 0                    # a row taken off the queue, not yet written

    # ------------------------------------------------------------ connections

    async def _connect(self):
        conn = await psycopg.AsyncConnection.connect(self.dsn, autocommit=True,
                                                     connect_timeout=5, row_factory=dict_row)
        if not self._schema_ready:
            await conn.execute(SCHEMA)
            self._schema_ready = True
        return conn

    def _down(self, e: Exception) -> None:
        if self.ok is not False:
            log.warning("postgres history unavailable (%s: %s); retrying", type(e).__name__, e)
        self.ok = False
        self.error = f"{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''}"

    def _up(self) -> None:
        if self.ok is False:
            log.info("postgres history available again")
        self.ok, self.error = True, None

    def status(self) -> dict:
        return {"enabled": True, "ok": self.ok, "error": self.error, "queued": self.queue.qsize() + self._holding,
                "dropped": self.dropped, "written": self.written}

    # ------------------------------------------------------------ writing

    def record(self, rec: dict) -> None:
        """Queue a finished request for saving. Never blocks, never raises."""
        try:
            self.queue.put_nowait(to_row(rec))
        except asyncio.QueueFull:
            self.dropped += 1
        except Exception:
            log.exception("postgres history: could not prepare request %s", rec.get("id"))

    async def run(self) -> None:
        """Background task: write queued rows, retrying while Postgres is down."""
        backoff = 1.0
        while True:
            row = await self.queue.get()
            self._holding = 1
            while True:
                try:
                    if self._write_conn is None or self._write_conn.closed:
                        self._write_conn = await self._connect()
                    await self._write_conn.execute(UPSERT, row)
                    self.written += 1
                    self._up()
                    backoff = 1.0
                    break
                except (psycopg.OperationalError, psycopg.InterfaceError, OSError) as e:
                    self._down(e)
                    await self._close("_write_conn")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                except Exception:
                    # Postgres rejected this row (a data problem): skip it, keep the rest.
                    log.exception("postgres history: could not save request %s", row.get("id"))
                    break
            self._holding = 0

    async def flush(self, timeout: float = 3.0) -> None:
        """On shutdown: write what's still queued, if Postgres answers quickly."""
        try:
            async with asyncio.timeout(timeout):
                while not self.queue.empty():
                    row = self.queue.get_nowait()
                    if self._write_conn is None or self._write_conn.closed:
                        self._write_conn = await self._connect()
                    await self._write_conn.execute(UPSERT, row)
                    self.written += 1
        except Exception as e:
            left = self.queue.qsize()
            if left:
                log.warning("postgres history: %d requests not saved at shutdown (%s)", left, e)

    async def _close(self, attr: str) -> None:
        conn = getattr(self, attr)
        setattr(self, attr, None)
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass

    async def close(self) -> None:
        await self._close("_write_conn")
        await self._close("_read_conn")

    # ------------------------------------------------------------ reading

    async def _read(self, sql: str, params=None) -> list[dict]:
        async with self._read_lock:
            for attempt in (1, 2):           # one reconnect if the connection went stale
                try:
                    if self._read_conn is None or self._read_conn.closed:
                        self._read_conn = await self._connect()
                    rows = await (await self._read_conn.execute(sql, params)).fetchall()
                    self._up()
                    return rows
                except (psycopg.OperationalError, psycopg.InterfaceError, OSError) as e:
                    await self._close("_read_conn")
                    if attempt == 2:
                        self._down(e)
                        raise

    async def recent(self, limit: int) -> list[dict]:
        rows = await self._read("SELECT * FROM requests ORDER BY started_at DESC, id DESC LIMIT %s",
                                (limit,))
        return [from_row(r) for r in rows]

    async def get(self, request_id: str) -> dict | None:
        rows = await self._read("SELECT * FROM requests WHERE id = %s", (request_id,))
        return from_row(rows[0]) if rows else None

    async def search(self, *, model=None, status=None, endpoint=None, client=None, q=None,
                     before=None, limit: int = 50) -> dict:
        """Newest first. `before` is the `next` value of the previous page."""
        where, params = [], {}
        for col, val in (("model", model), ("status", status), ("endpoint", endpoint)):
            if val:
                where.append(f"{col} = %({col})s")
                params[col] = val
        if client:
            where.append("client ILIKE %(client)s")
            params["client"] = _like(client)
        if q:
            # whole words via the full-text index, or any substring of the preview/response
            where.append("(search @@ plainto_tsquery('simple', %(q)s) OR preview ILIKE %(qlike)s "
                         "OR left(content, 100000) ILIKE %(qlike)s)")
            params["q"], params["qlike"] = q, _like(q)
        if before:
            ts, _, rid = before.partition("|")
            where.append("(started_at, id) < (%(before_ts)s::timestamptz, %(before_id)s)")
            params["before_ts"], params["before_id"] = ts, rid
        params["limit"] = max(1, min(int(limit), 200))
        sql = (f"SELECT {LIST_COLUMNS} FROM requests "
               f"{'WHERE ' + ' AND '.join(where) if where else ''} "
               "ORDER BY started_at DESC, id DESC LIMIT %(limit)s")
        rows = await self._read(sql, params)
        nxt = None
        if len(rows) == params["limit"]:
            last = rows[-1]
            nxt = f"{last['started_at'].isoformat()}|{last['id']}"
        return {"rows": [from_row(r) for r in rows], "next": nxt}

    async def facets(self) -> dict:
        rows = await self._read(
            "SELECT 'model' AS kind, model AS value, count(*) AS n FROM requests GROUP BY model "
            "UNION ALL SELECT 'endpoint', endpoint, count(*) FROM requests GROUP BY endpoint "
            "UNION ALL SELECT 'status', status, count(*) FROM requests GROUP BY status "
            "ORDER BY 1, 3 DESC")
        out = {"model": [], "endpoint": [], "status": []}
        for r in rows:
            if r["value"] is not None:
                out[r["kind"]].append({"value": r["value"], "count": r["n"]})
        return out
