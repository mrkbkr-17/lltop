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
    app                 text,               -- calling app (X-Lltop-App, LiteLLM key alias)
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
ALTER TABLE requests ADD COLUMN IF NOT EXISTS app text;   -- added after the first release
CREATE INDEX IF NOT EXISTS requests_started_idx ON requests (started_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS requests_model_idx ON requests (model, started_at DESC);
CREATE INDEX IF NOT EXISTS requests_search_idx ON requests USING gin (search);
-- One row per machine and minute of readings (machine.MinuteLog): averages, and peaks
-- where a short spike matters. Percentages are 0-100. No prompts or other text.
CREATE TABLE IF NOT EXISTS machine_minutes (
    machine     text NOT NULL DEFAULT '',   -- its name on the dashboard
    minute      timestamptz NOT NULL,
    samples     integer NOT NULL,
    cpu         real,               -- average use
    cpu_max     real,               -- highest reading in the minute
    mem         real,               -- average use
    mem_used    bigint,
    mem_total   bigint,
    load1       real,
    temp        real,               -- highest CPU temperature, °C
    disk        real,
    gpus        jsonb,              -- [{name, util, util_max, mem, temp, watts}], one per GPU
    PRIMARY KEY (machine, minute)
);
-- A table from before several machines were tracked: one row per minute, no name.
ALTER TABLE machine_minutes ADD COLUMN IF NOT EXISTS machine text NOT NULL DEFAULT '';
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'machine_minutes'::regclass
               AND contype = 'p' AND array_length(conkey, 1) = 1) THEN
        ALTER TABLE machine_minutes DROP CONSTRAINT machine_minutes_pkey;
        ALTER TABLE machine_minutes ADD PRIMARY KEY (machine, minute);
    END IF;
END $$;
"""

MACHINE_COLUMNS = ("machine", "minute", "samples", "cpu", "cpu_max", "mem", "mem_used", "mem_total", "load1",
                   "temp", "disk", "gpus")
MACHINE_UPSERT = (f"INSERT INTO machine_minutes ({', '.join(MACHINE_COLUMNS)}) VALUES "
                  f"({', '.join(f'%({c})s' for c in MACHINE_COLUMNS)}) ON CONFLICT (machine, minute) DO UPDATE SET "
                  + ", ".join(f"{c} = EXCLUDED.{c}" for c in MACHINE_COLUMNS if c not in ("machine", "minute")))
# Rows saved before machines had names belong to the machine lltop runs on.
MACHINE_ADOPT = """UPDATE machine_minutes m SET machine = %(name)s WHERE m.machine = '' AND NOT EXISTS
                   (SELECT 1 FROM machine_minutes n WHERE n.machine = %(name)s AND n.minute = m.minute)"""
# Bucket sizes for the Machine tab, in seconds: the smallest that keeps a window to
# about MACHINE_POINTS points.
MACHINE_STEPS = (60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400)
MACHINE_POINTS = 180


def machine_step(span_s: float) -> int:
    return next((s for s in MACHINE_STEPS if span_s / s <= MACHINE_POINTS), MACHINE_STEPS[-1])

STAT_COLUMNS = ("prompt_tokens", "output_tokens", "prompt_tokens_per_s", "output_tokens_per_s",
                "rate_source", "ttft_s", "load_s", "prompt_eval_s", "eval_s", "total_s", "wall_s",
                "cold_load", "done_reason")
TEXT_COLUMNS = ("model", "endpoint", "api", "client", "app", "preview", "prompt", "system", "content",
                "thinking", "error")
JSON_COLUMNS = ("messages", "tools", "options", "tool_calls", "raw_stats")
COLUMNS = (("id", "started_at", "first_token_at", "ended_at", "status", "stream", "was_loaded",
            "images", "http_status", "chunks") + TEXT_COLUMNS + JSON_COLUMNS + STAT_COLUMNS)

UPSERT = (f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES "
          f"({', '.join(f'%({c})s' for c in COLUMNS)}) ON CONFLICT (id) DO UPDATE SET "
          + ", ".join(f"{c} = EXCLUDED.{c}" for c in COLUMNS if c != "id"))

# What the History list needs (no full texts); the detail view loads one row in full.
LIST_COLUMNS = ("id, started_at, ended_at, status, model, endpoint, client, app, preview, error, "
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


# ---------------------------------------------------------------- charts

# range -> (bucket unit, first bucket as SQL, generate_series step)
RANGES = {
    "24h": ("hour", "date_trunc('hour', now() - interval '23 hours', %(tz)s)", "1 hour"),
    "7d": ("6h", "date_trunc('day', now() - interval '6 days', %(tz)s)", "6 hours"),
    "30d": ("day", "date_trunc('day', now() - interval '29 days', %(tz)s)", "1 day"),
    "90d": ("day", "date_trunc('day', now() - interval '89 days', %(tz)s)", "1 day"),
}

FAILED = "('error', 'cancelled', 'interrupted')"


def _tz(name: str | None) -> str:
    from zoneinfo import available_timezones
    return name if name and name in available_timezones() else "UTC"


# ---------------------------------------------------------------- the store

class PgStore:
    def __init__(self, dsn: str, queue_max: int = QUEUE_MAX):
        self.dsn = dsn
        self.queue: asyncio.Queue = asyncio.Queue(queue_max)
        self.ok: bool | None = None          # None until the first write or read
        self.error: str | None = None
        self.dropped = 0
        self.written = 0
        self.machine_written = 0
        self._write_conn = None
        self._read_conn = None
        self._read_lock = asyncio.Lock()
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()
        self.local_machine = ""              # set by the app: the name of the machine lltop runs on
        self._holding = 0                    # a row taken off the queue, not yet written

    # ------------------------------------------------------------ connections

    async def _connect(self):
        conn = await psycopg.AsyncConnection.connect(self.dsn, autocommit=True,
                                                     connect_timeout=5, row_factory=dict_row)
        async with self._schema_lock:
            if not self._schema_ready:
                await conn.execute(SCHEMA)
                if self.local_machine:
                    await conn.execute(MACHINE_ADOPT, {"name": self.local_machine})
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
            self.queue.put_nowait((UPSERT, to_row(rec)))
        except asyncio.QueueFull:
            self.dropped += 1
        except Exception:
            log.exception("postgres history: could not prepare request %s", rec.get("id"))

    def record_machine(self, row: dict) -> None:
        """Queue one minute of machine readings (machine.MinuteLog). Never blocks or raises."""
        try:
            self.queue.put_nowait((MACHINE_UPSERT, {
                **{c: row.get(c) for c in MACHINE_COLUMNS}, "machine": row.get("machine") or "",
                "minute": _ts(row["minute"]), "gpus": Jsonb(_clean(row.get("gpus") or []))}))
        except asyncio.QueueFull:
            pass                              # Postgres has been down for a long time
        except Exception:
            log.exception("postgres history: could not prepare a machine reading")

    async def _write(self, item) -> None:
        sql, row = item
        if self._write_conn is None or self._write_conn.closed:
            self._write_conn = await self._connect()
        await self._write_conn.execute(sql, row)
        if sql is UPSERT:
            self.written += 1
        else:
            self.machine_written += 1

    async def run(self) -> None:
        """Background task: write queued rows, retrying while Postgres is down."""
        backoff = 1.0
        while True:
            item = await self.queue.get()
            self._holding = 1
            while True:
                try:
                    await self._write(item)
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
                    log.exception("postgres history: could not save %s",
                                  item[1].get("id") or item[1].get("minute"))
                    break
            self._holding = 0

    async def flush(self, timeout: float = 3.0) -> None:
        """On shutdown: write what's still queued, if Postgres answers quickly."""
        try:
            async with asyncio.timeout(timeout):
                while not self.queue.empty():
                    await self._write(self.queue.get_nowait())
        except Exception as e:
            left = self.queue.qsize()
            if left:
                log.warning("postgres history: %d rows not saved at shutdown (%s)", left, e)

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
                     before=None, limit: int = 50, running_from: float | None = None,
                     running_to: float | None = None) -> dict:
        """Newest first. `before` is the `next` value of the previous page. running_from /
        running_to (epoch seconds) keep the requests that were running at some moment in
        that window."""
        where, params = [], {}
        if running_to is not None:
            where.append("started_at < %(run_to)s")
            params["run_to"] = _ts(running_to)
        if running_from is not None:
            where.append("coalesce(ended_at, started_at) >= %(run_from)s")
            params["run_from"] = _ts(running_from)
        for col, val in (("model", model), ("status", status), ("endpoint", endpoint)):
            if val:
                where.append(f"{col} = %({col})s")
                params[col] = val
        if client:                            # matches the app name or the address
            where.append("(client ILIKE %(client)s OR app ILIKE %(client)s)")
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

    async def machine_history(self, start: float, end: float, machine: str | None = None) -> dict:
        """The Machine tab: one machine's readings between two moments (epoch seconds), in
        buckets sized so any window is about MACHINE_POINTS points or fewer, plus the
        requests started in each bucket. A bucket without readings (lltop was not running)
        holds None, so lines show a gap there. Without a name: the machine lltop runs on,
        or else the first one with readings. "machines" lists every machine with readings."""
        known = await self._read("""SELECT machine AS name, min(minute) AS first, max(minute) AS last
                                    FROM machine_minutes GROUP BY 1 ORDER BY 1""")
        names = [m["name"] for m in known]
        if machine is None:
            machine = self.local_machine if self.local_machine in names or not names else names[0]
        end = max(end, start + 60)
        step = machine_step(end - start)
        first = int(start // step) * step
        times = list(range(first, int(end), step))
        p = {"step": step, "first": _ts(first), "end": _ts(end), "machine": machine}
        bucket = "date_bin(%(step)s * interval '1 second', {col}, %(first)s)"
        scope = "{col} >= %(first)s AND {col} < %(end)s"
        rows = await self._read(f"""
            SELECT {bucket.format(col='minute')} AS b, avg(cpu) AS cpu, max(cpu_max) AS cpu_max,
                   avg(mem) AS mem, avg(load1) AS load1, max(temp) AS temp, avg(disk) AS disk,
                   max(mem_total) AS mem_total
            FROM machine_minutes WHERE machine = %(machine)s AND {scope.format(col='minute')}
            GROUP BY 1""", p)
        gpu_rows = await self._read(f"""
            SELECT {bucket.format(col='minute')} AS b, g.n, max(g.e->>'name') AS name,
                   avg((g.e->>'util')::real) AS util, max((g.e->>'util_max')::real) AS util_max,
                   avg((g.e->>'mem')::real) AS mem, max((g.e->>'temp')::real) AS temp,
                   avg((g.e->>'watts')::real) AS watts
            FROM machine_minutes, jsonb_array_elements(gpus) WITH ORDINALITY AS g(e, n)
            WHERE machine = %(machine)s AND {scope.format(col='minute')} GROUP BY 1, 2""", p)
        req_rows = await self._read(f"""
            SELECT {bucket.format(col='started_at')} AS b, count(*) AS requests
            FROM requests WHERE {scope.format(col='started_at')} GROUP BY 1""", p)
        span = next((m for m in known if m["name"] == machine), {"first": None, "last": None})
        index = {t: i for i, t in enumerate(times)}
        fields = ("cpu", "cpu_max", "mem", "load1", "temp", "disk")
        series = {k: [None] * len(times) for k in fields}
        series["requests"] = [0] * len(times)
        gpus: dict[int, dict] = {}
        mem_total = None
        for r in rows:
            i = index.get(int(r["b"].timestamp()))
            if i is not None:
                for k in fields:
                    series[k][i] = None if r[k] is None else round(r[k], 1)
                mem_total = r["mem_total"] or mem_total
        for r in gpu_rows:
            i = index.get(int(r["b"].timestamp()))
            if i is None:
                continue
            g = gpus.setdefault(r["n"], {"name": r["name"], **{
                k: [None] * len(times) for k in ("util", "util_max", "mem", "temp", "watts")}})
            g["name"] = r["name"] or g["name"]
            for k in ("util", "util_max", "mem", "temp", "watts"):
                g[k][i] = None if r[k] is None else round(r[k], 1)
        for r in req_rows:
            i = index.get(int(r["b"].timestamp()))
            if i is not None:
                series["requests"][i] = r["requests"]
        return {"from": first, "to": int(end), "step": step, "times": times, "series": series,
                "gpus": [gpus[n] for n in sorted(gpus)], "mem_total": mem_total,
                "first": _epoch(span["first"]), "last": _epoch(span["last"]), "machine": machine,
                "machines": [{"name": m["name"], "first": _epoch(m["first"]), "last": _epoch(m["last"])}
                             for m in known]}

    async def stats(self, range_: str = "24h", model: str | None = None, tz: str | None = None) -> dict:
        """Everything the Charts tab draws, for one time range (and optionally one model).

        Buckets are in the viewer's time zone (days start at their midnight). Medians count
        finished requests only. Time to first token includes waiting in Ollama's queue and
        model loads, which is the point: it is what callers wait."""
        unit, start_sql, step = RANGES.get(range_, RANGES["24h"])
        p = {"tz": _tz(tz), "model": model or None, "step": step}
        start = (await self._read(f"SELECT {start_sql} AS start", p))[0]["start"]
        p["start"] = start
        bucket_sql = {"hour": "date_trunc('hour', started_at, %(tz)s)",
                      "day": "date_trunc('day', started_at, %(tz)s)",
                      "6h": "date_bin('6 hours', started_at, %(start)s)"}[unit]
        scope = "started_at >= %(start)s AND (%(model)s::text IS NULL OR model = %(model)s)"
        buckets = [r["b"] for r in await self._read(
            "SELECT generate_series(%(start)s::timestamptz, now(), %(step)s::interval, %(tz)s) AS b", p)]
        per = await self._read(f"""
            SELECT {bucket_sql} AS b, model,
                   count(*) AS requests,
                   count(*) FILTER (WHERE status IN {FAILED}) AS failed,
                   coalesce(sum(output_tokens), 0) AS output_tokens,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY output_tokens_per_s)
                       FILTER (WHERE status = 'finished') AS tps,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY ttft_s)
                       FILTER (WHERE status = 'finished') AS ttft
            FROM requests WHERE {scope} GROUP BY 1, 2""", p)
        kpi = (await self._read(f"""
            SELECT count(*) AS requests,
                   coalesce(sum(output_tokens), 0) AS output_tokens,
                   coalesce(sum(prompt_tokens), 0) AS prompt_tokens,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY output_tokens_per_s)
                       FILTER (WHERE status = 'finished') AS tps,
                   percentile_cont(0.5) WITHIN GROUP (ORDER BY ttft_s)
                       FILTER (WHERE status = 'finished') AS ttft,
                   percentile_cont(0.9) WITHIN GROUP (ORDER BY ttft_s)
                       FILTER (WHERE status = 'finished') AS ttft_p90,
                   count(*) FILTER (WHERE status IN {FAILED}) AS failed,
                   count(*) FILTER (WHERE cold_load) AS cold_loads
            FROM requests WHERE {scope}""", p))[0]
        clients = await self._read(f"""
            SELECT coalesce(app, regexp_replace(coalesce(client, '?'), ':[0-9]+$', '')) AS client,
                   app IS NOT NULL AS is_app,
                   count(*) AS requests, coalesce(sum(output_tokens), 0) AS output_tokens
            FROM requests WHERE {scope} GROUP BY 1, 2 ORDER BY 3 DESC, 1 LIMIT 10""", p)
        # Colors follow the model: order every model by when it was first seen, so a new model
        # or a filter never repaints the others.
        order = [r["model"] for r in await self._read(
            "SELECT model FROM requests WHERE model IS NOT NULL GROUP BY model ORDER BY min(started_at), model")]
        index = {b: i for i, b in enumerate(buckets)}
        models = sorted({r["model"] for r in per if r["model"]}, key=lambda m: order.index(m) if m in order else 999)
        series = {m: {k: [None if k in ("tps", "ttft") else 0] * len(buckets)
                      for k in ("requests", "failed", "output_tokens", "tps", "ttft")} for m in models}
        for r in per:
            i = index.get(r["b"])
            if i is None or r["model"] not in series:
                continue
            for k in ("requests", "failed", "output_tokens", "tps", "ttft"):
                series[r["model"]][k][i] = r[k]
        return {"range": range_ if range_ in RANGES else "24h", "unit": unit, "tz": p["tz"],
                "buckets": [b.isoformat() for b in buckets], "model_order": order,
                "models": models, "series": series, "kpis": kpi, "clients": clients}
