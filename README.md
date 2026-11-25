# lltop

*Like `htop`, but for LLMs.*

A live monitoring dashboard for [Ollama](https://ollama.com). Put it in front of Ollama
and watch every request as it happens: which model, who sent it, the prompt, the thinking
and the response appearing token by token, then the final speed and timing numbers.

It is a **streaming reverse proxy**. Clients talk to lltop exactly as they would
talk to Ollama; every request is forwarded unchanged and every response chunk is passed
straight back. A copy of each chunk goes to the dashboard.

Useful when Ollama is slow (CPU-only machines, big models) and you want to know what it is
doing right now, which app is using it, and how fast it really is.

> Status: milestone 2. The last 50 requests survive restarts (a JSON file); with
> PostgreSQL configured, every request is kept, searchable in the History tab and charted
> in the Charts tab.

## What it shows

- **One card per model**: loaded or not, size, how much of it sits in GPU memory vs
  system RAM, context length, and a countdown to when Ollama unloads it (from `/api/ps`,
  polled every 3 s).
- **Requests in progress**, inside their model's card: endpoint, client address, elapsed
  time, phase (reading prompt / thinking / generating), the prompt (plus the whole
  conversation, tools and options), and the thinking and response text as it streams,
  with a live tokens/s estimate.
- **History tab** (with PostgreSQL): every request ever made, newest first, filtered by
  model, status, endpoint and client, with full-text search in prompts and responses.
- **Charts tab** (with PostgreSQL), for the last 24 hours or 7, 30 or 90 days, all models
  or one: requests, tokens, median speed, median wait for the first token, failures and
  cold loads as headline numbers; requests over time stacked by model; output speed and
  wait for the first token over time, one line per model (the wait includes time spent
  in Ollama's queue); and which clients use the models. Each chart has a table view.
- **The last 50 finished requests** (saved to a file, so a restart keeps them): prompt and output tokens, time to first token,
  prompt and output tokens/s, total time, cold or warm model load, status (finished,
  error, cancelled). Click a row for the full prompt, response and Ollama's raw stats.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a running Ollama.

```sh
git clone https://github.com/mrkbkr-17/lltop.git
cd lltop
uv run lltop
```

Point a client at port **11435** instead of 11434:

```sh
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}]
}'
```

and open the dashboard at <http://127.0.0.1:11435/lltop/>.

### Options

| Flag | Environment | Default | |
|---|---|---|---|
| `--host` | `LLTOP_HOST` | `127.0.0.1` | Address to listen on. `0.0.0.0` exposes it to your network. |
| `--port` | `LLTOP_PORT` | `11435` | |
| `--upstream` | `LLTOP_UPSTREAM` | `http://127.0.0.1:11434` | Where Ollama is. |
| `--poll` | `LLTOP_POLL` | `3` | Seconds between `/api/ps` polls. |
| `--history` | `LLTOP_HISTORY` | `~/.local/state/lltop/history.json` | Where the last 50 requests are saved. `--no-history` keeps them in memory only. |
| `--pg-dsn` | `LLTOP_PG_DSN` | off | PostgreSQL for the full history, e.g. `postgresql://lltop@localhost:5432/lltop`. Password in `~/.pgpass`, not here. |
| `--log-level` | `LLTOP_LOG_LEVEL` | `info` | |

### Docker

```sh
docker build -t lltop .
docker run --rm -p 127.0.0.1:11435:11435 -v lltop-data:/data \
  --add-host=host.docker.internal:host-gateway lltop
```

The image listens on all interfaces inside the container and expects Ollama at
`http://host.docker.internal:11434` (set `LLTOP_UPSTREAM` otherwise, e.g.
`http://ollama:11434` when both run on the same Docker network).

### History in PostgreSQL (optional)

Every finished request becomes a row in table `requests` (created automatically): the
prompt or messages, tools, options, thinking, response, client, and the stats as columns.
It holds every prompt, so give lltop its own database and role that nobody else can
connect to:

```sql
CREATE ROLE lltop LOGIN PASSWORD '…';
CREATE DATABASE lltop OWNER lltop;
REVOKE CONNECT ON DATABASE lltop FROM PUBLIC;
```

Then put the password in `~/.pgpass` (`host:5432:lltop:lltop:password`, mode 0600) and
start lltop with `LLTOP_PG_DSN=postgresql://lltop@host:5432/lltop`. If PostgreSQL is
unreachable, lltop keeps working: rows wait in a queue (10,000) and are written when it is
back. On the first start it also copies the requests from the history file.

### Run as a service (systemd)

[`contrib/systemd/lltop.service`](contrib/systemd/lltop.service) runs lltop as a systemd
user service that starts at boot and restarts if it crashes:

```sh
uv sync --no-dev
systemctl --user link "$PWD/contrib/systemd/lltop.service"
systemctl --user enable --now lltop.service
journalctl --user -fu lltop
```

The unit expects the clone at `~/lltop`; edit `ExecStart` otherwise. Settings go in
`~/.config/lltop/env` (for example `LLTOP_HOST=0.0.0.0`). To keep it running while you are
logged out: `sudo loginctl enable-linger $USER`.

## Privacy

The dashboard shows every prompt and response in full, to anyone who can open the page.
Nothing leaves the machine. The last 50 requests, prompts and responses included, are
saved in the history file, readable by its owner only (mode 0600; `--no-history` to
turn it off). By default lltop only listens on `127.0.0.1` for that reason. Only use `--host 0.0.0.0` on a network you
trust: there is no login.

## How it works

```
  client ──HTTP──▶ lltop :11435 ──HTTP──▶ Ollama :11434
    ▲                 │   ▲                        │
    └──── chunks ◀────┤   └──────── chunks ◀───────┘
                      │ copy of each chunk
                      ▼
                    Tap ──events──▶ Hub ──WebSocket──▶ dashboard (/lltop/)
                                     ▲
                       /api/ps poller┘
```

- **Proxy** (`proxy.py`): forwards every method and path with its headers and body. The
  response is relayed chunk by chunk as it arrives; nothing is buffered. `POST`
  requests to `/api/chat`, `/api/generate` and `/v1/chat/completions` (streaming or
  not) are also handed to a tap, after the caller has been sent each chunk.
- **Tap** (`tap.py`): reads the request (model, messages or prompt, tools, options) and
  the response (Ollama's NDJSON, OpenAI-style Server-Sent Events, or a single JSON
  body), and emits `started`, `token`, `finished`, `error` and `cancelled` events.
- **Stats** (`stats.py`): tokens/s for prompt and output from Ollama's own durations,
  time to first token measured by the proxy, and a cold load when Ollama's
  `load_duration` is a second or more (or, when Ollama gives no stats, when `/api/ps`
  didn't list the model as the request started). `/v1/chat/completions` reports no
  durations, so its output speed is estimated from chunk timing (marked `*`).
- **Store** (`store.py`, optional): writes each finished request to PostgreSQL from a
  background queue and answers the History tab's queries.
- **Hub** (`hub.py`): the in-memory state (loaded models, requests in progress, last 50)
  and the fan-out to browsers. `history.py` saves the last 50 to a JSON file (at most
  every 2 s, and on shutdown, when requests still running are kept as `interrupted`). Each browser gets its own bounded queue; one that falls
  behind is disconnected and reconnects to a fresh snapshot.

### Design rules

- **Fail open.** Inspection can never break a request: tap methods catch their own
  errors and the proxy wraps every call to them again. If the dashboard code fails, the
  request still reaches Ollama and returns normally; the dashboard marks it as an error.
- **No added latency.** Chunks go to the caller before anything else looks at them, and
  publishing an event never waits for a browser.
- **Cancellation passes through.** When a caller hangs up, lltop closes its
  connection to Ollama, so Ollama stops generating, also while still waiting for the
  first token.
- **Transparent.** Everything outside `/lltop/` belongs to Ollama, including `/`.

### Endpoints of its own

| Path | |
|---|---|
| `/lltop/` | The dashboard |
| `/lltop/ws` | WebSocket: a `snapshot` on connect, then every event |
| `/lltop/api/state` | The snapshot as JSON |
| `/lltop/api/history` | PostgreSQL history: `?model=&status=&endpoint=&client=&q=&before=&limit=` |
| `/lltop/api/history/{id}` | One request in full |
| `/lltop/api/history-facets` | Models, endpoints and statuses with counts |
| `/lltop/api/stats` | Aggregates for the Charts tab: `?range=24h\|7d\|30d\|90d&model=&tz=` |

## Development

```sh
uv run pytest
```

The PostgreSQL tests need a throwaway database (they drop and recreate its `requests`
table) and are skipped otherwise:

```sh
LLTOP_TEST_PG_DSN=postgresql://lltop@localhost:5432/lltop_test uv run pytest
```

The tests use a fake Ollama: stream reassembly and parsing, the stats math, byte-for-byte
forwarding, fail-open behaviour, and, on real sockets, that chunks are passed on before
Ollama has sent the next one and that a caller hanging up closes the connection to
Ollama.

## Roadmap

- Retention settings (delete history older than N days).
- Which app is behind a gateway's requests (e.g. LiteLLM's `user` or key alias), so
  "who uses the models" shows apps instead of one gateway address.
- A React dashboard.
- Running as the default endpoint in front of Ollama for other gateways.

## License

MIT, see [LICENSE](LICENSE).
