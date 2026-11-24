# ollama-lens

A live monitoring dashboard for [Ollama](https://ollama.com). Put it in front of Ollama
and watch every request as it happens: which model, who sent it, the prompt, the thinking
and the response appearing token by token, then the final speed and timing numbers.

It is a **streaming reverse proxy**. Clients talk to ollama-lens exactly as they would
talk to Ollama; every request is forwarded unchanged and every response chunk is passed
straight back. A copy of each chunk goes to the dashboard.

Useful when Ollama is slow (CPU-only machines, big models) and you want to know what it is
doing right now, which app is using it, and how fast it really is.

> Status: milestone 1. History is kept in memory (last 50 requests); persistent history
> comes later.

## What it shows

- **One card per model**: loaded or not, size, how much of it sits in GPU memory vs
  system RAM, context length, and a countdown to when Ollama unloads it (from `/api/ps`,
  polled every 3 s).
- **Requests in progress**, inside their model's card: endpoint, client address, elapsed
  time, phase (reading prompt / thinking / generating), the prompt (plus the whole
  conversation, tools and options), and the thinking and response text as it streams,
  with a live tokens/s estimate.
- **The last 50 finished requests**: prompt and output tokens, time to first token,
  prompt and output tokens/s, total time, cold or warm model load, status (finished,
  error, cancelled). Click a row for the full prompt, response and Ollama's raw stats.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a running Ollama.

```sh
git clone https://github.com/mrkbkr-17/ollama-lens.git
cd ollama-lens
uv run ollama-lens
```

Point a client at port **11435** instead of 11434:

```sh
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}]
}'
```

and open the dashboard at <http://127.0.0.1:11435/lens/>.

### Options

| Flag | Environment | Default | |
|---|---|---|---|
| `--host` | `LENS_HOST` | `127.0.0.1` | Address to listen on. `0.0.0.0` exposes it to your network. |
| `--port` | `LENS_PORT` | `11435` | |
| `--upstream` | `LENS_UPSTREAM` | `http://127.0.0.1:11434` | Where Ollama is. |
| `--poll` | `LENS_POLL` | `3` | Seconds between `/api/ps` polls. |
| `--log-level` | `LENS_LOG_LEVEL` | `info` | |

### Docker

```sh
docker build -t ollama-lens .
docker run --rm -p 127.0.0.1:11435:11435 \
  --add-host=host.docker.internal:host-gateway ollama-lens
```

The image listens on all interfaces inside the container and expects Ollama at
`http://host.docker.internal:11434` (set `LENS_UPSTREAM` otherwise, e.g.
`http://ollama:11434` when both run on the same Docker network).

## Privacy

The dashboard shows every prompt and response in full, to anyone who can open the page.
Nothing is written to disk and nothing leaves the machine, but by default ollama-lens
only listens on `127.0.0.1` for that reason. Only use `--host 0.0.0.0` on a network you
trust: there is no login.

## How it works

```
  client ──HTTP──▶ ollama-lens :11435 ──HTTP──▶ Ollama :11434
    ▲                 │   ▲                        │
    └──── chunks ◀────┤   └──────── chunks ◀───────┘
                      │ copy of each chunk
                      ▼
                    Tap ──events──▶ Hub ──WebSocket──▶ dashboard (/lens/)
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
- **Hub** (`hub.py`): the in-memory state (loaded models, requests in progress, last 50)
  and the fan-out to browsers. Each browser gets its own bounded queue; one that falls
  behind is disconnected and reconnects to a fresh snapshot.

### Design rules

- **Fail open.** Inspection can never break a request: tap methods catch their own
  errors and the proxy wraps every call to them again. If the dashboard code fails, the
  request still reaches Ollama and returns normally; the dashboard marks it as an error.
- **No added latency.** Chunks go to the caller before anything else looks at them, and
  publishing an event never waits for a browser.
- **Cancellation passes through.** When a caller hangs up, ollama-lens closes its
  connection to Ollama, so Ollama stops generating, also while still waiting for the
  first token.
- **Transparent.** Everything outside `/lens/` belongs to Ollama, including `/`.

### Endpoints of its own

| Path | |
|---|---|
| `/lens/` | The dashboard |
| `/lens/ws` | WebSocket: a `snapshot` on connect, then every event |
| `/lens/api/state` | The snapshot as JSON |

## Development

```sh
uv run pytest
```

The tests use a fake Ollama: stream reassembly and parsing, the stats math, byte-for-byte
forwarding, fail-open behaviour, and, on real sockets, that chunks are passed on before
Ollama has sent the next one and that a caller hanging up closes the connection to
Ollama.

## Roadmap

- Persistent history (PostgreSQL) and charts over time.
- A React dashboard.
- Running as the default endpoint in front of Ollama for other gateways.

## License

MIT, see [LICENSE](LICENSE).
