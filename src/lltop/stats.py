"""Speed and timing numbers for one request. Pure functions, so they are easy to test.

Ollama's final chunk (the one with "done": true) reports durations in nanoseconds:

    prompt_eval_count / prompt_eval_duration   reading the prompt ("prefill")
    eval_count / eval_duration                 writing the answer ("decode")
    load_duration                              loading the model into memory
    total_duration                             all of the above, end to end

The proxy adds its own wall-clock times (time.monotonic(), in seconds): when the
request started, when the first token arrived, and when it ended.
"""

NS_PER_S = 1e9

# A warm model still reports a small load_duration (milliseconds). Loading a model
# from disk takes seconds, so anything at or above this counts as a cold load.
COLD_LOAD_SECONDS = 1.0

STAT_KEYS = ("prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
             "load_duration", "total_duration")


def ns_to_s(ns) -> float | None:
    return None if ns is None else ns / NS_PER_S


def per_second(count, duration_ns) -> float | None:
    """Tokens per second from a token count and a duration in nanoseconds."""
    if not count or not duration_ns or duration_ns <= 0:
        return None
    return count / (duration_ns / NS_PER_S)


def is_cold_load(load_s: float | None, was_loaded: bool | None) -> bool | None:
    """Did this request have to load the model first?

    load_s:     Ollama's load_duration in seconds (None if not reported, e.g. /v1).
    was_loaded: whether /api/ps listed the model when the request started
                (None if the poller has no data yet).
    Ollama's own number wins; None means "can't tell".
    """
    if load_s is not None:
        return load_s >= COLD_LOAD_SECONDS
    if was_loaded is not None:
        return not was_loaded
    return None


def summarize(final: dict | None, *, started: float, first_token: float | None,
              ended: float, chunks: int = 0, was_loaded: bool | None = None) -> dict:
    """All numbers for the dashboard's "finished" event.

    final:       Ollama's final stats (STAT_KEYS, any may be missing), or {}.
    chunks:      response chunks that carried text (about one token each when
                 streaming); used to estimate speed when Ollama gives no durations.
    """
    final = final or {}
    load_s = ns_to_s(final.get("load_duration"))
    out_rate = per_second(final.get("eval_count"), final.get("eval_duration"))
    rate_source = "ollama" if out_rate is not None else None
    if out_rate is None and first_token is not None and chunks > 1 and ended > first_token:
        # The first chunk marks t=0, so chunks-1 tokens came in (ended - first_token).
        out_rate = (chunks - 1) / (ended - first_token)
        rate_source = "chunks"
    return {
        "prompt_tokens": final.get("prompt_eval_count"),
        "output_tokens": final.get("eval_count"),
        "prompt_tokens_per_s": per_second(final.get("prompt_eval_count"),
                                          final.get("prompt_eval_duration")),
        "output_tokens_per_s": out_rate,
        "rate_source": rate_source,
        "load_s": load_s,
        "prompt_eval_s": ns_to_s(final.get("prompt_eval_duration")),
        "eval_s": ns_to_s(final.get("eval_duration")),
        "total_s": ns_to_s(final.get("total_duration")),
        "wall_s": ended - started,
        "ttft_s": None if first_token is None else first_token - started,
        "cold_load": is_cold_load(load_s, was_loaded),
        "chunks": chunks,
    }
