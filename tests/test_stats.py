import pytest

from lltop.stats import is_cold_load, per_second, summarize


def test_per_second():
    assert per_second(100, 10 * 1e9) == pytest.approx(10.0)    # 100 tokens in 10 s
    assert per_second(0, 1e9) is None
    assert per_second(5, 0) is None
    assert per_second(None, None) is None


@pytest.mark.parametrize("load_s, was_loaded, expected", [
    (0.012, None, False),   # warm: Ollama reports a few ms
    (8.5, True, True),      # Ollama's number wins over the poller
    (1.0, None, True),      # threshold is inclusive
    (None, False, True),    # no Ollama stats (/v1): fall back to /api/ps
    (None, True, False),
    (None, None, None),     # can't tell
])
def test_cold_load(load_s, was_loaded, expected):
    assert is_cold_load(load_s, was_loaded) is expected


def test_summarize_with_ollama_stats():
    final = {"prompt_eval_count": 40, "prompt_eval_duration": 2e9,     # 20 tok/s
             "eval_count": 30, "eval_duration": 5e9,                    # 6 tok/s
             "load_duration": 9e9, "total_duration": 16e9}
    s = summarize(final, started=100.0, first_token=111.5, ended=117.0, chunks=30)
    assert s["prompt_tokens_per_s"] == pytest.approx(20.0)
    assert s["output_tokens_per_s"] == pytest.approx(6.0)
    assert s["rate_source"] == "ollama"
    assert s["ttft_s"] == pytest.approx(11.5)
    assert s["wall_s"] == pytest.approx(17.0)
    assert s["load_s"] == pytest.approx(9.0)
    assert s["total_s"] == pytest.approx(16.0)
    assert s["cold_load"] is True


def test_summarize_estimates_rate_from_chunks():
    # /v1 streaming: no durations. 11 chunks: the first at t=2, the last at t=7,
    # so 10 tokens in 5 s.
    s = summarize({"eval_count": 11}, started=0.0, first_token=2.0, ended=7.0,
                  chunks=11, was_loaded=True)
    assert s["output_tokens_per_s"] == pytest.approx(2.0)
    assert s["rate_source"] == "chunks"
    assert s["cold_load"] is False


def test_summarize_without_output():
    s = summarize(None, started=0.0, first_token=None, ended=3.0)
    assert s["ttft_s"] is None
    assert s["output_tokens_per_s"] is None
    assert s["rate_source"] is None
    assert s["wall_s"] == pytest.approx(3.0)
