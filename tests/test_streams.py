import json

import pytest

from lltop.streams import (LineSplitter, LineTooLong, merge_final, ollama_delta,
                                 openai_delta, sse_data)


def test_line_splitter_reassembles_lines_split_across_chunks():
    lines = [b'{"a": 1}', b'{"b": 2}', b'{"c": 3}']
    data = b"\n".join(lines) + b"\n"
    # every possible split point, plus byte-by-byte
    for cut in range(len(data)):
        sp = LineSplitter()
        got = sp.feed(data[:cut]) + sp.feed(data[cut:]) + sp.flush()
        assert got == lines, cut
    sp = LineSplitter()
    got = [line for i in range(len(data)) for line in sp.feed(data[i:i + 1])]
    assert got == lines


def test_line_splitter_crlf_blank_lines_and_flush():
    sp = LineSplitter()
    assert sp.feed(b"data: 1\r\n\r\ndata: 2") == [b"data: 1"]
    assert sp.flush() == [b"data: 2"]
    assert sp.flush() == []


def test_line_splitter_limit():
    sp = LineSplitter(max_line=10)
    with pytest.raises(LineTooLong):
        sp.feed(b"x" * 11)


def test_ollama_chat_chunks():
    d = ollama_delta({"message": {"role": "assistant", "content": "Hel"}, "done": False}, "chat")
    assert (d.content, d.final, d.has_output) == ("Hel", None, True)
    d = ollama_delta({"message": {"content": "", "thinking": "hmm"}}, "chat")
    assert d.thinking == "hmm" and d.has_output
    calls = [{"function": {"name": "get_time", "arguments": {}}}]
    assert ollama_delta({"message": {"content": "", "tool_calls": calls}}, "chat").tool_calls == calls
    done = ollama_delta({"message": {"content": ""}, "done": True, "done_reason": "stop",
                         "eval_count": 7, "eval_duration": 10, "load_duration": 3}, "chat")
    assert done.final == {"eval_count": 7, "eval_duration": 10, "load_duration": 3,
                          "done_reason": "stop"}
    assert not done.has_output


def test_ollama_generate_and_error():
    assert ollama_delta({"response": "Hi", "thinking": "t"}, "generate").content == "Hi"
    assert ollama_delta({"error": "model 'x' not found"}, "chat").error == "model 'x' not found"


def test_sse_and_openai_chunks():
    line = b'data: ' + json.dumps({"choices": [{"delta": {"content": "Hi"}}]}).encode()
    assert openai_delta(sse_data(line)).content == "Hi"
    assert sse_data(b"data: [DONE]") == "DONE"
    assert sse_data(b": keep-alive") is None
    d = openai_delta({"choices": [{"delta": {"reasoning": "think"}, "finish_reason": None}]})
    assert d.thinking == "think" and d.final is None
    end = openai_delta({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    usage = openai_delta({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}})
    final = merge_final(merge_final(None, end.final), usage.final)
    assert final == {"done_reason": "stop", "prompt_eval_count": 9, "eval_count": 4}


def test_openai_whole_reply_and_error():
    reply = {"choices": [{"message": {"content": "Done."}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    d = openai_delta(reply)
    assert d.content == "Done." and d.final["eval_count"] == 2
    assert openai_delta({"error": {"message": "bad request"}}).error == "bad request"
