import json
from pathlib import Path

import pytest

from claude_subscription_adapter.streaming import (
    DEFAULT_FINISH_REASON,
    StreamState,
    advance,
    assemble_answer,
    close_stream,
    stream_chunks,
)
from claude_subscription_adapter.tools import read_client_tools

_FIXTURES = Path(__file__).parents[1] / "fixtures"


def _events(name):
    """Every conversion test reads a run recorded from the real CLI, so no test invents CLI output."""
    lines = (_FIXTURES / name).read_text(encoding="utf-8").splitlines()
    return tuple(json.loads(line) for line in lines if line.strip())


DELTA_RUN = _events("delta_run.ndjson")
SNAPSHOT_RUN = _events("snapshot_run.ndjson")
EMPTY_RUN = _events("empty_run.ndjson")
SEARCH_RUN = _events("search_run.ndjson")


def _text_deltas(events):
    return tuple(
        event["event"]["delta"]["text"]
        for event in events
        if event.get("type") == "stream_event"
        and event["event"].get("type") == "content_block_delta"
        and event["event"]["delta"].get("type") == "text_delta"
    )


def _snapshot_text(events):
    for event in events:
        if event.get("type") == "assistant":
            text = "".join(b.get("text", "") for b in event["message"]["content"] if b.get("type") == "text")
            if text:
                return text
    return ""


def _result_usage(events):
    return next(event["usage"] for event in events if event.get("type") == "result")


def _counts(usage):
    """The three headline counts, without the cache detail keys each closing chunk also carries."""
    return {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}


def test_the_recorded_delta_run_really_contains_several_text_deltas():
    """The dedup and ordering tests are only meaningful if the fixture actually streamed deltas."""
    assert len(_text_deltas(DELTA_RUN)) > 1


def test_the_recorded_snapshot_run_has_no_deltas_at_all():
    assert _text_deltas(SNAPSHOT_RUN) == ()


def test_each_text_delta_becomes_one_unfinished_chunk_in_arrival_order():
    chunks = stream_chunks(DELTA_RUN)
    deltas = _text_deltas(DELTA_RUN)

    streamed = tuple(chunk["text"] for chunk in chunks if not chunk["is_finished"])
    assert streamed == deltas
    assert all(not chunk["is_finished"] and chunk["usage"] is None for chunk in chunks[:-1])


def test_thinking_and_signature_deltas_never_reach_the_answer():
    """The fixture streams a thinking block before the text; leaking it would show the user private reasoning."""
    thinking = tuple(
        event["event"]["delta"]["thinking"]
        for event in DELTA_RUN
        if event.get("type") == "stream_event"
        and event["event"].get("type") == "content_block_delta"
        and event["event"]["delta"].get("type") == "thinking_delta"
    )
    text = assemble_answer(DELTA_RUN).text

    assert thinking, "the recorded run must contain thinking deltas for this to prove anything"
    assert text == "".join(_text_deltas(DELTA_RUN))
    assert not any(chunk in text for chunk in thinking)


@pytest.mark.parametrize("delta_type", ["thinking_delta", "signature_delta", "input_json_delta"])
def test_only_a_text_delta_contributes_to_the_answer(delta_type):
    """A non-text delta carrying a `text` key must still be ignored; the block type is what decides."""
    event = {
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"type": delta_type, "text": "leaked"}},
    }

    assert stream_chunks((event,)) == stream_chunks(())
    assert assemble_answer((event,)).text == ""


def test_closing_chunk_is_finished_and_carries_the_recorded_usage():
    closing = stream_chunks(DELTA_RUN)[-1]
    recorded = _result_usage(DELTA_RUN)

    assert closing["is_finished"] is True
    assert closing["text"] == ""
    assert closing["finish_reason"] == "stop"
    assert closing["usage"] is not None
    assert closing["usage"]["completion_tokens"] == recorded["output_tokens"]
    assert closing["usage"]["prompt_tokens"] == (
        recorded["input_tokens"] + recorded["cache_creation_input_tokens"] + recorded["cache_read_input_tokens"]
    )


def test_only_the_closing_chunk_carries_usage():
    chunks = stream_chunks(DELTA_RUN)

    assert [chunk["usage"] is not None for chunk in chunks] == [False] * (len(chunks) - 1) + [True]


def test_exactly_one_chunk_is_marked_finished():
    assert sum(chunk["is_finished"] for chunk in stream_chunks(DELTA_RUN)) == 1


def test_a_snapshot_only_run_emits_the_snapshot_text_then_the_close():
    chunks = stream_chunks(SNAPSHOT_RUN)
    expected = _snapshot_text(SNAPSHOT_RUN)

    assert expected
    assert tuple(chunk["text"] for chunk in chunks if not chunk["is_finished"]) == (expected,)
    assert chunks[-1]["is_finished"]


def test_a_snapshot_after_deltas_does_not_double_the_answer():
    """The recorded delta run ends with a full assistant snapshot of the same text."""
    joined = "".join(_text_deltas(DELTA_RUN))

    assert _snapshot_text(DELTA_RUN) == joined
    assert assemble_answer(DELTA_RUN).text == joined
    assert assemble_answer(DELTA_RUN).text.count(joined) == 1


@pytest.mark.parametrize("events", [DELTA_RUN, SNAPSHOT_RUN, EMPTY_RUN])
def test_the_assembled_answer_equals_the_joined_streamed_text_with_the_same_usage(events):
    chunks = stream_chunks(events)
    answer = assemble_answer(events)

    assert answer.text == "".join(chunk["text"] for chunk in chunks)
    assert answer.finish_reason == chunks[-1]["finish_reason"]
    assert chunks[-1]["usage"] is not None
    assert answer.usage.prompt_tokens == chunks[-1]["usage"]["prompt_tokens"]
    assert answer.usage.completion_tokens == chunks[-1]["usage"]["completion_tokens"]
    assert answer.usage.total_tokens == chunks[-1]["usage"]["total_tokens"]


def test_an_empty_run_still_closes_with_empty_text_and_zero_usage():
    chunks = stream_chunks(())

    assert len(chunks) == 1
    assert chunks[0]["text"] == ""
    assert chunks[0]["is_finished"] is True
    assert chunks[0]["finish_reason"] == DEFAULT_FINISH_REASON
    assert _counts(chunks[0]["usage"]) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_the_committed_empty_fixture_converts_to_a_single_closing_chunk():
    assert EMPTY_RUN == ()
    assert len(stream_chunks(EMPTY_RUN)) == 1


def _delta(text):
    return {
        "type": "stream_event",
        "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
    }


def _message_delta(*, stop_reason=None, usage=None):
    inner = {"type": "message_delta", "delta": {"stop_reason": stop_reason}}
    return {"type": "stream_event", "event": inner if usage is None else {**inner, "usage": usage}}


def _result(*, stop_reason=None, usage=None):
    event = {"type": "result", "subtype": "success"}
    return {
        **event,
        **({} if stop_reason is None else {"stop_reason": stop_reason}),
        **({} if usage is None else {"usage": usage}),
    }


def test_a_terminal_event_with_no_usage_still_closes_with_zero_counts():
    chunks = stream_chunks((_delta("hi"), _result()))

    assert chunks[0]["text"] == "hi"
    assert chunks[-1]["is_finished"] is True
    assert _counts(chunks[-1]["usage"]) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_a_malformed_usage_payload_is_coerced_rather_than_raising():
    chunks = stream_chunks((_result(usage={"input_tokens": "nope", "output_tokens": None}),))

    assert _counts(chunks[-1]["usage"]) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.parametrize(
    "event",
    [
        {"type": "system", "subtype": "init"},
        {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}},
        {"type": "stream_event", "event": {"type": "message_stop"}},
        {"type": "stream_event", "event": {"type": "content_block_start", "content_block": {"type": "text"}}},
        {"type": "brand_new_event_type_from_a_future_cli"},
        {"type": "assistant", "message": {"content": [{"type": "thinking", "thinking": "private"}]}},
    ],
)
def test_an_event_the_mapping_does_not_handle_is_skipped_without_failing(event):
    assert stream_chunks((event,)) == stream_chunks(())


def test_a_result_usage_beats_a_differing_message_delta_usage():
    """Both payloads carry real counts in a live run, so the winner must be pinned, not left to ordering."""
    events = (
        _delta("hi"),
        _message_delta(stop_reason="end_turn", usage={"input_tokens": 1, "output_tokens": 2}),
        _result(stop_reason="end_turn", usage={"input_tokens": 100, "output_tokens": 200}),
    )

    closing = stream_chunks(events)[-1]

    assert _counts(closing["usage"]) == {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}


def test_a_message_delta_usage_is_used_when_the_result_carries_none():
    events = (
        _message_delta(stop_reason="end_turn", usage={"input_tokens": 1, "output_tokens": 2}),
        _result(stop_reason="end_turn"),
    )

    assert _counts(stream_chunks(events)[-1]["usage"]) == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }


def test_a_message_delta_usage_beats_an_assistant_snapshot_usage():
    """The snapshot reports only the tokens emitted so far; taking it would under-report the whole turn."""
    events = (
        _message_delta(usage={"input_tokens": 10, "output_tokens": 500}),
        {"type": "assistant", "message": {"content": [], "usage": {"input_tokens": 10, "output_tokens": 7}}},
    )

    assert stream_chunks(events)[-1]["usage"]["completion_tokens"] == 500


def test_an_assistant_snapshot_usage_is_the_last_resort():
    events = ({"type": "assistant", "message": {"content": [], "usage": {"input_tokens": 4, "output_tokens": 6}}},)

    assert _counts(stream_chunks(events)[-1]["usage"]) == {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
    }


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("refusal", "content_filter"),
        ("tool_use", "tool_calls"),
        ("something_a_future_cli_invents", "stop"),
    ],
)
def test_the_closing_chunk_maps_the_cli_stop_reason(stop_reason, expected):
    events = (_delta("hi"), _result(stop_reason=stop_reason))

    assert stream_chunks(events)[-1]["finish_reason"] == expected
    assert assemble_answer(events).finish_reason == expected


def test_a_message_delta_stop_reason_is_used_when_the_result_omits_it():
    events = (_message_delta(stop_reason="max_tokens"), _result())

    assert stream_chunks(events)[-1]["finish_reason"] == "length"


def test_a_run_with_no_stop_reason_at_all_closes_as_a_plain_stop():
    assert stream_chunks((_delta("hi"),))[-1]["finish_reason"] == DEFAULT_FINISH_REASON


def test_advancing_event_by_event_matches_converting_the_whole_run_at_once():
    """The incremental path story 1.5 drives must not drift from the eager one."""
    state = StreamState()
    incremental = []
    for event in DELTA_RUN:
        state, emitted = advance(state, event)
        incremental.extend(emitted)
    incremental.append(close_stream(state))

    assert tuple(incremental) == stream_chunks(DELTA_RUN)


def test_the_fold_accepts_a_one_shot_iterator():
    """A live stream can only be walked once, so a second internal pass would silently lose the answer."""
    assert assemble_answer(iter(DELTA_RUN)).text == assemble_answer(DELTA_RUN).text


def _progress_kinds(chunks):
    return tuple(
        chunk["provider_specific_fields"]["claude_code_cli_web_search"]
        for chunk in chunks
        if chunk.get("provider_specific_fields")
    )


def test_a_search_run_emits_started_then_results_progress_before_any_answer_text():
    chunks = stream_chunks(SEARCH_RUN)
    non_progress_text = tuple(
        chunk["text"] for chunk in chunks if chunk["text"] and not chunk.get("provider_specific_fields")
    )

    assert _progress_kinds(chunks) == ("started", "results")
    progress_index = next(i for i, c in enumerate(chunks) if c.get("provider_specific_fields"))
    first_text_index = next(i for i, c in enumerate(chunks) if c["text"])
    assert progress_index < first_text_index
    assert non_progress_text
    assert not any(kind in text for kind in ("started", "results") for text in non_progress_text)


def test_progress_chunks_carry_no_text_and_are_not_marked_finished():
    chunks = stream_chunks(SEARCH_RUN)
    progress_chunks = [c for c in chunks if c.get("provider_specific_fields")]

    assert progress_chunks
    assert all(chunk["text"] == "" and chunk["is_finished"] is False for chunk in progress_chunks)


def test_a_run_with_no_search_emits_no_progress_chunks():
    assert _progress_kinds(stream_chunks(DELTA_RUN)) == ()
    assert _progress_kinds(stream_chunks(SNAPSHOT_RUN)) == ()
    assert _progress_kinds(stream_chunks(EMPTY_RUN)) == ()


def test_assemble_answer_on_the_search_run_has_no_progress_content_mixed_into_text():
    answer = assemble_answer(SEARCH_RUN)

    assert answer.text == "Example headline one is the current top story."
    assert "started" not in answer.text
    assert "results" not in answer.text


def test_an_unmatched_tool_result_produces_no_results_chunk_and_does_not_crash():
    event = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"tool_use_id": "toolu_never_seen", "type": "tool_result", "content": "x"}],
        },
    }

    assert stream_chunks((event,)) == stream_chunks(())


def test_a_tool_result_for_an_unstarted_search_id_is_ignored_even_within_a_real_search_run():
    """The fixture's first tool_result belongs to a non-search tool; it must never be mistaken for a search result."""
    unmatched = SEARCH_RUN[1]
    assert unmatched["type"] == "user"

    assert _progress_kinds(stream_chunks((unmatched,))) == ()


def test_the_closing_chunk_carries_the_cache_counts_the_run_reported():
    """Streamed spend is wrong without these: prompt caching is where this provider's cost lives."""
    closing = stream_chunks(DELTA_RUN)[-1]
    recorded = _result_usage(DELTA_RUN)
    answer = assemble_answer(DELTA_RUN)

    assert recorded["cache_read_input_tokens"] > 0
    assert recorded["cache_creation_input_tokens"] > 0
    assert closing["usage"]["cache_read_input_tokens"] == recorded["cache_read_input_tokens"]
    assert closing["usage"]["cache_creation_input_tokens"] == recorded["cache_creation_input_tokens"]
    assert closing["usage"]["prompt_tokens_details"]["cached_tokens"] == recorded["cache_read_input_tokens"]
    assert closing["usage"]["prompt_tokens_details"]["cache_write_tokens"] == recorded["cache_creation_input_tokens"]
    assert answer.usage.cache_read_input_tokens == closing["usage"]["cache_read_input_tokens"]
    assert answer.usage.cache_creation_input_tokens == closing["usage"]["cache_creation_input_tokens"]


CLIENT_TOOL_RUN = _events("client_tool_run.ndjson")


def _function(name):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def _renamed(events, cli_name):
    return tuple(json.loads(json.dumps(event).replace("mcp__c__create_file", cli_name)) for event in events)


def test_a_long_caller_tool_name_comes_back_under_its_own_name():
    name = "mcp_some_server_with_a_really_long_name_create_pull_request_review"
    tools = read_client_tools([_function(name)], None)

    answer = assemble_answer(_renamed(CLIENT_TOOL_RUN, f"mcp__c__{tools[0].alias}"), tools)

    assert len(f"mcp__c__{tools[0].alias}") <= 64
    assert [call.name for call in answer.tool_calls] == [name, name]


def test_a_web_search_in_the_same_turn_is_not_handed_to_the_caller():
    tools = read_client_tools([_function("create_file")], None)
    search_snapshot = {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "id": "toolu_search", "name": "WebSearch", "input": {}}]},
    }

    answer = assemble_answer((search_snapshot, *CLIENT_TOOL_RUN), tools)

    assert [call.id for call in answer.tool_calls] == [
        "toolu_01CLIENTTOOLPLACEHOLDER01",
        "toolu_01CLIENTTOOLPLACEHOLDER02",
    ]


def test_the_turn_is_only_handed_back_once_its_message_delta_arrives():
    """Stopping at the first call would drop the parallel second one and the turn's usage."""
    tools = read_client_tools([_function("create_file")], None)
    state = StreamState(client_tools=tools)
    ready_at = []
    for position, event in enumerate(CLIENT_TOOL_RUN):
        state, _ = advance(state, event)
        if state.awaiting_caller:
            ready_at.append(position)

    delta_position = next(i for i, e in enumerate(CLIENT_TOOL_RUN) if e.get("event", {}).get("type") == "message_delta")
    assert ready_at[0] == delta_position
    assert len(state.tool_calls) == 2


def test_a_tool_use_stop_with_no_caller_calls_does_not_hand_back():
    """A turn that stopped for WebSearch alone must keep running inside the CLI."""
    state, _ = advance(
        StreamState(client_tools=read_client_tools([_function("create_file")], None)),
        {"type": "stream_event", "event": {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}},
    )

    assert not state.awaiting_caller
