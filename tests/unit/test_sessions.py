import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from claude_subscription_adapter.sessions import (
    NewSession,
    ResumeSession,
    SessionStore,
)


@pytest.mark.asyncio
async def test_a_first_request_for_a_key_gets_a_new_session():
    decision = await SessionStore().resolve_session("caller-a")

    assert isinstance(decision, NewSession)


@pytest.mark.asyncio
async def test_a_repeat_request_for_the_same_key_resumes_the_same_session():
    store = SessionStore()
    first = await store.resolve_session("caller-a")

    second = await store.resolve_session("caller-a")

    assert isinstance(second, ResumeSession)
    assert second.session_id == first.session_id


@pytest.mark.asyncio
async def test_different_keys_get_different_session_ids():
    store = SessionStore()

    first = await store.resolve_session("caller-a")
    second = await store.resolve_session("caller-b")

    assert first.session_id != second.session_id


@pytest.mark.asyncio
async def test_concurrent_first_requests_for_the_same_new_key_share_one_session_id():
    """Two near-simultaneous first turns for a brand-new key must not mint two UUIDs.

    If they did, one of the two turns would silently vanish from the CLI's own transcript.
    """
    store = SessionStore()

    results = await asyncio.gather(*(store.resolve_session("caller-a") for _ in range(10)))

    assert len({decision.session_id for decision in results}) == 1
    assert sum(isinstance(decision, NewSession) for decision in results) == 1


@pytest.mark.asyncio
async def test_touch_advances_last_touched_and_records_turns_sent_for_the_next_resume():
    store = SessionStore()
    await store.resolve_session("caller-a")

    await store.touch("caller-a", turns_sent=3)
    decision = await store.resolve_session("caller-a")

    assert isinstance(decision, ResumeSession)
    assert decision.turns_already_sent == 3


@pytest.mark.asyncio
async def test_a_prompt_that_extends_what_was_seen_resumes_but_a_changed_one_starts_fresh():
    """The CLI replays the first-turn prompt on resume, so a prompt whose earlier part changed must not resume."""
    store = SessionStore()
    first = await store.resolve_session("caller-a", system_prompt="base")

    extended = await store.resolve_session("caller-a", system_prompt="base + note")
    changed = await store.resolve_session("caller-a", system_prompt="different base")

    assert isinstance(extended, ResumeSession)
    assert extended.session_id == first.session_id
    assert isinstance(changed, NewSession)
    assert changed.session_id != first.session_id


@pytest.mark.asyncio
async def test_a_session_replaced_by_a_changed_prompt_has_its_transcript_deleted_on_the_next_sweep(tmp_path):
    """Live-observed: Claude Code's titling request gets a fresh session every turn; without this the replaced
    sessions' transcripts would pile up on the proxy server forever, since nothing tracks them any more."""
    store = SessionStore()
    replaced = await store.resolve_session("caller-a", system_prompt="title turn 1")
    await store.resolve_session("caller-a", system_prompt="title turn 2")
    transcript = tmp_path / f"{replaced.session_id}.jsonl"
    transcript.write_text("{}")

    await store.sweep(now=datetime.now(), inactivity_window=timedelta(hours=4), transcripts_dir=tmp_path)

    assert not transcript.exists()
    assert "caller-a" in store.snapshot()


@pytest.mark.asyncio
async def test_touch_on_an_unknown_key_does_not_raise_or_create_an_entry():
    store = SessionStore()

    await store.touch("never-seen", turns_sent=1)

    assert store.snapshot() == {}


@pytest.mark.asyncio
async def test_sweep_deletes_the_transcript_and_drops_the_tracking_entry_once_stale(tmp_path):
    store = SessionStore()
    decision = await store.resolve_session("caller-a")
    transcript = tmp_path / f"{decision.session_id}.jsonl"
    transcript.write_text("{}")

    deleted = await store.sweep(
        now=datetime.now() + timedelta(hours=5),
        inactivity_window=timedelta(hours=4),
        transcripts_dir=tmp_path,
    )

    assert deleted == ("caller-a",)
    assert not transcript.exists()
    assert store.snapshot() == {}


@pytest.mark.asyncio
async def test_sweep_leaves_a_session_touched_within_the_window(tmp_path):
    store = SessionStore()
    await store.resolve_session("caller-a")

    deleted = await store.sweep(
        now=datetime.now() + timedelta(hours=1),
        inactivity_window=timedelta(hours=4),
        transcripts_dir=tmp_path,
    )

    assert deleted == ()
    assert "caller-a" in store.snapshot()


@pytest.mark.asyncio
async def test_sweep_tolerates_a_transcript_file_that_is_already_gone(tmp_path):
    store = SessionStore()
    await store.resolve_session("caller-a")

    deleted = await store.sweep(
        now=datetime.now() + timedelta(hours=5),
        inactivity_window=timedelta(hours=4),
        transcripts_dir=tmp_path,
    )

    assert deleted == ("caller-a",)


@pytest.mark.asyncio
async def test_a_swept_key_starts_a_new_session_on_its_next_request(tmp_path):
    store = SessionStore()
    first = await store.resolve_session("caller-a")

    await store.sweep(
        now=datetime.now() + timedelta(hours=5),
        inactivity_window=timedelta(hours=4),
        transcripts_dir=tmp_path,
    )
    decision = await store.resolve_session("caller-a")

    assert isinstance(decision, NewSession)
    assert decision.session_id != first.session_id


def test_transcripts_dir_escapes_the_working_directory_to_match_the_real_cli_layout():
    from claude_subscription_adapter.transport import transcripts_dir

    result = transcripts_dir("/tmp/claude-code-cli")

    assert result == Path.home() / ".claude" / "projects" / "-tmp-claude-code-cli"
