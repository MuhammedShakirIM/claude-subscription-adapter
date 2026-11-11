from typing import get_args

import pytest

from claude_subscription_adapter.errors import (
    CLIFailure,
    CLINotAuthenticated,
    CLINotInstalled,
    PromptTooLong,
    QueueFull,
    RateLimited,
    TimedOut,
    UnexpectedExit,
    classify_failure,
    raise_public,
)

_STATUS_BY_KIND = {
    "CLINotInstalled": 503,
    "CLINotAuthenticated": 503,
    "RateLimited": 429,
    "PromptTooLong": 400,
    "TimedOut": 504,
    "QueueFull": 429,
    "UnexpectedExit": 502,
}


def test_spawn_filenotfound_classifies_as_not_installed():
    failure = classify_failure(exit_code=127, stderr="", spawn_error=FileNotFoundError(), executable="/opt/claude")

    assert failure == CLINotInstalled(executable="/opt/claude")


@pytest.mark.parametrize(
    "stderr",
    ["spawn claude ENOENT", "spawn claude enoent", "The system cannot find the file specified"],
)
def test_missing_binary_text_classifies_as_not_installed(stderr):
    assert classify_failure(exit_code=127, stderr=stderr) == CLINotInstalled(executable="claude")


def test_permission_error_on_spawn_keeps_its_text():
    failure = classify_failure(exit_code=1, stderr="", spawn_error=PermissionError(13, "Permission denied"))

    assert isinstance(failure, UnexpectedExit)
    assert "Permission denied" in failure.stderr


@pytest.mark.parametrize(
    "stderr",
    ["Please run /login", "Authentication error: token expired", "401 Unauthorized", "Invalid API key"],
)
def test_login_and_auth_text_classifies_as_not_authenticated(stderr):
    failure = classify_failure(exit_code=1, stderr=stderr)

    assert isinstance(failure, CLINotAuthenticated)
    assert stderr in failure.detail


@pytest.mark.parametrize(
    "stderr", ["authenticated successfully, but disk full", "wrote /var/log/login.log", "deauthenticate failed"]
)
def test_incidental_auth_words_do_not_classify_as_not_authenticated(stderr):
    assert isinstance(classify_failure(exit_code=1, stderr=stderr), UnexpectedExit)


def test_auth_failure_reported_only_in_events_is_classified():
    failure = classify_failure(exit_code=1, stderr="", stdout_events=({"error": "Authentication failed"},))

    assert isinstance(failure, CLINotAuthenticated)
    assert "Authentication failed" in failure.detail


@pytest.mark.parametrize("status", [429, "429"])
def test_api_error_status_429_classifies_as_rate_limited_for_int_or_string(status):
    failure = classify_failure(exit_code=1, stderr="", stdout_events=({"api_error_status": status},))

    assert failure == RateLimited(detail="")


def test_api_error_status_429_classifies_as_rate_limited():
    failure = classify_failure(
        exit_code=1, stderr="", stdout_events=({"api_error_status": 429, "result": "slow down"},)
    )

    assert failure == RateLimited(detail="slow down")


def test_rate_limit_text_in_events_classifies_as_rate_limited():
    failure = classify_failure(exit_code=1, stderr="", stdout_events=({"error": "rate_limit exceeded"},))

    assert isinstance(failure, RateLimited)
    assert "rate_limit exceeded" in failure.detail


def test_rate_limit_text_in_stderr_classifies_as_rate_limited():
    failure = classify_failure(exit_code=1, stderr="Error: rate limit reached")

    assert isinstance(failure, RateLimited)
    assert "rate limit reached" in failure.detail


def test_prompt_too_long_text_classifies_as_prompt_too_long():
    failure = classify_failure(exit_code=1, stderr="Error: prompt is too long")

    assert isinstance(failure, PromptTooLong)
    assert "prompt is too long" in failure.detail


def test_prompt_too_long_found_in_events_carries_the_event_text():
    failure = classify_failure(
        exit_code=1,
        stderr="Node warning: deprecated flag",
        stdout_events=({"result": "API Error: prompt is too long: 210000 tokens"},),
    )

    assert isinstance(failure, PromptTooLong)
    assert "210000 tokens" in failure.detail
    assert "deprecated flag" not in failure.detail


def test_long_prompt_wins_over_rate_limit_because_it_is_not_retryable():
    failure = classify_failure(exit_code=1, stderr="rate limit hit; prompt is too long")

    assert isinstance(failure, PromptTooLong)


def test_unmatched_failure_falls_through_to_unexpected_exit():
    assert classify_failure(exit_code=3, stderr="something odd") == UnexpectedExit(exit_code=3, stderr="something odd")


def test_not_installed_wins_over_auth_text():
    assert isinstance(classify_failure(exit_code=127, stderr="ENOENT while checking login"), CLINotInstalled)


def test_missing_cli_says_proxy_server_with_500():
    error = raise_public(CLINotInstalled(executable="claude"))

    assert error.status_code == 503
    assert "not installed on the proxy server" in error.message
    assert "claude" in error.message


def test_unauthenticated_says_proxy_server_and_names_claude_with_401():
    error = raise_public(CLINotAuthenticated(detail=""))

    assert error.status_code == 503
    assert "not authenticated on the proxy server" in error.message
    assert "`claude`" in error.message


def test_unauthenticated_includes_upstream_detail_when_present():
    assert "token expired" in raise_public(CLINotAuthenticated(detail="token expired")).message


def test_rate_limit_preserves_upstream_message_with_429():
    error = raise_public(RateLimited(detail="Rate limit reached. Retry after 60s."))

    assert error.status_code == 429
    assert "Retry after 60s." in error.message


def test_prompt_too_long_identifies_the_prompt_with_400():
    error = raise_public(PromptTooLong(detail="210000 tokens"))

    assert error.status_code == 400
    assert "too long" in error.message
    assert "210000 tokens" in error.message


def test_timeout_names_the_duration_with_504():
    error = raise_public(TimedOut(seconds=600))

    assert error.status_code == 504
    assert "600s" in error.message
    assert "timed out" in error.message


def test_a_full_line_says_the_proxy_is_busy_and_when_to_retry_with_429():
    error = raise_public(QueueFull(waiting=20, retry_after_seconds=45))

    assert error.status_code == 429
    assert "busy on the proxy server" in error.message
    assert "20 requests" in error.message
    assert "45s" in error.message
    assert error.num_retries == 0


def test_unexpected_exit_includes_code_and_trimmed_stderr_with_500():
    error = raise_public(UnexpectedExit(exit_code=2, stderr="  boom  "))

    assert error.status_code == 502
    assert "code 2" in error.message
    assert "boom" in error.message


@pytest.mark.parametrize(
    "failure",
    [
        UnexpectedExit(exit_code=1, stderr="\u00e9" * 5000),
        CLINotAuthenticated(detail="\u00e9" * 5000),
        RateLimited(detail="\u00e9" * 5000),
        PromptTooLong(detail="\u00e9" * 5000),
    ],
)
def test_oversized_detail_is_trimmed_on_every_path(failure):
    assert raise_public(failure).message.count("\u00e9") == 2000, "detail must be trimmed to the 2000-char limit"


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), -5])
def test_nonsense_timeout_duration_does_not_reach_the_operator(seconds):
    assert raise_public(TimedOut(seconds=seconds)).message == "Claude Code CLI timed out after 0s on the proxy server."


def test_fractional_and_large_timeouts_read_plainly():
    assert "after 1.5s" in raise_public(TimedOut(seconds=1.5)).message
    assert "after 10000000s" in raise_public(TimedOut(seconds=1e7)).message


def test_duplicate_event_values_are_not_repeated_in_the_detail():
    failure = classify_failure(
        exit_code=1, stderr="", stdout_events=({"result": "rate limit hit", "message": "rate limit hit"},)
    )

    assert failure == RateLimited(detail="rate limit hit")


def test_non_string_event_values_do_not_leak_into_the_detail():
    failure = classify_failure(
        exit_code=1, stderr="", stdout_events=({"api_error_status": 429, "error": {"code": 429}},)
    )

    assert isinstance(failure, RateLimited)
    assert "{" not in failure.detail


def test_deployment_problems_get_distinct_messages_and_codes():
    missing = raise_public(CLINotInstalled(executable="claude"))
    unauthenticated = raise_public(CLINotAuthenticated(detail=""))

    assert missing.message != unauthenticated.message
    assert (missing.status_code, unauthenticated.status_code) == (503, 503)


def test_every_failure_kind_in_the_union_is_covered_here():
    assert {kind.__name__ for kind in get_args(CLIFailure)} == set(_STATUS_BY_KIND)


def test_every_failure_kind_maps_to_its_specified_status():
    mapped = {
        type(failure).__name__: raise_public(failure).status_code
        for failure in (
            CLINotInstalled(executable="claude"),
            CLINotAuthenticated(detail=""),
            RateLimited(detail="x"),
            PromptTooLong(detail="x"),
            TimedOut(seconds=1),
            QueueFull(waiting=1, retry_after_seconds=1),
            UnexpectedExit(exit_code=1, stderr="x"),
        )
    }

    assert mapped == _STATUS_BY_KIND
    assert mapped == {
        "CLINotInstalled": 503,
        "CLINotAuthenticated": 503,
        "RateLimited": 429,
        "PromptTooLong": 400,
        "TimedOut": 504,
        "QueueFull": 429,
        "UnexpectedExit": 502,
    }


@pytest.mark.parametrize(
    "stderr",
    [
        "login required",
        "login failed",
        "You must log in first",
        "not logged in",
        "Not authenticated",
        "authentication token missing",
        "credentials expired",
    ],
)
def test_real_auth_wordings_are_not_missed(stderr):
    failure = classify_failure(exit_code=1, stderr=stderr)

    assert isinstance(failure, CLINotAuthenticated)
    assert raise_public(failure).status_code == 503


@pytest.mark.parametrize(
    "stderr",
    ["/bin/sh: claude: command not found", "claude: not found", "executable file not found in $PATH"],
)
def test_shell_and_container_missing_binary_wordings_classify_as_not_installed(stderr):
    assert classify_failure(exit_code=127, stderr=stderr) == CLINotInstalled(executable="claude")


def test_missing_workspace_file_is_not_reported_as_a_missing_cli():
    failure = classify_failure(exit_code=1, stderr="Error: no such file or directory: /home/x/data.json")

    assert isinstance(failure, UnexpectedExit)


def test_missing_binary_no_such_file_wording_still_classifies_when_it_names_the_executable():
    assert classify_failure(exit_code=127, stderr="claude: no such file or directory") == CLINotInstalled(
        executable="claude"
    )


def test_auth_failure_wins_over_a_rate_limit_mentioned_in_the_same_output():
    failure = classify_failure(exit_code=1, stderr="Authentication error: rate limit")

    assert isinstance(failure, CLINotAuthenticated)


@pytest.mark.parametrize("stderr", ["HTTP 429 Too Many Requests", "too many requests, slow down"])
def test_rate_limit_recognised_from_status_text_without_an_event(stderr):
    assert isinstance(classify_failure(exit_code=1, stderr=stderr), RateLimited)


@pytest.mark.parametrize("stderr", ["prompt too long", "context length exceeded"])
def test_overflow_wordings_beyond_the_canonical_phrase_classify_as_prompt_too_long(stderr):
    assert isinstance(classify_failure(exit_code=1, stderr=stderr), PromptTooLong)


def test_prompt_too_long_token_in_an_event_classifies_as_prompt_too_long():
    failure = classify_failure(exit_code=1, stderr="", stdout_events=({"error": "prompt_too_long"},))

    assert isinstance(failure, PromptTooLong)


@pytest.mark.parametrize("key", ["result", "message", "error"])
def test_each_detail_key_alone_carries_the_event_text(key):
    failure = classify_failure(exit_code=1, stderr="", stdout_events=({key: "rate limit exceeded"},))

    assert failure == RateLimited(detail="rate limit exceeded")


def test_empty_first_rate_limit_event_does_not_discard_a_later_detail():
    failure = classify_failure(
        exit_code=1,
        stderr="",
        stdout_events=({"api_error_status": 429}, {"api_error_status": 429, "result": "retry in 60s"}),
    )

    assert failure == RateLimited(detail="retry in 60s")


def test_non_mapping_events_do_not_crash_the_classifier():
    failure = classify_failure(exit_code=1, stderr="", stdout_events=(None, "junk", {"result": "rate limit"}))

    assert failure == RateLimited(detail="rate limit")


def test_control_characters_and_ansi_escapes_do_not_reach_the_message():
    error = raise_public(classify_failure(exit_code=1, stderr="rate limit\r\n\x1b[31mred\x00"))

    assert "\x1b" not in error.message
    assert "\r" not in error.message and "\n" not in error.message


def test_stderr_is_trimmed_on_the_failure_value_not_only_in_the_message():
    failure = classify_failure(exit_code=1, stderr="x" * 50_000)

    assert isinstance(failure, UnexpectedExit)
    assert len(failure.stderr) < 2_100


def test_truncated_detail_is_marked_so_the_operator_knows_it_was_cut():
    error = raise_public(UnexpectedExit(exit_code=1, stderr="y" * 5_000))

    assert error.message.endswith("...")


def test_surrounding_whitespace_is_removed_from_the_reported_stderr():
    error = raise_public(UnexpectedExit(exit_code=2, stderr="  boom  "))

    assert error.message.endswith("boom")


def test_running_out_of_turns_reports_the_clis_own_reason_instead_of_an_empty_message():
    """The CLI exits 1 with nothing on stderr; its `result` event's `errors` list is the only explanation."""
    failure = classify_failure(
        exit_code=1,
        stderr="",
        stdout_events=(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
                "result": None,
                "errors": ["Reached maximum number of turns (10)"],
            },
        ),
    )

    assert failure == UnexpectedExit(exit_code=1, stderr="Reached maximum number of turns (10)")
    assert "Reached maximum number of turns (10)" in raise_public(failure).message
