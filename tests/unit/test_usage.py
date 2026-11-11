import pytest

from claude_subscription_adapter.usage import read_cli_usage


def test_full_payload_reports_all_four_counts():
    usage = read_cli_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 5,
        }
    )

    assert usage.prompt_tokens == 135
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 155
    assert usage.cache_creation_input_tokens == 30
    assert usage.cache_read_input_tokens == 5
    assert usage.prompt_tokens_details.cache_creation_tokens == 30
    assert usage.prompt_tokens_details.cached_tokens == 5
    assert usage.prompt_tokens_details.text_tokens == 100


def test_prompt_tokens_include_cache_counts():
    usage = read_cli_usage(
        {
            "input_tokens": 10,
            "output_tokens": 1,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 3,
        }
    )

    assert usage.prompt_tokens == 20
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


def test_missing_cache_fields_read_as_zero():
    usage = read_cli_usage({"input_tokens": 50, "output_tokens": 8})

    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 8
    assert usage.cache_creation_input_tokens == 0
    assert usage.cache_read_input_tokens == 0


def test_empty_payload_yields_all_zero():
    usage = read_cli_usage({})

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.cache_creation_input_tokens == 0
    assert usage.cache_read_input_tokens == 0


@pytest.mark.parametrize("bad", ["twelve", None, {}, [], True, False])
def test_malformed_value_reads_as_zero_without_raising(bad):
    usage = read_cli_usage(
        {
            "input_tokens": bad,
            "output_tokens": 5,
            "cache_creation_input_tokens": bad,
            "cache_read_input_tokens": bad,
        }
    )

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 5
    assert usage.cache_creation_input_tokens == 0
    assert usage.cache_read_input_tokens == 0


def test_negative_count_clamps_to_zero():
    usage = read_cli_usage({"input_tokens": -5, "output_tokens": -1})

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0


def test_float_count_truncates_to_int():
    usage = read_cli_usage({"input_tokens": 10.9, "output_tokens": 2.0})

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_reads_as_zero_without_raising(bad):
    usage = read_cli_usage({"input_tokens": bad, "output_tokens": bad, "cache_read_input_tokens": bad})

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_cache_write_tokens_mirrors_cache_creation():
    usage = read_cli_usage({"input_tokens": 1, "output_tokens": 1, "cache_creation_input_tokens": 12})

    assert usage.prompt_tokens_details.cache_write_tokens == 12
    assert usage.prompt_tokens_details.cache_creation_tokens == 12


def test_text_tokens_excludes_cache_counts():
    usage = read_cli_usage(
        {
            "input_tokens": 100,
            "output_tokens": 1,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 5,
        }
    )

    assert usage.prompt_tokens_details.text_tokens == 100
    assert usage.prompt_tokens == 135


def test_unknown_extra_keys_are_ignored():
    usage = read_cli_usage({"input_tokens": 3, "output_tokens": 4, "service_tier": "standard", "iterations": [1, 2]})

    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 4
    assert usage.total_tokens == 7


@pytest.mark.parametrize(
    ("text", "expected"),
    [("100", 100), ("0", 0), ("10.9", 10), ("-5", 0), (" 42 ", 42)],
)
def test_numeric_string_count_is_parsed_not_read_as_zero(text, expected):
    usage = read_cli_usage({"input_tokens": text, "output_tokens": 1})

    assert usage.prompt_tokens == expected


@pytest.mark.parametrize("text", ["twelve", "", "nan", "inf", "-inf", "1e400"])
def test_non_numeric_string_still_reads_as_zero(text):
    usage = read_cli_usage({"input_tokens": text, "output_tokens": 1})

    assert usage.prompt_tokens == 0
