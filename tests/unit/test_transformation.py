import json
from itertools import pairwise

import pytest

from claude_subscription_adapter.transformation import ConversationTurn, prepare_cli_input


def test_system_messages_join_into_one_prompt():
    result = prepare_cli_input(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "system", "content": "Answer in English."},
            {"role": "user", "content": "Hi"},
        ]
    )

    assert result.system_prompt == "You are terse.\n\nAnswer in English."
    assert result.conversation == (ConversationTurn(role="user", content="Hi"),)


def test_developer_role_counts_as_system():
    result = prepare_cli_input(
        [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
    )

    assert result.system_prompt == "Be concise."
    assert result.conversation == (ConversationTurn(role="user", content="Hi"),)


def test_no_system_message_yields_empty_prompt_and_preserves_order():
    result = prepare_cli_input(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
    )

    assert result.system_prompt == ""
    assert result.conversation == (
        ConversationTurn(role="user", content="first"),
        ConversationTurn(role="assistant", content="second"),
        ConversationTurn(role="user", content="third"),
    )


def test_consecutive_same_role_messages_merge_with_blank_line():
    result = prepare_cli_input(
        [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "three"},
        ]
    )

    assert result.conversation == (
        ConversationTurn(role="user", content="one\n\ntwo"),
        ConversationTurn(role="assistant", content="three"),
    )


def test_three_consecutive_same_role_messages_merge_into_one_turn():
    result = prepare_cli_input(
        [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
    )

    assert result.conversation == (ConversationTurn(role="assistant", content="a\n\nb\n\nc"),)


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "three"},
            {"role": "assistant", "content": "four"},
            {"role": "user", "content": "five"},
        ],
        [
            {"role": "assistant", "content": "lead"},
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
            {"role": "user", "content": "next"},
        ],
    ],
)
def test_roles_strictly_alternate(messages):
    conversation = prepare_cli_input(messages).conversation

    assert all(a.role != b.role for a, b in pairwise(conversation))


def test_list_content_text_parts_extracted_in_order():
    result = prepare_cli_input(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "alpha"},
                    {"type": "text", "text": "beta"},
                ],
            }
        ]
    )

    assert result.conversation == (ConversationTurn(role="user", content="alpha\nbeta"),)


def test_non_text_content_part_is_serialised_not_dropped():
    image_part = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
    result = prepare_cli_input([{"role": "user", "content": [{"type": "text", "text": "look"}, image_part]}])

    content = result.conversation[0].content
    assert content.startswith("look\n")
    assert "https://example.com/a.png" in content
    assert json.loads(content.split("\n", 1)[1]) == image_part


def test_dict_content_is_serialised_not_iterated_as_keys():
    result = prepare_cli_input([{"role": "user", "content": {"type": "text", "text": "hello"}}])

    assert result.conversation == (ConversationTurn(role="user", content="hello"),)


def test_text_part_without_text_key_is_serialised_not_dropped():
    result = prepare_cli_input([{"role": "user", "content": [{"type": "text"}]}])

    assert result.conversation == (ConversationTurn(role="user", content='{"type": "text"}'),)


def test_text_part_with_non_string_text_is_serialised_not_dropped():
    result = prepare_cli_input([{"role": "user", "content": [{"type": "text", "text": 42}]}])

    content = result.conversation[0].content
    assert "42" in content
    assert json.loads(content) == {"type": "text", "text": 42}


def test_tuple_content_is_accepted_like_a_list():
    result = prepare_cli_input(
        [{"role": "user", "content": ({"type": "text", "text": "a"}, {"type": "text", "text": "b"})}]
    )

    assert result.conversation == (ConversationTurn(role="user", content="a\nb"),)


def test_non_serialisable_part_does_not_raise():
    result = prepare_cli_input([{"role": "user", "content": [{"type": "blob", "value": object()}]}])

    assert result.conversation[0].content.startswith("{")
    assert "blob" in result.conversation[0].content


def test_non_ascii_text_survives_serialisation_unescaped():
    result = prepare_cli_input([{"role": "user", "content": [{"type": "img", "alt": "héllo"}]}])

    assert "héllo" in result.conversation[0].content
    assert "\\u00e9" not in result.conversation[0].content


def test_empty_part_does_not_inject_a_blank_line():
    result = prepare_cli_input(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "alpha"},
                    {"type": "text", "text": ""},
                    {"type": "text", "text": "beta"},
                ],
            }
        ]
    )

    assert result.conversation == (ConversationTurn(role="user", content="alpha\nbeta"),)


def test_function_role_folds_into_user_turn_like_tool():
    result = prepare_cli_input([{"role": "function", "content": "9", "name": "adder"}])

    assert result.conversation == (ConversationTurn(role="user", content="Tool result from `adder`:\n9"),)


def test_tool_message_without_name_or_id_uses_generic_label():
    result = prepare_cli_input([{"role": "tool", "content": "42", "name": "", "tool_call_id": ""}])

    assert result.conversation[0].content == "Tool result from `tool`:\n42"


def test_unknown_role_contributes_no_turn():
    result = prepare_cli_input([{"role": "chatbot", "content": "dropped"}, {"role": "user", "content": "kept"}])

    assert result.conversation == (ConversationTurn(role="user", content="kept"),)


def test_tool_message_folds_into_user_turn_labelled_with_tool_name():
    result = prepare_cli_input([{"role": "tool", "content": "42", "tool_call_id": "call_1", "name": "calculator"}])

    assert result.conversation == (
        ConversationTurn(role="user", content="Tool result from `calculator` (id call_1):\n42"),
    )


def test_tool_message_without_name_falls_back_to_tool_call_id():
    result = prepare_cli_input([{"role": "tool", "content": "42", "tool_call_id": "call_9"}])

    assert result.conversation[0].content == "Tool result from `call_9`:\n42"


@pytest.mark.parametrize("content", ["", "   \n  ", None])
def test_empty_assistant_content_contributes_no_turn(content):
    result = prepare_cli_input([{"role": "user", "content": "hi"}, {"role": "assistant", "content": content}])

    assert result.conversation == (ConversationTurn(role="user", content="hi"),)


def test_empty_message_list_yields_empty_input():
    result = prepare_cli_input([])

    assert result.system_prompt == ""
    assert result.conversation == ()


def test_blank_system_message_does_not_add_separator():
    result = prepare_cli_input(
        [
            {"role": "system", "content": "  "},
            {"role": "system", "content": "real"},
            {"role": "user", "content": "hi"},
        ]
    )

    assert result.system_prompt == "real"


def test_input_messages_are_not_mutated():
    messages = [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
    ]
    snapshot = json.dumps(messages, sort_keys=True)

    prepare_cli_input(messages)

    assert json.dumps(messages, sort_keys=True) == snapshot


def test_whitespace_only_part_does_not_inject_a_blank_line():
    result = prepare_cli_input(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "alpha"},
                    {"type": "text", "text": "   "},
                    {"type": "text", "text": "beta"},
                ],
            }
        ]
    )

    assert result.conversation == (ConversationTurn(role="user", content="alpha\nbeta"),)


def test_bytes_content_decodes_instead_of_rendering_a_python_repr():
    result = prepare_cli_input([{"role": "user", "content": [b"hello"]}])

    assert result.conversation == (ConversationTurn(role="user", content="hello"),)


def test_undecodable_bytes_do_not_raise():
    result = prepare_cli_input([{"role": "user", "content": [b"\xff\xfe"]}])

    assert result.conversation[0].content == "��"


def test_serialised_part_keys_are_ordered_deterministically():
    result = prepare_cli_input([{"role": "user", "content": [{"zeta": 1, "alpha": 2, "type": "img"}]}])

    assert result.conversation[0].content == '{"alpha": 2, "type": "img", "zeta": 1}'


def test_unserialisable_value_still_renders_as_json():
    result = prepare_cli_input([{"role": "user", "content": [{"type": "blob", "value": object()}]}])

    assert json.loads(result.conversation[0].content)["type"] == "blob"


def test_an_assistant_turn_that_only_called_tools_is_replayed_with_named_results():
    result = prepare_cli_input(
        [
            {"role": "user", "content": "make a.txt"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "create_file", "arguments": '{"filePath": "a.txt"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_1", "content": "created"},
        ]
    )

    assert result.conversation == (
        ConversationTurn(role="user", content="make a.txt"),
        ConversationTurn(
            role="assistant",
            content='Called tool `create_file` (id toolu_1) with arguments: {"filePath": "a.txt"}',
        ),
        ConversationTurn(role="user", content="Tool result from `create_file` (id toolu_1):\ncreated"),
    )


def test_assistant_text_and_several_calls_all_survive_in_order():
    result = prepare_cli_input(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "On it.",
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "t2", "type": "function", "function": {"name": "b", "arguments": {"x": 1}}},
                ],
            },
        ]
    )

    assert result.conversation[1].content == (
        'On it.\nCalled tool `a` (id t1) with arguments: {}\nCalled tool `b` (id t2) with arguments: {"x": 1}'
    )


def test_a_tool_result_whose_call_is_unknown_is_labelled_with_its_id():
    result = prepare_cli_input(
        [{"role": "user", "content": "go"}, {"role": "tool", "tool_call_id": "toolu_9", "content": "done"}]
    )

    assert result.conversation == (ConversationTurn(role="user", content="go\n\nTool result from `toolu_9`:\ndone"),)


def test_an_empty_tool_result_is_still_replayed_so_the_call_is_not_left_unanswered():
    result = prepare_cli_input(
        [{"role": "user", "content": "go"}, {"role": "tool", "tool_call_id": "t1", "name": "run", "content": ""}]
    )

    assert result.conversation[-1].content.endswith("Tool result from `run` (id t1):\n(empty result)")


def test_parallel_results_from_the_same_tool_each_name_the_call_they_answer():
    """Three `read_file` results in a row are otherwise told apart only by their order."""
    calls = [{"id": f"t{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}} for i in (1, 2)]
    result = prepare_cli_input(
        [
            {"role": "user", "content": "read both"},
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "t1", "content": "first"},
            {"role": "tool", "tool_call_id": "t2", "content": "second"},
        ]
    )

    assert result.conversation[-1].content == (
        "Tool result from `read_file` (id t1):\nfirst\n\nTool result from `read_file` (id t2):\nsecond"
    )


# --------------------------------------------------------------------------------------
# skip_turns: the unsent tail for a resumed session
# --------------------------------------------------------------------------------------


_GROWING_HISTORY = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "second"},
    {"role": "user", "content": "third"},
]


def test_skip_turns_drops_that_many_turns_off_the_front():
    result = prepare_cli_input(_GROWING_HISTORY, skip_turns=2)

    assert result.conversation == (ConversationTurn(role="user", content="third"),)


def test_no_skip_turns_returns_the_full_conversation():
    result = prepare_cli_input(_GROWING_HISTORY, skip_turns=0)

    assert len(result.conversation) == 3


def test_a_skip_count_covering_the_whole_conversation_still_leaves_the_last_turn():
    """A stale or mismatched count must never make the caller's own newest turn vanish."""
    result = prepare_cli_input(_GROWING_HISTORY, skip_turns=3)

    assert result.conversation == (ConversationTurn(role="user", content="third"),)


def test_a_skip_count_larger_than_the_conversation_is_clamped_the_same_way():
    result = prepare_cli_input(_GROWING_HISTORY, skip_turns=100)

    assert result.conversation == (ConversationTurn(role="user", content="third"),)
