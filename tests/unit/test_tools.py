import pytest

from claude_subscription_adapter.tools import caller_name, read_client_tools


def _function(name, **extra):
    return {"type": "function", "function": {"name": name, **extra}}


def test_short_names_are_offered_unchanged_and_map_back():
    tools = read_client_tools([_function("create_file", description="d", parameters={"type": "object"})], None)

    assert tools[0].alias == "create_file"
    assert tools[0].description == "d"
    assert caller_name(tools, "mcp__c__create_file") == "create_file"


def test_long_names_get_a_distinct_alias_that_fits_the_64_character_limit():
    first = "x" * 60 + "_one"
    second = "x" * 60 + "_two"
    tools = read_client_tools([_function(first), _function(second)], None)

    assert all(len(f"mcp__c__{tool.alias}") <= 64 for tool in tools)
    assert tools[0].alias != tools[1].alias
    assert [caller_name(tools, f"mcp__c__{tool.alias}") for tool in tools] == [first, second]


@pytest.mark.parametrize("cli_name", ["WebSearch", "mcp__other__create_file", "mcp__c__unknown"])
def test_names_that_are_not_the_callers_tools_map_to_nothing(cli_name):
    assert caller_name(read_client_tools([_function("create_file")], None), cli_name) is None


@pytest.mark.parametrize(
    ("tools", "tool_choice"),
    [(None, None), ([_function("a")], "none"), ("not a list", None), ([{"function": {}}], None)],
)
def test_no_tools_are_offered_when_there_are_none_or_they_are_unreadable(tools, tool_choice):
    assert read_client_tools(tools, tool_choice) == ()


def test_a_missing_schema_defaults_to_an_empty_object():
    assert read_client_tools([_function("a")], "auto")[0].parameters == {"type": "object", "properties": {}}


def test_one_malformed_tool_does_not_cost_the_caller_the_others():
    tools = read_client_tools([{"type": "web_search_preview"}, {"function": None}, _function("create_file")], None)

    assert [tool.name for tool in tools] == ["create_file"]


def test_a_no_argument_tool_still_offers_an_object_schema():
    assert read_client_tools([_function("refresh", parameters={})], None)[0].parameters == {"type": "object"}
