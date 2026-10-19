import json

import pytest

from automl_agent.tools import ToolRegistry, parse_response_calls, validate_candidate_code


def test_tool_schema_is_enforced_before_side_effect():
    calls = []
    registry = ToolRegistry()
    registry.register("fit", "Fit", {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}, "required": ["n"], "additionalProperties": False}, lambda n: calls.append(n) or n)
    assert registry.execute("fit", '{"n": 3}')["value"] == 3
    assert not registry.execute("fit", '{"n": -1}')["ok"]
    assert not registry.execute("fit", '{broken')["ok"]
    assert not registry.execute("unknown", '{}')["ok"]
    assert calls == [3]


def test_native_and_text_json_calls_preserve_arguments():
    args = {"hypothesis": "Try a tree", "code": "def build_model(context):\n    return None\n"}
    native = {"tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "run_experiment", "arguments": json.dumps(args)}}]}
    assert parse_response_calls(native)[0]["id"] == "call-1"
    text = {"content": '<think>hidden</think>```json\n' + json.dumps({"tool": "run_experiment", "arguments": args}) + '\n```'}
    call = parse_response_calls(text)[0]
    assert call["function"]["name"] == "run_experiment"
    assert json.loads(call["function"]["arguments"]) == args


def test_generated_code_contract_accepts_model_pipeline():
    validate_candidate_code("from sklearn.dummy import DummyRegressor\ndef build_model(context):\n    return DummyRegressor()\n")


@pytest.mark.parametrize("code", [
    "def broken(:", "x = 1", "import os\ndef build_model(context):\n return os.system('true')",
    "def build_model(context):\n return open('/tmp/x')", "def build_model(context):\n return object.__subclasses__()",
])
def test_generated_code_obvious_unsafe_or_invalid_operations_rejected(code):
    with pytest.raises(ValueError):
        validate_candidate_code(code)
