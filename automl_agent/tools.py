"""Explicit capability contracts and conservative candidate linting.

The AST check catches obvious mistakes. It is not a security sandbox; the
process supervisor and the deployment container provide separate controls.
"""

from __future__ import annotations

import ast
import json
import re
from uuid import uuid4

from jsonschema import Draft202012Validator, ValidationError


class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.hidden = set()

    def register(self, name, description, parameters, handler, *, advertised=True):
        if name in self.tools:
            raise ValueError(f"Duplicate tool: {name}")
        Draft202012Validator.check_schema(parameters)
        self.tools[name] = (description, parameters, handler)
        if not advertised:
            self.hidden.add(name)

    def schemas(self):
        return [{"type": "function", "function": {"name": name, "description": item[0], "parameters": item[1]}} for name, item in self.tools.items() if name not in self.hidden]

    def execute(self, name, arguments, *, allow_hidden=False):
        if name in self.hidden and not allow_hidden:
            return {"ok": False, "error": {"code": "UNAVAILABLE_TOOL", "message": "Use an advertised tool for this experiment mode"}}
        if name not in self.tools:
            return {"ok": False, "error": {"code": "UNKNOWN_TOOL", "message": f"Unknown tool: {name}"}}
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            Draft202012Validator(self.tools[name][1]).validate(parsed)
        except (ValueError, TypeError, ValidationError) as exc:
            return {"ok": False, "error": {"code": "INVALID_ARGUMENTS", "message": str(exc)[:1500]}}
        try:
            value = self.tools[name][2](**parsed)
            return {"ok": True, "value": value}
        except Exception as exc:
            return {"ok": False, "error": {"code": "TOOL_ERROR", "message": f"{type(exc).__name__}: {exc}"[:3000]}}


def parse_response_calls(message):
    calls = message.get("tool_calls")
    if calls:
        normalized = []
        for call in calls:
            function = call.get("function", {})
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            normalized.append({"id": call.get("id") or f"call_{uuid4().hex}", "type": "function", "function": {"name": function.get("name", ""), "arguments": arguments}})
        return normalized
    content = message.get("content") or ""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    if "{" not in content or "}" not in content:
        raise ValueError("Return a tool call or a JSON object containing tool and arguments")
    try:
        action = json.loads(content[content.index("{"):content.rindex("}") + 1])
    except (ValueError, TypeError) as exc:
        raise ValueError("Response does not contain valid JSON tool arguments") from exc
    if not isinstance(action, dict):
        raise ValueError("A tool action must be an object")
    name = action.get("tool", action.get("action", action.get("name")))
    if name is None and "experiments" in action:
        proposals = action["experiments"]
        typed = isinstance(proposals, list) and proposals and all(isinstance(p, dict) and "base_trial_id" in p for p in proposals)
        name = "run_model_experiments" if typed else "run_experiments"
    if name is None and "code" in action:
        name = "run_experiment"
    if name is None and "reason" in action:
        name = "finish_research"
    arguments = action.get("arguments", {key: value for key, value in action.items() if key not in {"tool", "action", "name"}})
    if not isinstance(name, str):
        raise ValueError("A tool name is required")
    return [{"id": f"call_{uuid4().hex}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]


def validate_candidate_code(code):
    if not isinstance(code, str) or not code.strip() or len(code) > 64000:
        raise ValueError("Code must contain between 1 and 64000 characters")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python: {exc.msg} on line {exc.lineno}") from exc
    if not any(isinstance(node, ast.FunctionDef) and node.name == "build_model" for node in tree.body):
        raise ValueError("Define build_model(context) returning an unfitted estimator or Pipeline")
    allowed = {"sklearn", "catboost", "lightgbm", "xgboost", "numpy", "pandas", "scipy", "math", "functools", "collections", "typing", "itertools", "operator", "statistics", "warnings", "re"}
    forbidden_calls = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "globals", "locals", "getattr", "setattr", "delattr", "vars", "exit", "quit"}
    forbidden_attrs = {"read_csv", "read_parquet", "read_pickle", "to_csv", "to_pickle", "load", "loadtxt", "fromfile", "tofile", "dump", "save", "savez", "system", "popen", "connect", "urlopen"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports = [item.name.split(".")[0] for item in node.names]
            if any(item not in allowed for item in imports):
                raise ValueError("Only modelling-library imports are allowed in candidate code")
        if isinstance(node, ast.ImportFrom) and (node.level or not node.module or node.module.split(".")[0] not in allowed):
            raise ValueError("Only absolute modelling-library imports are allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
            raise ValueError(f"Candidate code cannot call {node.func.id}")
        if isinstance(node, ast.Attribute) and ((node.attr.startswith("__") and node.attr != "__init__") or node.attr in forbidden_attrs):
            raise ValueError(f"Candidate code cannot access {node.attr}")
        if isinstance(node, ast.Name) and node.id in {"__builtins__", "__loader__", "__spec__"}:
            raise ValueError(f"Candidate code cannot access {node.id}")
