from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from rlm.orchestrator import OrchestratorConfig, RLMOrchestrator
from rlm.python_env import PythonEnv


@dataclass
class FakeToolCall:
    id: str
    function: SimpleNamespace


class FakeLLMClient:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice=None, model=None, temperature=0.2, max_tokens=None):
        self.calls += 1
        if self.calls == 1:
            func = SimpleNamespace(
                name="python_exec",
                arguments=json.dumps({"code": "print(get_slice(0, 5))"}),
            )
            return SimpleNamespace(content=None, tool_calls=[FakeToolCall("1", func)], raw=None)
        func = SimpleNamespace(
            name="final",
            arguments=json.dumps({"answer": "done"}),
        )
        return SimpleNamespace(content=None, tool_calls=[FakeToolCall("2", func)], raw=None)


def test_orchestrator_tool_loop():
    env = PythonEnv()
    env.set_context("hello world")
    client = FakeLLMClient()
    orch = RLMOrchestrator(
        client=client,
        env=env,
        config=OrchestratorConfig(max_turns=3, max_subcalls=2, max_obs_chars=2000),
        model="fake",
        submodel="fake",
    )
    answer = orch.run("Pregunta")
    assert answer == "done"
