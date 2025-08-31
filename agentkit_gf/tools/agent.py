# tools/agent.py
from __future__ import annotations

from typing import Callable

from pydantic_ai import Agent


def create_agent_delegation_tool(
    agent_factory: Callable[[str], Agent],
) -> Callable[..., dict]:
    """
    Factory for a soft-tool registry entry that delegates to another Agent.

    Returns a callable:

        delegate_agent(*, agent_name: str, prompt: str) -> dict

    Behavior:
      - Resolves an agent via agent_factory(agent_name)
      - Runs it synchronously on 'prompt'
      - Returns {"output": <str>}

    Fail-fast philosophy:
      - Clear validation for inputs and factory behavior.
    """
    if not callable(agent_factory):
        raise ValueError("agent_factory must be callable and accept a single 'agent_name: str' argument")

    def delegate_agent(*, agent_name: str, prompt: str) -> dict:
        if not agent_name or not isinstance(agent_name, str):
            raise ValueError("agent_name must be a non-empty string")

        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        agent = agent_factory(agent_name)
        if not isinstance(agent, Agent):
            raise TypeError("agent_factory must return a pydantic_ai.Agent instance")

        result = agent.run_sync(prompt)
        return {"output": result.output}

    return delegate_agent
