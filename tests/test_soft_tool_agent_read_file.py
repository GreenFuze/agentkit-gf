# tests/test_soft_tool_agent_read_file.py
from __future__ import annotations

from agentkit_gf.soft_tool_agent import SoftToolAgent
from agentkit_gf.tools.fs import FileTools


def test_soft_tool_agent_reads_file(temp_text_file, file_tools: FileTools):
    path, token, _ = temp_text_file

    # Soft tool registry: expose the tool name the model must request
    registry = {
        "read_text": file_tools.read_text,  # callable(**kwargs) -> dict
    }

    agent = SoftToolAgent(
        model="openai:gpt-5-nano",
        system_prompt=(
            "Be terse and precise. Prefer OPEN_RESPONSE when possible. "
            "When a file must be read, use a TOOL_CALL Envelope with message.kind='TOOL_CALL', "
            "tool 'read_text', and args_json containing {\"path\": ..., \"max_bytes\": ...}. "
            "After the host returns a TOOL_RESULT Envelope, respond with an OPEN_RESPONSE Envelope."
        ),
    )

    # Strongly bias the model toward the TOOL_CALL path with explicit Envelope wording.
    prompt = (
        f"You must read the local file at this exact path: {path}\n"
        "Available soft tool: read_text(path: str, max_bytes: int).\n"
        "First, return a TOOL_CALL Envelope for 'read_text' with args_json "
        f"{ {'path': path, 'max_bytes': 10000} } because you need to inspect the file.\n"
        "After I return a TOOL_RESULT Envelope, return an OPEN_RESPONSE Envelope that includes "
        "the exact token from the first line."
    )

    res = agent.run_soft_sync(prompt, registry, max_steps=5)

    # The agent should have executed at least one tool call named 'read_text'
    assert len(res.steps) >= 1
    assert res.steps[0].tool == "read_text"
    assert res.steps[0].success is True

    # The final text should reflect actual file contents (the unique token)
    assert token in res.final_text
