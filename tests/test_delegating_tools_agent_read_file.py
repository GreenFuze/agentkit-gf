# tests/test_delegating_tools_agent_read_file.py
from __future__ import annotations

from agentkit_gf.delegating_tools_agent import DelegatingToolsAgent
from agentkit_gf.tools.fs import FileTools


def test_delegating_tools_agent_reads_file(temp_text_file):
    path, token, _ = temp_text_file

    # Build a delegating agent exposing only the gateway tool.
    # Mount FileTools on the internal executor; prefix public methods with "fs_".
    agent = DelegatingToolsAgent(
        model="openai:gpt-5-nano",
        builtin_enums=[],  # no provider built-ins needed for this test
        tool_sources=[FileTools()],
        class_prefix="fs",
        system_prompt=(
            "Answer-first. If you need to read a file, call delegate_ops. "
            "When you call delegate_ops, the 'tool' must be 'fs_read_text', "
            "args_json must include the file path and a reasonable max_bytes, "
            "and 'why' must contain the word 'because' explaining the need."
        ),
        ops_system_prompt="Execute exactly one tool and return only its result.",
    )

    user_msg = (
        f"Read the file at {path}. Do NOT guess. "
        "Use delegate_ops with tool 'fs_read_text', args_json "
        f"{{\"path\": \"{path}\", \"max_bytes\": 10000}}, and provide a 'why' that includes 'because' "
        "you need the actual file contents. Then answer with the first line."
    )

    out = agent.run_sync(user_msg).output

    # If the tool wasn't invoked, the model wouldn't know this unique token.
    assert token in out
