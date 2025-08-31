# GF AgentKit

Lightweight agents on top of **Pydantic AI** with:

* **True *soft tools*** (reasoning-first; tools are optional, not mandatory).
* **A single “gateway” tool** that delegates to a separate executor (to reduce tool gravity).
* **Built-in conversation history** (persistent transcript managed by the agent).
* A few **ready-to-use tools** (file I/O, safe process execution, agent delegation).

> ⚠️ This is a work-in-progress. The API may evolve.

---

## Why?

Many agent frameworks push models to use tools aggressively (“hard tools”), which can be great for workflow graphs but not for **reasoning-first** tasks. GF AgentKit makes tool use **opt-in**:

* **SoftToolAgent**: the model decides when to ask for a tool using a **schema-guided envelope** (OPEN\_RESPONSE / TOOL\_CALL / TOOL\_RESULT). The host runs tools and feeds results back as context.
* **DelegatingToolsAgent**: the model sees **one** gateway tool (`delegate_ops`). That tool delegates to an internal executor which owns all the real tools. This reduces tool gravity and lets you keep a clean reasoning loop.

Both agents keep a **persistent transcript** so follow-ups naturally refer back to earlier turns.

---

## Installation

```bash
pip install agentkit-gf
```

Requires: Python 3.11+.

You’ll also need an OpenAI key to use `gpt-5-nano`:

```bash
# macOS/Linux
export OPENAI_API_KEY=sk-...

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

---

## At a Glance

```
agentkit_gf/
  ├─ _base_agent.py               # shared transcript + constructor glue
  ├─ soft_tool_agent.py           # true soft tools (Envelope schema)
  ├─ delegating_tools_agent.py    # single gateway tool; delegates to executor
  └─ tools/
       ├─ fs.py                   # FileTools: read/write/stat/hash/list (sandboxable)
       ├─ os.py                   # ProcessTools: run_process/run_shell (policy controlled)
       ├─ agent.py                # create_agent_delegation_tool(...) factory
       └─ builtin_tools_matrix.py # BuiltinTool enums + provider validation
```

---

## Agents

### 1) SoftToolAgent (true soft tool)

* The model returns a single **Envelope** JSON object each hop:

  * `OPEN_RESPONSE { text, confidence? }`
  * `TOOL_CALL { tool, args_json, reason }`
  * `TOOL_RESULT { tool, args_json, result_json, success, note? }` (normally emitted by the host)
* You provide a **registry** of Python callables (`tool_name -> callable(**kwargs)`).
* The agent executes tools in host code and feeds a `TOOL_RESULT` back to the model.
* Maintains an internal transcript across turns.

**Minimal example (read a file):**

```python
from agentkit_gf.soft_tool_agent import SoftToolAgent
from agentkit_gf.tools.fs import FileTools

file_tools = FileTools(root_dir=".")   # restrict if you like
registry = {"read_text": file_tools.read_text}

agent = SoftToolAgent(model="openai:gpt-5-nano")

prompt = (
    "Read ./notes.txt and tell me the first line.\n"
    "If you need the file, return a TOOL_CALL Envelope for tool 'read_text' with args_json "
    '{"path": "./notes.txt", "max_bytes": 10000}. Then, after TOOL_RESULT, respond with OPEN_RESPONSE.'
)

result = agent.run_soft_sync(prompt, registry, max_steps=5)
print(result.final_text)
```

**Transcript helpers:**

```python
print(agent.export_history_text())
agent.reset_history()
```

### 2) DelegatingToolsAgent (single gateway tool)

* Presents **one** tool (`delegate_ops`) to the model.
* Internally spins up a private **executor agent** that owns all real tools (including optional provider built-ins like WebSearch).
* You pass in objects or callables; **public methods** are automatically exposed as tools (optionally prefixed).

**Example:**

```python
from agentkit_gf.delegating_tools_agent import DelegatingToolsAgent
from agentkit_gf.tools.fs import FileTools
from agentkit_gf.tools.os import ProcessTools
from agentkit_gf.tools.builtin_tools_matrix import BuiltinTool

agent = DelegatingToolsAgent(
    model="openai:gpt-5-nano",
    builtin_enums=[BuiltinTool.WEB_SEARCH],  # optional provider built-ins
    tool_sources=[
        FileTools(root_dir="."),
        ProcessTools(root_cwd=".", allowed_basenames=["python", "bash", "ls"])
    ],
    class_prefix="fs",  # public tool names become "fs_read_text", etc.
    system_prompt=(
        "Answer-first. Use delegate_ops only if a specific missing fact requires it."
    ),
    ops_system_prompt="Execute exactly one tool and return only its result.",
)

reply = agent.run_sync(
    "Read ./notes.txt (use delegate_ops/tool 'fs_read_text' if needed) and summarize the first line."
).output

print(reply)
```

---

## Included Tools

### FileTools (`agentkit_gf.tools.fs`)

* `read_text(path, max_bytes=200_000, encoding="utf-8")`
* `read_bytes_base64(path, max_bytes=200_000)`
* `write_text(path, content, overwrite=False, encoding="utf-8")`
* `stat(path)` / `list_dir(path, include_hidden=False, max_entries=1000)`
* `hash_file(path, algorithm=HashAlgorithm.SHA256)`

All enforce **fail-fast** validation and can be **sandboxed** with `root_dir`.

### ProcessTools (`agentkit_gf.tools.os`)

* `run_process(argv: Sequence[str], timeout_sec=10, cwd=None)` (no shell; recommended)
* `run_shell(command: str, timeout_sec=10, cwd=None)` (flexible; riskier)

Policy controls:

* `root_cwd` (path sandbox)
* `allowed_basenames` (allowlist executables)
* `max_output_bytes` (clip stdout/stderr)

### Agent Delegation Tool (`agentkit_gf.tools.agent`)

* `create_agent_delegation_tool(agent_factory: Callable[[str], Agent]) -> Callable[..., dict]`
* Produces a registry callable: `delegate_agent(agent_name: str, prompt: str) -> {"output": ...}`
* Handy if your soft tool needs to **spin up or discover** another agent on demand.

---

## Soft vs. Hard Tools

* **Soft tools** (this library):

  * The model **asks** to call a tool via a schema; the host decides and executes.
  * Great for reasoning-first flows where the model should **prefer answering** from context.
* **Hard tools**:

  * Registered with the provider; models are often biased to call them.
  * Better for rigid flows or “do X with Y, then Z” pipelines.

You can mix: use `SoftToolAgent` for reasoning, and `DelegatingToolsAgent` when you need a single, auditable gateway to real tools (including provider built-ins).

---

## Extending with Your Own Tools

You can pass **objects** or **callables**:

```python
class MyDataOps:
    def summarize_csv(self, path: str, top_n: int = 5) -> dict:
        # ... return JSON-serializable result ...
        return {"summary": "...", "top_n": top_n}

agent = DelegatingToolsAgent(
    model="openai:gpt-5-nano",
    builtin_enums=[],
    tool_sources=[MyDataOps()],
    class_prefix="data",  # exposes "data_summarize_csv"
)
```

For **SoftToolAgent**, add to the registry:

```python
registry = {"summarize_csv": MyDataOps().summarize_csv}
```

---

## API Reference (quick)

### `SoftToolAgent`

```python
SoftToolAgent(
    model: str,
    system_prompt: str | None = None,
    allow_llm_tool_result: bool = False,
)

run_soft_sync(
    prompt: str,
    tool_registry: Mapping[str, Callable[..., Any]],
    max_steps: int = 4,
) -> SoftRunResult

# history helpers
export_history_text() -> str
export_history_blocks() -> Sequence[str]
reset_history() -> None
```

**Envelope schema:** the model returns exactly one JSON object with `message.kind` in:

* `OPEN_RESPONSE { text, confidence? }`
* `TOOL_CALL { tool, args_json, reason }`  (args\_json must be an object string)
* `TOOL_RESULT { tool, args_json, result_json, success, note? }`

### `DelegatingToolsAgent`

```python
DelegatingToolsAgent(
    model: str,
    builtin_enums: Sequence[BuiltinTool],
    tool_sources: Sequence[Callable | object | AbstractToolset] = (),
    class_prefix: str | None = None,
    system_prompt: str | None = None,
    ops_system_prompt: str | None = None,
)

# single-step run (history-aware)
run_sync(prompt: str) -> RunResult
```

This agent exposes only **one** public tool to the model: `delegate_ops(tool, args_json, why)` (the executor runs exactly one real tool; results are recorded to the transcript).

---

## License

See `LICENSE` - MIT

---
