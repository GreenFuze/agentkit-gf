# tests/conftest.py
from __future__ import annotations

import os
import uuid
from pathlib import Path
import pytest

from agentkit_gf.tools.fs import FileTools

# Auto-skip all tests if there's no API key
@pytest.fixture(autouse=True)
def _require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping integration tests")

@pytest.fixture
def temp_text_file(tmp_path: Path):
    """
    Creates a file with a hard-to-guess token to verify true file reads.
    Returns (path_posix, token, full_text).
    """
    token = f"AGENTKIT_GF_TEST_{uuid.uuid4().hex}"
    text = token + "\nsecond line here"
    p = tmp_path / "notes.txt"
    p.write_text(text, encoding="utf-8")
    return p.as_posix(), token, text

@pytest.fixture
def file_tools(tmp_path: Path):
    # Sandbox file operations to the test temp dir
    return FileTools(root_dir=str(tmp_path))
