# tools/fs.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional

import base64
import hashlib
import os
import stat
import time

from pydantic import BaseModel, Field, NonNegativeInt


class HashAlgorithm(Enum):
    SHA256 = "sha256"
    SHA1 = "sha1"
    MD5 = "md5"  # consider disabling in security-sensitive contexts


class FileText(BaseModel):
    path: str
    size_bytes: NonNegativeInt
    truncated: bool
    content: str


class FileBytes(BaseModel):
    path: str
    size_bytes: NonNegativeInt
    truncated: bool
    content_base64: str


class FileStat(BaseModel):
    path: str
    size_bytes: NonNegativeInt
    is_file: bool
    is_dir: bool
    mode_octal: str
    mtime_epoch: float


class DirEntry(BaseModel):
    name: str
    is_file: bool
    is_dir: bool
    size_bytes: Optional[NonNegativeInt] = None
    mtime_epoch: Optional[float] = None


class DirListing(BaseModel):
    path: str
    count: NonNegativeInt
    entries: List[DirEntry]


class FileHash(BaseModel):
    path: str
    algorithm: HashAlgorithm
    hexdigest: str


@dataclass(frozen=True)
class _Root:
    root: Optional[Path]

    def resolve_and_check(self, user_path: str) -> Path:
        if not user_path:
            raise ValueError("path is required and must be non-empty")

        p = Path(user_path).expanduser().resolve(strict=False)

        if self.root is None:
            return p

        root = self.root
        # Python <3.9 compatibility: use try/except on relative_to
        try:
            _ = p.resolve().relative_to(root.resolve())
        except Exception:
            raise PermissionError(f"path {p} is outside the allowed root {root}")  # fail-fast

        return p


class FileTools:
    """
    File toolset with optional sandboxing to a root directory.

    Public methods are intended to be exposed as LLM tools via your toolset assembler.
    Methods return Pydantic models for strong typing, but callers exposing them as tools
    should convert to `model_dump()` so results are JSON-serializable.
    """

    def __init__(self, *, root_dir: Optional[str] = None) -> None:
        self._root = _Root(Path(root_dir).resolve()) if root_dir else _Root(None)

    # ----- Public Tool Methods -------------------------------------------------

    def read_text(self, path: str, max_bytes: int = 200_000, encoding: str = "utf-8") -> dict:
        """
        Read a UTF-8 text file (truncated to max_bytes).
        Fails if the path is a directory or does not exist.
        """
        p = self._root.resolve_and_check(path)

        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"file not found: {p}")

        data = p.read_text(encoding=encoding, errors="replace")

        truncated = len(data.encode(encoding, errors="replace")) > max_bytes
        if truncated:
            # truncate on bytes boundary
            b = data.encode(encoding, errors="replace")[:max_bytes]
            data = b.decode(encoding, errors="replace") + "\n[...truncated...]"

        res = FileText(
            path=str(p),
            size_bytes=p.stat().st_size,
            truncated=truncated,
            content=data,
        )

        return res.model_dump()

    def read_bytes_base64(self, path: str, max_bytes: int = 200_000) -> dict:
        """
        Read file bytes and return base64 (truncated to max_bytes).
        Use for binary files or unknown encodings.
        """
        p = self._root.resolve_and_check(path)

        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"file not found: {p}")

        raw = p.read_bytes()

        truncated = len(raw) > max_bytes
        if truncated:
            raw = raw[:max_bytes]

        res = FileBytes(
            path=str(p),
            size_bytes=p.stat().st_size,
            truncated=truncated,
            content_base64=base64.b64encode(raw).decode("ascii"),
        )

        return res.model_dump()

    def write_text(self, path: str, content: str, *, overwrite: bool = False, encoding: str = "utf-8") -> dict:
        """
        Write text to a file. Fails if file exists and overwrite=False.
        Creates parent directories if needed.
        """
        p = self._root.resolve_and_check(path)

        if p.exists() and not p.is_file():
            raise IsADirectoryError(f"cannot write: {p} exists and is not a file")

        if p.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {p}")

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)

        return FileStat(
            path=str(p),
            size_bytes=p.stat().st_size,
            is_file=True,
            is_dir=False,
            mode_octal=oct(p.stat().st_mode & 0o777),
            mtime_epoch=p.stat().st_mtime,
        ).model_dump()

    def stat(self, path: str) -> dict:
        """
        Return file or directory stat info.
        """
        p = self._root.resolve_and_check(path)

        if not p.exists():
            raise FileNotFoundError(f"path not found: {p}")

        st = p.stat()

        return FileStat(
            path=str(p),
            size_bytes=st.st_size,
            is_file=p.is_file(),
            is_dir=p.is_dir(),
            mode_octal=oct(st.st_mode & 0o777),
            mtime_epoch=st.st_mtime,
        ).model_dump()

    def list_dir(self, path: str, *, include_hidden: bool = False, max_entries: int = 1000) -> dict:
        """
        List directory entries (non-recursive). Returns up to max_entries.
        """
        p = self._root.resolve_and_check(path)

        if not p.exists():
            raise FileNotFoundError(f"path not found: {p}")

        if not p.is_dir():
            raise NotADirectoryError(f"not a directory: {p}")

        entries: List[DirEntry] = []

        with os.scandir(p) as it:
            for de in it:
                if not include_hidden and de.name.startswith("."):
                    continue

                try:
                    st = de.stat(follow_symlinks=False)
                except FileNotFoundError:
                    # broken symlink or racing deletion; skip
                    continue

                entries.append(
                    DirEntry(
                        name=de.name,
                        is_file=stat.S_ISREG(st.st_mode),
                        is_dir=stat.S_ISDIR(st.st_mode),
                        size_bytes=st.st_size if stat.S_ISREG(st.st_mode) else None,
                        mtime_epoch=st.st_mtime,
                    )
                )

                if len(entries) >= max_entries:
                    break

        return DirListing(path=str(p), count=len(entries), entries=entries).model_dump()

    def hash_file(self, path: str, algorithm: HashAlgorithm = HashAlgorithm.SHA256, *, chunk_size: int = 1 << 20) -> dict:
        """
        Compute a file hash (default sha256). Streams in chunks to avoid memory spikes.
        """
        p = self._root.resolve_and_check(path)

        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"file not found: {p}")

        if algorithm is HashAlgorithm.SHA256:
            h = hashlib.sha256()
        elif algorithm is HashAlgorithm.SHA1:
            h = hashlib.sha1()
        elif algorithm is HashAlgorithm.MD5:
            h = hashlib.md5()
        else:
            raise ValueError(f"unsupported hash algorithm: {algorithm}")

        with p.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)

        return FileHash(path=str(p), algorithm=algorithm, hexdigest=h.hexdigest()).model_dump()
