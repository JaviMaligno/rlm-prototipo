from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FileEntry:
    index: int
    name: str
    start: int
    end: int
    size: int


@dataclass(frozen=True)
class LoadedDoc:
    text: str
    sources: list[str]
    char_len: int
    token_estimate: int
    file_entries: list[FileEntry]


def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars per token in English/Spanish mixed text.
    return max(1, len(text) // 4)


def _expand_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in inputs:
        matches = glob(pattern)
        if matches:
            paths.extend(Path(p) for p in matches)
        else:
            paths.append(Path(pattern))
    return paths


def load_documents(inputs: Iterable[str]) -> LoadedDoc:
    paths = _expand_inputs(inputs)
    if not paths:
        raise RuntimeError("No input files found.")

    parts: list[str] = []
    sources: list[str] = []
    file_entries: list[FileEntry] = []
    offset = 0
    for idx, path in enumerate(paths):
        if not path.exists():
            raise RuntimeError(f"File not found: {path}")
        if path.is_dir():
            raise RuntimeError(f"Expected file but got directory: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        sources.append(str(path))
        header = f"\n\n===== FILE: {path} =====\n\n"
        chunk = header + text
        start = offset + len(header)
        end = offset + len(chunk)
        file_entries.append(FileEntry(
            index=idx,
            name=str(path),
            start=start,
            end=end,
            size=len(text),
        ))
        parts.append(chunk)
        offset += len(chunk)

    full_text = "".join(parts)
    return LoadedDoc(
        text=full_text,
        sources=sources,
        char_len=len(full_text),
        token_estimate=estimate_tokens(full_text),
        file_entries=file_entries,
    )
