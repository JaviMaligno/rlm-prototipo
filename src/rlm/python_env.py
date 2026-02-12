from __future__ import annotations

import ast
import io
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .utils import FileEntry


class PythonEnv:
    def __init__(self) -> None:
        self.globals: dict[str, object] = {}
        self._install_helpers()

    def _install_helpers(self) -> None:
        def search(pattern: str, max_results: int = 5, flags: int = 0):
            text = self.globals.get("context", "")
            results = []
            for match in re.finditer(pattern, text, flags):
                start, end = match.span()
                snippet = text[max(0, start - 80) : min(len(text), end + 80)]
                results.append(
                    {
                        "match": match.group(0),
                        "start": start,
                        "end": end,
                        "snippet": snippet,
                    }
                )
                if len(results) >= max_results:
                    break
            return results

        self.globals["search"] = search

    def set_context(self, text: str, file_entries: list[FileEntry] | None = None) -> None:
        self.globals["context"] = text
        self.globals["context_len"] = len(text)

        def get_slice(start: int, end: int):
            return text[start:end]

        self.globals["get_slice"] = get_slice

        entries = file_entries or []
        self.globals["file_count"] = len(entries)

        def list_files() -> list[dict]:
            return [
                {"index": e.index, "name": e.name, "start": e.start, "end": e.end, "size": e.size}
                for e in entries
            ]

        def get_file(i: int) -> str:
            if not entries:
                raise RuntimeError("No file entries available.")
            if i < 0 or i >= len(entries):
                raise IndexError(f"File index {i} out of range (0..{len(entries)-1})")
            e = entries[i]
            return text[e.start : e.end]

        self.globals["list_files"] = list_files
        self.globals["get_file"] = get_file

    def set_llm_query(self, handler: Callable[[str], str]) -> None:
        def llm_query(prompt_text: str) -> str:
            return handler(prompt_text)

        self.globals["llm_query"] = llm_query

    def set_llm_query_batch(self, handler: Callable[[list[str], int], list[str]]) -> None:
        def llm_query_batch(prompts: list[str], max_workers: int = 5) -> list[str]:
            return handler(prompts, max_workers)

        self.globals["llm_query_batch"] = llm_query_batch

    @staticmethod
    def _split_last_expr(code: str) -> tuple[str, str | None]:
        """Split code into (body, last_expr) if the last statement is an expression."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, None
        if not tree.body:
            return code, None
        last = tree.body[-1]
        if not isinstance(last, ast.Expr):
            return code, None
        # Compile everything except the last expression as exec,
        # and the last expression separately for eval.
        if len(tree.body) == 1:
            return "", ast.get_source_segment(code, last.value) or code.strip()
        # Get the source line range for the last expression
        last_start = last.lineno - 1  # 0-indexed
        lines = code.splitlines(keepends=True)
        body = "".join(lines[:last_start])
        expr = "".join(lines[last_start:]).strip()
        return body, expr

    def exec(self, code: str) -> dict[str, str | bool]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        ok = True
        try:
            body, last_expr = self._split_last_expr(code)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                if body.strip():
                    exec(body, self.globals, self.globals)
                if last_expr is not None:
                    result = eval(last_expr, self.globals, self.globals)
                    if result is not None:
                        stdout.write(repr(result) if not isinstance(result, str) else result)
                        stdout.write("\n")
        except Exception:
            ok = False
            stderr.write(traceback.format_exc())
        return {"ok": ok, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
