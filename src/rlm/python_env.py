from __future__ import annotations

import io
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable


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

    def set_context(self, text: str) -> None:
        self.globals["context"] = text
        self.globals["context_len"] = len(text)

        def get_slice(start: int, end: int):
            return text[start:end]

        self.globals["get_slice"] = get_slice

    def set_llm_query(self, handler: Callable[[str], str]) -> None:
        def llm_query(prompt_text: str) -> str:
            return handler(prompt_text)

        self.globals["llm_query"] = llm_query

    def exec(self, code: str) -> dict[str, str | bool]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        ok = True
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, self.globals, self.globals)
        except Exception:
            ok = False
            stderr.write(traceback.format_exc())
        return {"ok": ok, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
