from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .utils import FileEntry

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .llm import LLMClient
from .python_env import PythonEnv


SYSTEM_PROMPT = """\
Eres un RLM (Recursive Language Model). El documento completo NO esta en tu contexto,
PERO tienes acceso TOTAL a el via codigo Python. Puedes leer cualquier parte del documento
ejecutando codigo. NUNCA digas que no tienes acceso al documento.

## Herramientas en el entorno Python

- `context` : str completo del documento (accesible via python_exec)
- `context_len` : int con la longitud total del documento
- `file_count` : int con el numero de archivos cargados
- `list_files()` -> lista de dicts `{index, name, start, end, size}` con todos los archivos
- `get_file(i)` -> str con el contenido del archivo i (0-indexed)
- `get_slice(start, end)` -> str
- `search(pattern, max_results=5)` -> lista de matches con snippets
- `llm_query(prompt_text)` -> str   ← SUB-LLAMADA A OTRO LLM (consume 1 subcall)
- `llm_query_batch(prompts, max_workers=5)` -> list[str]  ← MULTIPLES sub-llamadas EN PARALELO

## Reglas OBLIGATORIAS

1. **SIEMPRE usa herramientas**: Responde SOLO via la herramienta `final`. NUNCA respondas
   con texto plano. Si quieres responder, llama a `final(answer=...)`.
2. **Codigo corto**: Maximo 25 lineas por python_exec. Nada de regex complejas.
4. **Usa llm_query() para analizar texto**: No intentes parsear contenido con regex.
   Ejemplo correcto:
     summary = llm_query(f"Resume en 1 frase la contribucion principal:\\n{fragment}")
   Ejemplo INCORRECTO:
     m = re.search(r"(?:we propose|this paper)...", text)  # NO hagas esto
5. **Usa llm_query_batch() para TODOS los archivos de una vez**: Crea un prompt por archivo
   y lanza UN SOLO batch con todos. NUNCA hagas llm_query() en un bucle ni multiples batches.
   Ejemplo correcto:
     prompts = [f"Resume en 1 frase:\\n{get_file(i)[:6000]}" for i in range(file_count)]
     results = llm_query_batch(prompts, max_workers=5)
   Si el batch excede el budget, se ejecutaran los que quepan y el resto se marca [skipped].
6. **Se ESPECIFICO**: Tus respuestas deben contener datos concretos del documento (nombres,
   cifras, conceptos). NUNCA respondas con generalidades vacias como "se propone una nueva
   arquitectura" sin decir CUAL o QUE hace.
7. **Gestiona tu budget**: Cada llm_query() gasta 1 subcall. llm_query_batch() gasta
   len(prompts) subcalls. Plan: 1er batch = file_count prompts (usa TODO el budget).
   La sintesis la haces tu en python_exec sin gastar subcalls.
8. **Sintetiza TU MISMO**: Despues del batch, los resultados estan en variables Python.
   Agrupa y sintetiza directamente en python_exec, luego llama `final(answer=...)`.
   NO uses llm_query() para sintetizar — el prompt seria demasiado largo y se truncaria.
9. **Separa analisis de respuesta**: Primero usa python_exec para analizar (el ultimo
   valor de expresion se muestra automaticamente). Luego, en una llamada SEPARADA,
   usa `final(answer=...)` con esos datos. NO intentes hacer todo en un solo python_exec.

## Flujo recomendado

1. Lee la tabla de contenidos (ya incluida en tu primer mensaje) para conocer los archivos.
2. **Cubre TODOS los archivos en UN solo batch**: crea un prompt por archivo y lanza
   `llm_query_batch(prompts)` con TODOS a la vez. NO hagas multiples batches pequeños ni
   llm_query() individuales para cada archivo. Con N archivos y B subcalls, tu primer batch
   debe ser de N prompts (o B-3 si N > B-3, reservando 3 para sintesis).
3. Sintetiza TU MISMO en python_exec: agrupa los resultados por tema/categoria
   usando codigo Python (los datos ya estan en variables). NO uses llm_query() para esto.
4. Responde: llama a `final(answer=...)` con la sintesis. NUNCA respondas sin usar `final`.
"""


@dataclass
class OrchestratorConfig:
    max_turns: int = 15
    max_subcalls: int = 90
    max_obs_chars: int = 8000
    max_subcall_prompt_chars: int = 6000
    temperature: float = 0.2


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as mm:ss or hh:mm:ss."""
    m, s = divmod(int(seconds), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class RLMOrchestrator:
    def __init__(
        self,
        client: LLMClient,
        env: PythonEnv,
        config: OrchestratorConfig,
        console: Console | None = None,
        model: str | None = None,
        submodel: str | None = None,
    ) -> None:
        self.client = client
        self.env = env
        self.config = config
        # Pin Console to the real stdout so prints are visible even inside
        # PythonEnv.exec() which redirects sys.stdout to a StringIO.
        self.console = console or Console(file=sys.stdout)
        self.model = model
        self.submodel = submodel
        self.subcalls = 0
        self._subcall_lock = threading.Lock()
        self._run_start: float = 0.0

        self.env.set_llm_query(self._llm_query)
        self.env.set_llm_query_batch(self._llm_query_batch)

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Print and immediately flush so output is visible even in piped/background mode."""
        self.console.print(*args, **kwargs)
        try:
            (self.console.file or sys.stdout).flush()
        except Exception:
            pass

    def _elapsed(self) -> str:
        return _fmt_elapsed(time.time() - self._run_start)

    def _llm_query(self, prompt_text: str) -> str:
        with self._subcall_lock:
            if self.subcalls >= self.config.max_subcalls:
                raise RuntimeError("Max subcalls reached.")
            self.subcalls += 1
            current = self.subcalls

        # Truncate very long prompts — GPT-5 returns null content on large inputs
        max_prompt = self.config.max_subcall_prompt_chars
        if len(prompt_text) > max_prompt:
            prompt_text = prompt_text[:max_prompt] + "\n...[truncated]"

        self._print(
            f"  [cyan]⤷ llm_query #{current}/{self.config.max_subcalls}[/cyan] "
            f"[dim]({self._elapsed()})[/dim] "
            f"[dim]{len(prompt_text)}ch — {prompt_text[:100].replace(chr(10), ' ')}...[/dim]"
        )

        t0 = time.time()
        messages = [
            {
                "role": "system",
                "content": "Respond concisely and precisely using only the given fragment.",
            },
            {"role": "user", "content": prompt_text},
        ]

        answer = ""
        max_retries = 4
        for attempt in range(max_retries):
            response = self.client.chat(
                messages=messages,
                tools=None,
                tool_choice=None,
                model=self.submodel or self.model,
                max_tokens=8000,
            )
            answer = response.content or ""
            if answer:
                break
            if attempt < max_retries - 1:
                wait = 1.0 * (attempt + 1)
                self._print(f"    [yellow]⟳ empty, retrying in {wait:.0f}s... (attempt {attempt+1}/{max_retries})[/yellow]")
                time.sleep(wait)

        dt = time.time() - t0
        status = "[green]✓[/green]" if answer else "[yellow]∅[/yellow]"
        self._print(
            f"    {status} [dim]{dt:.1f}s — {len(answer)} chars[/dim]"
        )
        return answer

    def _llm_query_batch(self, prompts: list[str], max_workers: int = 5) -> list[str]:
        needed = len(prompts)
        with self._subcall_lock:
            available = self.config.max_subcalls - self.subcalls

        # Partial execution: process as many as fit in the budget
        to_run = min(needed, available)
        skipped = needed - to_run

        if to_run == 0:
            self._print(
                f"  [yellow]⚠ llm_query_batch: 0/{needed} — budget exhausted[/yellow]"
            )
            return [f"[skipped: budget exhausted]"] * needed

        if skipped > 0:
            self._print(
                f"  [yellow]⚠ llm_query_batch: running {to_run}/{needed} "
                f"(skipping last {skipped} — budget)[/yellow]"
            )

        self._print(
            f"  [cyan]⤷ llm_query_batch: {to_run} prompts, max_workers={max_workers}[/cyan] "
            f"[dim]({self._elapsed()})[/dim]"
        )

        results: list[str | None] = [None] * needed

        def _run_one(idx: int, prompt: str) -> tuple[int, str]:
            try:
                return idx, self._llm_query(prompt)
            except Exception as exc:
                return idx, f"[error: {exc}]"

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_one, i, p): i for i, p in enumerate(prompts[:to_run])
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        dt = time.time() - t0
        ok_count = sum(1 for r in results if r and not r.startswith("[error:"))

        # Mark skipped prompts
        for i in range(to_run, needed):
            results[i] = f"[skipped: budget exhausted (index {i})]"

        self._print(
            f"  [green]✓ batch done[/green] [dim]{dt:.1f}s — "
            f"{ok_count}/{to_run} succeeded"
            + (f", {skipped} skipped" if skipped else "")
            + "[/dim]"
        )
        return [r or "[error: no result]" for r in results]

    @staticmethod
    def _build_toc(file_entries: list[FileEntry], total_chars: int) -> str:
        lines = [f"## Tabla de Contenidos ({len(file_entries)} archivos, {total_chars:,} chars)\n"]
        for e in file_entries:
            name = e.name.rsplit("/", 1)[-1] if "/" in e.name else e.name
            lines.append(f"  [{e.index}] {name} ({e.size:,} chars)")
        lines.append("")
        lines.append("Usa `get_file(i)` para leer el archivo i. Usa `list_files()` para ver detalles.")
        return "\n".join(lines)

    def _tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "python_exec",
                    "description": "Ejecuta codigo Python en un entorno persistente. Usa `context` para leer el documento.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Codigo Python a ejecutar.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "final",
                    "description": "Finaliza la respuesta al usuario.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Respuesta final.",
                            }
                        },
                        "required": ["answer"],
                    },
                },
            },
        ]

    @staticmethod
    def _serialize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Serialize SDK tool call objects to plain dicts for the messages array."""
        serialized = []
        for tc in tool_calls:
            serialized.append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            })
        return serialized

    def run(
        self,
        question: str,
        file_entries: list[FileEntry] | None = None,
        total_chars: int = 0,
    ) -> str:
        self._run_start = time.time()

        user_content = question
        if file_entries:
            toc = self._build_toc(file_entries, total_chars)
            user_content = f"{toc}\n\n---\n\n{question}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        empty_turns = 0

        for turn in range(1, self.config.max_turns + 1):
            self._print()
            self._print(Rule(
                f"[bold]Turn {turn}/{self.config.max_turns}[/bold]  "
                f"[dim]subcalls={self.subcalls}/{self.config.max_subcalls}  "
                f"elapsed={self._elapsed()}[/dim]"
            ))

            t0 = time.time()
            response = self.client.chat(
                messages=messages,
                tools=self._tools(),
                tool_choice="auto",
                model=self.model,
                max_tokens=4096,
            )
            api_time = time.time() - t0

            n_tools = len(response.tool_calls)
            has_content = bool(response.content)
            self._print(
                f"  [dim]LLM responded in {api_time:.1f}s — "
                f"content={has_content} tool_calls={n_tools}[/dim]"
            )
            if has_content and response.content:
                preview = response.content[:200].replace("\n", " ")
                self._print(f"  [dim italic]» {preview}...[/dim italic]")

            if response.tool_calls:
                empty_turns = 0
                messages.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": self._serialize_tool_calls(response.tool_calls),
                    }
                )
                for call in response.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments or "{}")

                    if name == "python_exec":
                        code = args.get("code", "")

                        # Guardrail: reject code that's too long
                        line_count = code.count("\n") + 1
                        if line_count > 50:
                            self._print(
                                f"  [yellow]⚠ Code too long ({line_count} lines), "
                                f"rejecting[/yellow]"
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call.id,
                                    "content": (
                                        f"[error] Code too long ({line_count} lines). "
                                        "Max 50 lines. Simplify: use llm_query() "
                                        "instead of complex regex or parsing."
                                    ),
                                }
                            )
                            continue

                        self._print(
                            Panel(
                                code[:1500],
                                title=f"python_exec ({line_count}L)  [dim]{self._elapsed()}[/dim]",
                                expand=False,
                            )
                        )

                        t0 = time.time()
                        result = self.env.exec(code)
                        exec_time = time.time() - t0
                        stdout = result.get("stdout", "")
                        stderr = result.get("stderr", "")
                        ok = result.get("ok", False)

                        status = "[green]ok[/green]" if ok else "[red]FAIL[/red]"
                        self._print(
                            f"  {status} [dim]exec={exec_time:.1f}s  "
                            f"stdout={len(stdout)}ch  stderr={len(stderr)}ch[/dim]"
                        )

                        obs = ""
                        if stdout:
                            obs += f"[stdout]\n{stdout}\n"
                        if stderr:
                            obs += f"[stderr]\n{stderr}\n"
                        if not ok and stderr and "SyntaxError" in stderr:
                            obs += "\n[hint] SyntaxError: simplify your code. Use llm_query() instead of complex parsing.\n"
                        elif not ok and "Max subcalls reached" in stderr:
                            obs += "\n[hint] Max subcalls reached. Synthesize with the data you already have and call final.\n"
                        if not obs:
                            obs = "[no output]"

                        # Inject budget info so the model can plan
                        remaining_sub = self.config.max_subcalls - self.subcalls
                        remaining_turns = self.config.max_turns - turn
                        obs += (
                            f"\n[budget] subcalls restantes: {remaining_sub}/{self.config.max_subcalls} "
                            f"| turnos restantes: {remaining_turns}/{self.config.max_turns}\n"
                        )

                        if len(obs) > self.config.max_obs_chars:
                            obs = (
                                obs[: self.config.max_obs_chars]
                                + "\n...[truncated]\n"
                            )

                        self._print(
                            Panel(
                                obs[:2000],
                                title=f"python_exec result (ok={ok})",
                                expand=False,
                            )
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": obs,
                            }
                        )
                    elif name == "final":
                        answer = args.get("answer", "").strip()
                        if not answer and response.content:
                            answer = response.content.strip()
                        total = time.time() - self._run_start
                        self._print()
                        self._print(Rule("[bold green]Final Answer[/bold green]"))
                        self._print(
                            f"  [dim]Completed in {_fmt_elapsed(total)} — "
                            f"{turn} turns, {self.subcalls} subcalls[/dim]"
                        )
                        return answer or "[empty answer]"
                    else:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": f"[error] Unknown tool: {name}",
                            }
                        )
            else:
                # No tool calls — the model responded with plain text
                empty_turns += 1

                if response.content:
                    self._print(
                        Panel(response.content[:500], title="LLM text (no tool call)", expand=False)
                    )
                    # Only accept plain text as final answer on the very last turn
                    # or after repeated failures.  Otherwise nudge back to tools.
                    if empty_turns >= 3:
                        self._print("[yellow]3 turns without tool calls, returning text[/yellow]")
                        return response.content

                    remaining_sub = self.config.max_subcalls - self.subcalls
                    messages.append(
                        {"role": "assistant", "content": response.content},
                    )
                    if remaining_sub == 0:
                        self._print(
                            "  [yellow]⚠ Budget exhausted + text response — "
                            "nudging to call final()[/yellow]"
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Budget agotado. Llama AHORA a la herramienta `final(answer=...)` "
                                    "con los datos que ya tienes. NO uses python_exec."
                                ),
                            },
                        )
                    else:
                        self._print(
                            "  [yellow]⚠ Model responded with text instead of tools — "
                            "nudging back to tool use[/yellow]"
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"NO respondas con texto. Tienes {remaining_sub} subcalls disponibles. "
                                    "Usa python_exec para explorar `context` y luego llama a `final(answer=...)` "
                                    "con datos ESPECIFICOS del documento."
                                ),
                            },
                        )
                else:
                    # Truly empty response
                    if empty_turns >= 3:
                        self._print("[yellow]3 empty turns, aborting[/yellow]")
                        return "Model returned empty responses repeatedly."
                    messages.append(
                        {"role": "assistant", "content": ""},
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": "Usa la herramienta python_exec para explorar `context` y responde con la herramienta `final`.",
                        },
                    )

        total = time.time() - self._run_start
        self._print(
            f"\n[yellow]Max turns reached after {_fmt_elapsed(total)}[/yellow]"
        )
        return "Max turns reached without final answer."
