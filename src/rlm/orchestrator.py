from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from .llm import LLMClient
from .python_env import PythonEnv


SYSTEM_PROMPT = """\
Eres un RLM (Recursive Language Model). El documento completo NO esta en tu contexto.
El texto esta cargado en un entorno Python persistente como variable `context` (string).

## Herramientas en el entorno Python

- `context` : str completo del documento
- `get_slice(start, end)` -> str
- `search(pattern, max_results=5)` -> lista de matches con snippets
- `llm_query(prompt_text)` -> str   ← SUB-LLAMADA A OTRO LLM (consume 1 subcall)

## Reglas

1. **Codigo corto**: Maximo 25 lineas por python_exec. Nada de regex complejas.
2. **Usa llm_query() para analizar texto**: No intentes parsear contenido con regex.
   Ejemplo correcto:
     summary = llm_query(f"Resume en 1 frase la contribucion principal:\\n{fragment[:8000]}")
   Ejemplo INCORRECTO:
     m = re.search(r"(?:we propose|this paper)...", text)  # NO hagas esto
3. **Sintetiza**: Agrupa resultados en categorias/temas. No listes elementos uno por uno.
4. **Gestiona tu budget**: Cada llm_query() gasta 1 subcall. Reserva siempre al menos
   2 subcalls para la sintesis final. NO iteres sobre todas las secciones si son muchas;
   en su lugar, samplea un subconjunto representativo.
5. **Llama a final pronto**: Tienes turnos limitados. Cuando tengas suficiente, llama a `final`.

## Flujo recomendado

1. Explora (turno 1): `len(context)`, cuenta secciones, identifica separadores.
2. Samplea (turno 1-2): Si hay N secciones y B subcalls disponibles, analiza min(N, B-2)
   secciones distribuidas uniformemente (ej: cada N//budget secciones).
3. Sintetiza (ultimo turno): Con las respuestas recopiladas, agrupa por tema/categoria.
4. Responde: Llama a `final` con la sintesis.
"""


@dataclass
class OrchestratorConfig:
    max_turns: int = 12
    max_subcalls: int = 20
    max_obs_chars: int = 8000
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
        self._run_start: float = 0.0

        self.env.set_llm_query(self._llm_query)

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
        if self.subcalls >= self.config.max_subcalls:
            raise RuntimeError("Max subcalls reached.")
        self.subcalls += 1

        # Truncate very long prompts — GPT-5 returns null content on large inputs
        max_prompt = 6000
        if len(prompt_text) > max_prompt:
            prompt_text = prompt_text[:max_prompt] + "\n...[truncated]"

        self._print(
            f"  [cyan]⤷ llm_query #{self.subcalls}/{self.config.max_subcalls}[/cyan] "
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
        for attempt in range(2):
            response = self.client.chat(
                messages=messages,
                tools=None,
                tool_choice=None,
                model=self.submodel or self.model,
                max_tokens=800,
            )
            answer = response.content or ""
            if answer:
                break
            if attempt == 0:
                self._print("    [yellow]⟳ empty, retrying...[/yellow]")

        dt = time.time() - t0
        status = "[green]✓[/green]" if answer else "[yellow]∅[/yellow]"
        self._print(
            f"    {status} [dim]{dt:.1f}s — {len(answer)} chars[/dim]"
        )
        return answer

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

    def run(self, question: str) -> str:
        self._run_start = time.time()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
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
                if response.content:
                    self._print(
                        Panel(response.content[:500], title="LLM text response", expand=False)
                    )
                    return response.content

                # Empty response -- nudge the model to use tools
                empty_turns += 1
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
