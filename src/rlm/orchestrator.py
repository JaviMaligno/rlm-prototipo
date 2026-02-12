from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel

from .llm import LLMClient
from .python_env import PythonEnv


SYSTEM_PROMPT = """\
Eres un RLM (Recursive Language Model). El documento completo NO esta en tu contexto.
El texto esta cargado en un entorno Python persistente como variable `context` (string).

## Herramientas en el entorno Python

- `context` : str completo del documento
- `get_slice(start, end)` -> str
- `search(pattern, max_results=5)` -> lista de matches con snippets
- `llm_query(prompt_text)` -> str   ← SUB-LLAMADA A OTRO LLM

## Reglas

1. **Codigo corto**: Maximo 25 lineas por python_exec. Nada de regex complejas.
2. **Usa llm_query() para analizar texto**: No intentes parsear contenido con regex.
   Ejemplo correcto:
     summary = llm_query(f"Resume en 1 frase la contribucion principal:\\n{fragment[:8000]}")
   Ejemplo INCORRECTO:
     m = re.search(r"(?:we propose|this paper)...", text)  # NO hagas esto
3. **Sintetiza**: Agrupa resultados en categorias/temas. No listes elementos uno por uno.
4. **Llama a final pronto**: Tienes turnos limitados. Cuando tengas suficiente, llama a `final`.

## Flujo recomendado

1. Explora: `len(context)`, identifica secciones (busca separadores como "===== FILE:").
2. Analiza: Itera sobre secciones, llama a `llm_query()` por cada una con un fragmento.
3. Sintetiza: Agrupa las respuestas por tema/categoria.
4. Responde: Llama a `final` con la sintesis.
"""


@dataclass
class OrchestratorConfig:
    max_turns: int = 12
    max_subcalls: int = 20
    max_obs_chars: int = 8000
    temperature: float = 0.2


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
        self.console = console or Console()
        self.model = model
        self.submodel = submodel
        self.subcalls = 0

        self.env.set_llm_query(self._llm_query)

    def _llm_query(self, prompt_text: str) -> str:
        if self.subcalls >= self.config.max_subcalls:
            raise RuntimeError("Max subcalls reached.")
        self.subcalls += 1

        self.console.print(
            Panel(
                f"Subcall #{self.subcalls}\n{prompt_text[:500]}",
                title="llm_query",
                expand=False,
            )
        )

        messages = [
            {
                "role": "system",
                "content": "Responde de forma concisa y precisa usando solo el fragmento dado.",
            },
            {"role": "user", "content": prompt_text},
        ]
        response = self.client.chat(
            messages=messages,
            tools=None,
            tool_choice=None,
            model=self.submodel or self.model,
            max_tokens=800,
        )
        return response.content or ""

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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        empty_turns = 0

        for turn in range(1, self.config.max_turns + 1):
            self.console.print(Panel(f"Turn {turn}", title="LLM", expand=False))
            response = self.client.chat(
                messages=messages,
                tools=self._tools(),
                tool_choice="auto",
                model=self.model,
                max_tokens=4096,
            )

            self.console.print(
                f"  [dim]content={bool(response.content)} "
                f"tool_calls={len(response.tool_calls)}[/dim]"
            )

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
                            self.console.print(
                                f"[yellow]Code too long ({line_count} lines), "
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

                        self.console.print(
                            Panel(code[:1500], title=f"python_exec code ({line_count}L)", expand=False)
                        )
                        result = self.env.exec(code)
                        stdout = result.get("stdout", "")
                        stderr = result.get("stderr", "")
                        ok = result.get("ok", False)

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

                        if len(obs) > self.config.max_obs_chars:
                            obs = (
                                obs[: self.config.max_obs_chars]
                                + "\n...[truncated]\n"
                            )

                        self.console.print(
                            Panel(
                                obs[:2000],
                                title=f"python_exec (ok={ok})",
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
                    self.console.print(
                        Panel(response.content[:500], title="LLM text response", expand=False)
                    )
                    return response.content

                # Empty response -- nudge the model to use tools
                empty_turns += 1
                if empty_turns >= 3:
                    self.console.print("[yellow]3 empty turns, aborting[/yellow]")
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

        return "Max turns reached without final answer."
