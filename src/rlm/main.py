from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import load_azure_config
from .llm import LLMClient
from .orchestrator import OrchestratorConfig, RLMOrchestrator
from .python_env import PythonEnv
from .utils import load_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlm")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run RLM demo")
    run.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input file path or glob. Use multiple --input flags.",
    )
    run.add_argument("--question", required=True, help="Question for the RLM")
    run.add_argument("--max-turns", type=int, default=12)
    run.add_argument("--max-subcalls", type=int, default=20)
    run.add_argument("--max-obs-chars", type=int, default=8000)
    run.add_argument("--temperature", type=float, default=0.2)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    console = Console()

    if args.command != "run":
        console.print("Unknown command.")
        sys.exit(1)

    load_dotenv()
    try:
        config = load_azure_config()
    except RuntimeError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        sys.exit(1)

    doc = load_documents(args.input)
    env = PythonEnv()
    env.set_context(doc.text)

    table = Table(title="Document Loaded")
    table.add_column("Sources")
    table.add_column("Chars")
    table.add_column("Token Est.")
    table.add_row(", ".join(doc.sources), str(doc.char_len), str(doc.token_estimate))
    console.print(table)

    client = LLMClient(config)
    orchestrator = RLMOrchestrator(
        client=client,
        env=env,
        config=OrchestratorConfig(
            max_turns=args.max_turns,
            max_subcalls=args.max_subcalls,
            max_obs_chars=args.max_obs_chars,
            temperature=args.temperature,
        ),
        console=console,
        model=config.deployment_name,
        submodel=config.submodel_name,
    )

    answer = orchestrator.run(args.question)
    console.print(Panel(answer, title="Final Answer", expand=False))


if __name__ == "__main__":
    main()
