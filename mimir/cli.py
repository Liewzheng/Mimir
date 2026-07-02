"""Command-line interface for Mimir."""

import argparse
import sys
from pathlib import Path

from mimir.application.factories import create_embedding_engine, create_mimir
from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.domain.model.engine import EmbeddingEngine
from mimir.setup import create_setup, list_supported_agents


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mimir",
        description="Mimir: a plastic embedding system that remembers.",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--backend",
        choices=["llama-server", "sentence-transformer", "fake"],
        default="llama-server",
        help="Embedding backend to use",
    )
    common.add_argument(
        "--base-url",
        default="http://127.0.0.1:11435",
        help="Base URL for llama-server backend",
    )
    common.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Model name for sentence-transformer backend",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("mcp", parents=[common], help="Start MCP stdio server")

    encode_parser = subparsers.add_parser("encode", parents=[common], help="Encode text(s)")
    encode_parser.add_argument("text", nargs="+", help="Text to encode")

    learn_parser = subparsers.add_parser("learn", parents=[common], help="Learn from text(s)")
    learn_parser.add_argument("text", nargs="+", help="Text to learn")
    learn_parser.add_argument(
        "--importance",
        type=float,
        default=1.0,
        help="Learning importance multiplier",
    )

    save_parser = subparsers.add_parser("save", parents=[common], help="Save Mimir state")
    save_parser.add_argument("--path", required=True, help="Checkpoint path")

    load_parser = subparsers.add_parser("load", parents=[common], help="Load Mimir state")
    load_parser.add_argument("--path", required=True, help="Checkpoint path")

    setup_parser = subparsers.add_parser("setup", help="Configure Mimir hooks for an agent CLI (kimi-code, claude-code, codex)")
    setup_parser.add_argument(
        "agent",
        choices=list_supported_agents(),
        help="Agent CLI to configure: kimi-code, claude-code, or codex",
    )
    setup_parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Override the agent's config directory",
    )

    return parser


def _create_engine(args: argparse.Namespace) -> EmbeddingEngine:
    """Create an embedding engine from CLI arguments."""
    return create_embedding_engine(
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
    )


def _create_mimir(args: argparse.Namespace) -> Mimir:
    """Create an Mimir instance from CLI arguments."""
    engine = _create_engine(args)
    config = MimirConfig(
        base_model=args.model if args.backend == "sentence-transformer" else args.base_url,
        num_prototypes=64,
        learning_rate_base=0.05,
    )
    return create_mimir(config, engine=engine)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "mcp":
            from mimir.mcp.server import run_server

            run_server(
                backend=args.backend,
                base_url=args.base_url,
                model=args.model,
            )
            return 0

        if args.command == "encode":
            mimir = _create_mimir(args)
            embeddings = mimir.encode(args.text)
            print(f"Shape: {list(embeddings.shape)}")
            print(embeddings)

        elif args.command == "learn":
            mimir = _create_mimir(args)
            report = mimir.learn(args.text, importance=args.importance)
            print(report)

        elif args.command == "save":
            mimir = _create_mimir(args)
            mimir.learn(["initialization"])
            mimir.save(Path(args.path))
            print(f"Saved to {args.path}")

        elif args.command == "load":
            mimir = _create_mimir(args)
            mimir.load(Path(args.path))
            print(f"Loaded from {args.path}; step={mimir.step}")

        elif args.command == "setup":
            setup = create_setup(args.agent, base_dir=args.base_dir)
            path = setup.install()
            print(f"Configured Mimir hooks for {args.agent}: {path}")

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
