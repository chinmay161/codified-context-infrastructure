"""CLI for the deterministic Smart Factory."""

from __future__ import annotations

import argparse

from .smart_factory import SmartFactory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctx-factory",
        description="Initialize a deterministic Context Engine project scaffold.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new Context Engine project.")
    init_parser.add_argument("project_name", help="Directory name for the generated project.")
    init_parser.add_argument("--domain", required=True, help="Short domain description for the knowledge base.")
    init_parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=".",
        help="Parent directory where the project will be created.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command != "init":
        raise SystemExit(2)

    result = SmartFactory(
        project_name=args.project_name,
        domain=args.domain,
        base_path=args.output_dir,
    ).run()
    print(result["summary"])
    print(f"Project root: {result['project_root']}")
    print("Next steps:")
    print(f"1. cd {args.project_name}")
    print("2. Review ai-context/, schemas/, and config/factory_config.yaml")
    print("3. Point your Context Engine at this project root")
