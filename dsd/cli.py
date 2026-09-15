"""Command line entry point.

    python -m dsd run                    # every stage, in order
    python -m dsd run --stages vad,embed # a slice of the pipeline
    python -m dsd diarize --limit 20     # one stage
    python -m dsd backends               # what is registered
    python -m dsd verify --sample 200

Global flags come before the subcommand:

    python -m dsd --config configs/default.yaml --set build.chunks=true build
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import backends  # noqa: F401  (registers model backends)
from . import clustering  # noqa: F401  (registers clustering backends)
from .config import Config
from .registry import CLUSTERERS, DIARIZERS, EMBEDDERS, ENHANCERS, GENDER, VADS
from .stages import BY_NAME, ORDER, PIPELINE

DEFAULT_CONFIG = "configs/default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsd",
        description="Build a 2-speaker separation dataset from dual-channel telephony.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file")
    parser.add_argument("--root", help="project root; defaults to the config file's directory")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set build.chunks=true (repeatable)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run several stages in order")
    run_parser.add_argument(
        "--stages",
        help=f"comma-separated subset of: {', '.join(PIPELINE)} (default: all)",
    )
    run_parser.add_argument("--from", dest="start", help="start at this stage and continue")
    run_parser.add_argument("--limit", type=int, help="passed to every stage that accepts it")
    run_parser.add_argument("--overwrite", action="store_true", help="passed to every stage")
    run_parser.add_argument("--chunks", action="store_true", help="build fixed-length windows too")
    run_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="build: roll s2 to synthesize speech overlap (off by default)",
    )

    for module in ORDER:
        stage_parser = subparsers.add_parser(module.NAME, help=(module.__doc__ or "").split("\n")[0])
        module.add_args(stage_parser)

    subparsers.add_parser("backends", help="list registered backends")
    subparsers.add_parser("config", help="print the resolved configuration")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    root = args.root or (config_path.parent.parent if config_path.exists() else ".")
    cfg = Config.load(config_path if config_path.exists() else None, args.overrides, root=root)

    if args.command == "backends":
        for label, registry in (
            ("diarizer", DIARIZERS),
            ("vad", VADS),
            ("embedder", EMBEDDERS),
            ("gender", GENDER),
            ("clusterer", CLUSTERERS),
            ("enhancer", ENHANCERS),
        ):
            print(f"{label:<10} {', '.join(registry.names())}")
        return 0

    if args.command == "config":
        import json

        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "run":
        return _run_pipeline(cfg, args, parser)

    module = BY_NAME[args.command]
    _run_stage(module, cfg, args)
    return 0


def _run_pipeline(cfg, args, parser) -> int:
    if args.stages:
        names = [name.strip() for name in args.stages.split(",") if name.strip()]
    elif args.start:
        if args.start not in PIPELINE:
            parser.error(f"--from {args.start!r} is not one of: {', '.join(PIPELINE)}")
        names = PIPELINE[PIPELINE.index(args.start) :]
    else:
        names = list(PIPELINE)

    unknown = [name for name in names if name not in BY_NAME]
    if unknown:
        parser.error(f"unknown stage(s): {', '.join(unknown)}")

    print(f"[run] {' -> '.join(names)}")
    for name in names:
        module = BY_NAME[name]
        # Each stage parses its own flags, so build a namespace holding that
        # stage's defaults and copy over only the shared flags it declares.
        stage_parser = argparse.ArgumentParser(prog=name, add_help=False)
        module.add_args(stage_parser)
        stage_args = stage_parser.parse_args([])
        for shared in ("limit", "overwrite", "chunks", "shuffle"):
            if hasattr(stage_args, shared) and getattr(args, shared, None):
                setattr(stage_args, shared, getattr(args, shared))
        _run_stage(module, cfg, stage_args)
    return 0


def _run_stage(module, cfg, args) -> None:
    print(f"\n{'=' * 70}\n== {module.NAME}\n{'=' * 70}")
    started = time.time()
    module.run(cfg, args)
    print(f"[{module.NAME}] finished in {time.time() - started:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
