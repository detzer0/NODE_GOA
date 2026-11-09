"""Command-line entry point: run one adaptive hierarchy and write the output.

    lgoa-run peaks --marking dorfler --dorfler-fraction 0.5 --output runs/peaks

Any preset field can additionally be overridden with ``--set name=value``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import load_dataset, load_preset, plots, run, self_check
from .backend import backend_report


def _parse_value(text: str):
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lgoa-run",
        description="Layerwise goal-oriented adaptive training of a neural ODE.",
    )
    parser.add_argument("preset", nargs="?", default="swiss_roll",
                        help="preset name (default: swiss_roll)")
    parser.add_argument("--marking", default=None,
                        choices=["conservative", "dorfler"])
    parser.add_argument("--dorfler-fraction", type=float, default=None,
                        help="bulk fraction for Dorfler marking")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="stationarity tolerance eps_stat")
    parser.add_argument("--reg-lambda", type=float, default=None,
                        help="H^1 regularization parameter")
    parser.add_argument("--initial-depth", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="model seed")
    parser.add_argument("--set", dest="extra", action="append", default=[],
                        metavar="NAME=VALUE",
                        help="override any other preset field (repeatable)")
    parser.add_argument("--output", type=Path, default=Path("runs"),
                        help="directory for figures and tables (default: runs)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides: dict = {}
    for flag, name in (
        ("marking", "marking"),
        ("dorfler_fraction", "dorfler_fraction"),
        ("tolerance", "stationarity_tolerance"),
        ("reg_lambda", "reg_lambda"),
        ("initial_depth", "initial_depth"),
        ("max_depth", "max_depth"),
        ("seed", "model_seed"),
    ):
        value = getattr(args, flag)
        if value is not None:
            overrides[name] = value
    for item in args.extra:
        if "=" not in item:
            raise SystemExit(f"--set expects NAME=VALUE, got {item!r}")
        name, _, value = item.partition("=")
        overrides[name.strip()] = _parse_value(value)

    config = load_preset(args.preset, **overrides)
    if not args.quiet:
        print(backend_report())
        print(config.summary(), end="\n\n")

    if not args.skip_checks:
        table = self_check()
        if not args.quiet:
            print(table.to_string(index=False), end="\n\n")

    data = load_dataset(config.problem)
    history = run(config, data, progress=not args.quiet)

    if not args.quiet:
        print()
        print(history.summary())

    args.output.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output)
    (args.output / "totals.json").write_text(
        json.dumps(history.totals(), indent=2, default=str), encoding="utf-8"
    )
    if not args.no_figures:
        plots.save_all(history, data, args.output)
    if not args.quiet:
        print(f"\nWrote results to {args.output.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
