"""Command line entry point.

``optlab audit`` reproduces every number in the report from a raw snapshot,
which is the only way to keep a written claim and the code that produced it
from drifting apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from optlab.pipeline import audit_snapshot

__all__ = ["main", "build_parser"]

DEFAULT_OUTPUT = Path("reports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="optlab", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="run the no-arbitrage audit on a snapshot")
    audit.add_argument("chain", type=Path, help="raw option-chain CSV")
    audit.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directory for CSV output")
    audit.add_argument(
        "--forward-band",
        type=float,
        default=0.05,
        help="half-width in log-moneyness of the parity regression window",
    )
    audit.add_argument(
        "--naive-rate",
        type=float,
        default=0.04,
        help="the fixed rate whose convention error is being measured",
    )
    audit.add_argument(
        "--discount-rate",
        type=float,
        default=0.043,
        help="externally observed money-market rate used to build discount factors",
    )
    audit.add_argument(
        "--exercise-style",
        choices=("american", "european"),
        default="american",
        help="bounds convention; listed US equity options are American",
    )
    audit.add_argument("--quiet", action="store_true", help="write files without printing tables")
    audit.add_argument(
        "--figures", type=Path, default=Path("figures"), help="directory for figures"
    )
    audit.add_argument("--no-figures", action="store_true", help="skip plotting entirely")

    subparsers.add_parser("bench", help="benchmark the implied-volatility solvers")

    fetch = subparsers.add_parser("fetch", help="download a fresh option-chain snapshot")
    fetch.add_argument("--ticker", default="SPY")
    fetch.add_argument("--output", type=Path, default=Path("data/raw"))
    fetch.add_argument(
        "--target-days",
        type=int,
        nargs="+",
        default=[7, 14, 30, 60, 90],
        help="maturities to pick the nearest listed expiry to",
    )
    return parser


def _print_section(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if frame.empty:
        print("(none)")
        return
    with pd.option_context("display.width", 160, "display.max_columns", 40):
        print(frame.to_string(index=False))


def _run_audit(args: argparse.Namespace) -> int:
    result = audit_snapshot(
        str(args.chain),
        forward_band=args.forward_band,
        naive_rate=args.naive_rate,
        discount_rate=args.discount_rate,
        exercise_style=args.exercise_style,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "forward_curve.csv": result.forwards,
        "forward_curve_joint_fit.csv": result.forwards_joint,
        "liquidity_breakdown.csv": result.liquidity,
        "violation_funnel.csv": result.funnel,
        "staleness_sensitivity.csv": result.staleness,
        "exercise_style_comparison.csv": result.exercise_styles,
        "violations.csv": result.violations,
        "violation_summary.csv": result.violation_summary,
        "survival.csv": result.survival,
        "parity_residuals.csv": result.parity,
        "box_implied_rates.csv": result.box_rates,
        "iv_convention_comparison.csv": result.iv_comparison,
        "iv_gap_summary.csv": result.iv_gap_summary,
        "quote_quality.csv": result.quote_quality,
        "filter_report.csv": result.filter_report.to_frame(),
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False)

    if not args.no_figures:
        from optlab.report.plots import make_all_figures

        written = make_all_figures(result, args.figures)
        print(f"wrote {len(written)} figures to {args.figures}/")

    if not args.quiet:
        _print_section("Implied forward curve", result.forwards.round(4))
        _print_section("Quote quality near the money", result.quote_quality.round(4))
        _print_section("Violations by check and basis", result.violation_summary.round(4))
        _print_section("Survival across the spread", result.survival.round(4))
        _print_section("Where the violations live", result.liquidity.round(4))
        _print_section("Violation funnel", result.funnel.round(4))
        _print_section("Sensitivity to the staleness cut", result.staleness.round(4))
        _print_section("American vs European bounds", result.exercise_styles.round(4))
        _print_section("Call/put IV gap by convention", result.iv_gap_summary.round(4))
        print("\nHeadline")
        print("--------")
        for key, value in result.headline().items():
            print(f"{key:>34}: {value:,.4f}")
    print(f"\nwrote {len(outputs)} files to {args.output}/")
    return 0


def _run_fetch(args: argparse.Namespace) -> int:
    from optlab.market.sources.yahoo import download_chain

    frame = download_chain(args.ticker, target_days=args.target_days)
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    destination = args.output / f"{args.ticker}_{stamp}_chain.csv"
    frame.to_csv(destination, index=False)
    print(f"wrote {len(frame):,} rows to {destination}")
    return 0


def _run_bench(_: argparse.Namespace) -> int:
    from optlab.report.benchmark import benchmark_solvers

    frame = benchmark_solvers()
    _print_section("Implied-volatility solver benchmark", frame.round(10))
    Path("reports").mkdir(exist_ok=True)
    frame.to_csv("reports/solver_benchmark.csv", index=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return _run_audit(args)
    if args.command == "fetch":
        return _run_fetch(args)
    if args.command == "bench":
        return _run_bench(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
