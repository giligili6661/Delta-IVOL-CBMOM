import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd

import backtest_event_driven_old_main as base
from backtest_event_driven_old_main_attention_variants import (
    AttentionVariantBacktestEngine,
    DEFAULT_CACHE_NAME,
    INTRADAY_DIR,
    build_attention_panel,
    compute_summary,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-03-06"
DEFAULT_OUTPUT_PREFIX = "event_driven_old_main_attention_penalty012_maxbuy10_full"
DEFAULT_PENALTY_WEIGHT = 0.12
DEFAULT_MAX_NEW_BUYS = 10


class DailyCappedAttentionPenaltyBacktestEngine(AttentionVariantBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        max_new_buys_per_day=DEFAULT_MAX_NEW_BUYS,
        backup_multiple=3,
        backup_extra=20,
    ):
        self.max_new_buys_per_day = None if max_new_buys_per_day is None else int(max_new_buys_per_day)
        suffix = "all" if self.max_new_buys_per_day is None else str(self.max_new_buys_per_day)
        super().__init__(
            data_loader,
            attention_panel,
            variant_name=f"attention_penalty_w012_maxbuy{suffix}",
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            penalty_weight=penalty_weight,
        )

    def _select_buys(self, date, candidate_codes, n_slots):
        effective_slots = n_slots
        if self.max_new_buys_per_day is not None and self.max_new_buys_per_day > 0:
            effective_slots = min(n_slots, self.max_new_buys_per_day)
        return super()._select_buys(date, candidate_codes, effective_slots)


def plot_compare(nav_map, output_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    color_map = {
        "baseline": "#1f77b4",
        "attention_penalty012_maxbuy10": "#ff7f0e",
    }
    for name, nav_df in nav_map.items():
        if nav_df.empty:
            continue
        ax.plot(
            nav_df["date"],
            nav_df["nav"] / base.INITIAL_CAPITAL,
            linewidth=1.5,
            color=color_map.get(name, None),
            label=name,
        )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_title(
        "Old-main + attention_penalty(0.12) + capped daily new buys\n"
        f"{base.START_DATE} ~ {base.END_DATE}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("NAV (start=1.0)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Old-main attention penalty 0.12 with capped daily new buys")
    parser.add_argument("--start", default=DEFAULT_START, help="Backtest start date in YYYY-MM-DD format.")
    parser.add_argument("--end", default=DEFAULT_END, help="Backtest end date in YYYY-MM-DD format.")
    parser.add_argument(
        "--positions",
        type=int,
        default=base.TARGET_POSITIONS,
        help=f"Target number of holdings. Default: {base.TARGET_POSITIONS}",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=base.SLIPPAGE_BPS_PER_SIDE,
        help=f"Per-side slippage in bps applied to open-price execution. Default: {base.SLIPPAGE_BPS_PER_SIDE}",
    )
    parser.add_argument(
        "--intraday-dir",
        default=str(INTRADAY_DIR),
        help=f"15m raw directory. Default: {INTRADAY_DIR}",
    )
    parser.add_argument(
        "--attention-cache",
        default=os.path.join(BASE_DIR, DEFAULT_CACHE_NAME),
        help="Path to cached intraday attention panel npz.",
    )
    parser.add_argument("--backup-multiple", type=int, default=3, help="Backup multiple. Default: 3")
    parser.add_argument("--backup-extra", type=int, default=20, help="Backup extra candidates. Default: 20")
    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=DEFAULT_PENALTY_WEIGHT,
        help=f"Attention crowding penalty weight. Default: {DEFAULT_PENALTY_WEIGHT}",
    )
    parser.add_argument(
        "--max-new-buys",
        type=int,
        default=DEFAULT_MAX_NEW_BUYS,
        help=f"Maximum number of new buys executed per trading day. Default: {DEFAULT_MAX_NEW_BUYS}",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Prefix used for output files. Default: {DEFAULT_OUTPUT_PREFIX}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base.set_backtest_window(args.start, args.end)
    base.set_target_positions(args.positions)
    base.set_slippage_bps(args.slippage_bps)
    base.tqdm = lambda iterable, **kwargs: iterable

    t0 = time.time()
    print("=" * 60)
    print("  Event-driven backtest")
    print("  old_main + attention_penalty(0.12) + max_new_buys")
    print(f"  Window: {base.START_DATE} ~ {base.END_DATE}")
    print(f"  Target positions: {base.TARGET_POSITIONS}")
    print(f"  Penalty weight: {args.penalty_weight:.2f}")
    print(f"  Max new buys/day: {args.max_new_buys}")
    print("=" * 60)

    attention_panel = build_attention_panel(
        cache_path=args.attention_cache,
        start_date=base.START_DATE,
        end_date=base.END_DATE,
        intraday_dir=args.intraday_dir,
    )

    loader = base.DataLoader()
    loader.load_stock_data()
    loader.load_ff3_factors()
    loader.compute_returns()

    print("\n[baseline] running old_main...")
    baseline_engine = base.EventDrivenBacktestEngine(loader)
    baseline_engine.run()
    baseline_summary = compute_summary(baseline_engine.daily_nav, baseline_engine.trade_log)

    print("\n[attention_penalty012_maxbuy] running capped-buy version...")
    strategy_engine = DailyCappedAttentionPenaltyBacktestEngine(
        loader,
        attention_panel,
        penalty_weight=args.penalty_weight,
        max_new_buys_per_day=args.max_new_buys,
        backup_multiple=args.backup_multiple,
        backup_extra=args.backup_extra,
    )
    strategy_engine.run()
    strategy_summary = compute_summary(strategy_engine.daily_nav, strategy_engine.trade_log)

    compare_rows = [
        {"strategy": "baseline", **baseline_summary},
        {
            "strategy": f"attention_penalty012_maxbuy{args.max_new_buys}",
            **strategy_summary,
            "baseline_total_return": baseline_summary["total_return"],
            "baseline_annual_return": baseline_summary["annual_return"],
            "baseline_max_drawdown": baseline_summary["max_drawdown"],
            "baseline_sharpe": baseline_summary["sharpe"],
            "total_return_diff": strategy_summary["total_return"] - baseline_summary["total_return"],
            "annual_return_diff": strategy_summary["annual_return"] - baseline_summary["annual_return"],
            "max_drawdown_diff": strategy_summary["max_drawdown"] - baseline_summary["max_drawdown"],
            "sharpe_diff": strategy_summary["sharpe"] - baseline_summary["sharpe"],
        },
    ]
    compare_df = pd.DataFrame(compare_rows)

    compare_path = os.path.join(BASE_DIR, f"{args.output_prefix}_compare_summary.csv")
    trades_path = os.path.join(BASE_DIR, f"{args.output_prefix}_trades.csv")
    status_path = os.path.join(BASE_DIR, f"{args.output_prefix}_daily_status.csv")
    nav_path = os.path.join(BASE_DIR, f"{args.output_prefix}_nav.csv")
    plot_path = os.path.join(BASE_DIR, f"{args.output_prefix}_compare_nav.png")
    summary_path = os.path.join(BASE_DIR, f"{args.output_prefix}_summary.json")

    compare_df.to_csv(compare_path, index=False, encoding="utf-8-sig")
    print(f"Compare summary exported: {compare_path}")

    if strategy_engine.trade_log:
        pd.DataFrame(strategy_engine.trade_log).to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"Trade log exported: {trades_path} ({len(strategy_engine.trade_log)} rows)")

    if strategy_engine.daily_status:
        pd.DataFrame(strategy_engine.daily_status).to_csv(status_path, index=False, encoding="utf-8-sig")
        print(f"Daily status exported: {status_path}")

    if strategy_engine.daily_nav:
        pd.DataFrame(strategy_engine.daily_nav, columns=["date", "nav"]).to_csv(nav_path, index=False, encoding="utf-8-sig")
        print(f"NAV exported: {nav_path}")

    plot_compare(
        {
            "baseline": pd.DataFrame(baseline_engine.daily_nav, columns=["date", "nav"]),
            f"attention_penalty012_maxbuy{args.max_new_buys}": pd.DataFrame(
                strategy_engine.daily_nav, columns=["date", "nav"]
            ),
        },
        plot_path,
    )
    print(f"Compare plot exported: {plot_path}")

    payload = {
        "params": {
            "start": base.START_DATE,
            "end": base.END_DATE,
            "positions": base.TARGET_POSITIONS,
            "slippage_bps": base.SLIPPAGE_BPS_PER_SIDE,
            "penalty_weight": float(args.penalty_weight),
            "max_new_buys": int(args.max_new_buys),
            "attention_cache": os.path.abspath(args.attention_cache),
            "backup_multiple": int(args.backup_multiple),
            "backup_extra": int(args.backup_extra),
        },
        "attention_dates": {
            "start": attention_panel.dates.min().strftime("%Y-%m-%d"),
            "end": attention_panel.dates.max().strftime("%Y-%m-%d"),
        },
        "baseline": baseline_summary,
        "attention_penalty012_maxbuy": strategy_summary,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Summary exported: {summary_path}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
