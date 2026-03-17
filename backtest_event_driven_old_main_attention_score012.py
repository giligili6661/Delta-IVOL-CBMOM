import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest_event_driven_old_main as base
from backtest_event_driven_old_main_attention_penalty012_ivolnorm import NormalizedFactorEngine
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
DEFAULT_OUTPUT_PREFIX = "event_driven_old_main_attention_score012_full"
DEFAULT_PENALTY_WEIGHT = 0.12
DEFAULT_DELTA_WEIGHT = 0.50
DEFAULT_IVOL_WEIGHT = 0.50
DEFAULT_MAD_CLIP = 3.0


class ScoreAttentionTriggerEngine(base.EventTriggerEngine):
    def __init__(self, factor_engine, delta_weight=DEFAULT_DELTA_WEIGHT, ivol_weight=DEFAULT_IVOL_WEIGHT, backup_multiple=3, backup_extra=20):
        super().__init__(factor_engine)
        self.delta_weight = float(delta_weight)
        self.ivol_weight = float(ivol_weight)
        self.backup_multiple = max(int(backup_multiple), 1)
        self.backup_extra = max(int(backup_extra), 0)
        self.last_candidate_scores = {}
        self.last_candidate_meta = {}

    def _get_eligible_pool(self, factor_cache):
        if factor_cache is None or len(factor_cache) == 0:
            return pd.DataFrame()

        cutoff_low = max(int(len(factor_cache) * base.LONG_PCT), 1)
        low_ivol = factor_cache.sort_values("ivol_z").head(cutoff_low)
        converging = low_ivol[low_ivol["delta_ivol_raw"] < 0]
        positive = converging[converging["cbmom"] > 0].copy()
        return positive

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        self.last_candidate_scores = {}
        self.last_candidate_meta = {}

        if n_needed <= 0:
            return []

        eligible_pool = self._get_eligible_pool(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        # Lower z-scores are better, so negate the weighted sum to get a higher-is-better score.
        candidates["factor_score"] = -(
            self.delta_weight * candidates["delta_ivol_z"].astype(float)
            + self.ivol_weight * candidates["ivol_z"].astype(float)
        )
        candidates = candidates.sort_values(["factor_score", "delta_ivol_z", "ivol_z"], ascending=[False, True, True])

        backup_n = max(n_needed, n_needed * self.backup_multiple, n_needed + self.backup_extra)
        selected = candidates.head(backup_n).copy()
        self.last_candidate_scores = dict(zip(selected["code"], selected["factor_score"]))
        self.last_candidate_meta = {
            row["code"]: {
                "factor_score": float(row["factor_score"]),
                "delta_ivol_z": float(row["delta_ivol_z"]),
                "ivol_z": float(row["ivol_z"]),
            }
            for _, row in selected.iterrows()
        }
        return selected["code"].tolist()


class AttentionScoreBacktestEngine(AttentionVariantBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        delta_weight=DEFAULT_DELTA_WEIGHT,
        ivol_weight=DEFAULT_IVOL_WEIGHT,
        mad_clip=DEFAULT_MAD_CLIP,
        backup_multiple=3,
        backup_extra=20,
    ):
        super().__init__(
            data_loader,
            attention_panel,
            variant_name="attention_score_w012",
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            penalty_weight=penalty_weight,
        )
        self.factor_engine = NormalizedFactorEngine(data_loader.df_hfq, data_loader.df_ff3, mad_clip=mad_clip)
        self.trigger_engine = ScoreAttentionTriggerEngine(
            self.factor_engine,
            delta_weight=delta_weight,
            ivol_weight=ivol_weight,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )
        self.delta_weight = float(delta_weight)
        self.ivol_weight = float(ivol_weight)
        self.mad_clip = float(mad_clip)

    def _select_buys(self, date, candidate_codes, n_slots):
        if not candidate_codes or n_slots <= 0:
            return [], {}, self._empty_stats(len(candidate_codes), n_slots)

        date = pd.Timestamp(date)
        meta_by_code = {}
        scored = []
        fallback_last_idx = max(len(candidate_codes) - 1, 1)

        for idx, code in enumerate(candidate_codes):
            factor_meta = self.trigger_engine.last_candidate_meta.get(code, {})
            factor_score = float(
                factor_meta.get("factor_score", 1.0 - idx / fallback_last_idx)
            )
            crowding = self.attention_panel.get_crowding(date, code)
            penalty = 0.0 if pd.isna(crowding) else self.penalty_weight * float(crowding)
            combined_score = factor_score - penalty
            meta_by_code[code] = {
                "base_rank": int(idx + 1),
                "factor_score": factor_score,
                "delta_ivol_z": factor_meta.get("delta_ivol_z", np.nan),
                "ivol_z": factor_meta.get("ivol_z", np.nan),
                "crowding_pct": float(crowding) if pd.notna(crowding) else np.nan,
                "penalty": float(penalty),
                "combined_score": float(combined_score),
            }
            scored.append((code, float(combined_score), idx))

        scored.sort(key=lambda x: (-x[1], x[2]))
        selected = [code for code, _, _ in scored[:n_slots]]
        stats = {
            "candidate_count": int(len(candidate_codes)),
            "slot_count": int(n_slots),
            "selected_count": int(len(selected)),
            "avg_selected_crowding": float(np.nanmean([meta_by_code[c]["crowding_pct"] for c in selected]))
            if selected
            else np.nan,
            "avg_selected_factor_score": float(np.nanmean([meta_by_code[c]["factor_score"] for c in selected]))
            if selected
            else np.nan,
            "blocked_count": 0,
        }
        return selected, meta_by_code, stats


def plot_compare(nav_map, output_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    color_map = {
        "baseline": "#1f77b4",
        "attention_penalty012": "#ff7f0e",
        "attention_score012": "#2ca02c",
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
        "Old-main with attention-enhanced factor score\n"
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


def print_summary(summary):
    print("\n" + "=" * 60)
    print("  Performance")
    print("=" * 60)

    if not summary:
        print("No performance data.")
        return

    print(f"\nInitial capital: {base.INITIAL_CAPITAL:,.0f}")
    print(f"Final NAV:       {summary['final_nav']:,.0f}")
    print(f"Total return:    {summary['total_return']:.2%}")
    print(f"Annual return:   {summary['annual_return']:.2%}")
    print(f"Max drawdown:    {summary['max_drawdown']:.2%}")
    print(f"Sharpe:          {summary['sharpe']:.2f}")
    print(f"Backtest days:   {summary['backtest_days']}")

    print("\nTrade stats")
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Sells:        {summary['sell_trades']}")
    print(f"  Buys:         {summary['buy_trades']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Old-main with z-scored factor score and attention penalty 0.12")
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
        "--delta-weight",
        type=float,
        default=DEFAULT_DELTA_WEIGHT,
        help=f"Weight on z-scored delta_ivol in factor score. Default: {DEFAULT_DELTA_WEIGHT}",
    )
    parser.add_argument(
        "--ivol-weight",
        type=float,
        default=DEFAULT_IVOL_WEIGHT,
        help=f"Weight on z-scored ivol in factor score. Default: {DEFAULT_IVOL_WEIGHT}",
    )
    parser.add_argument(
        "--mad-clip",
        type=float,
        default=DEFAULT_MAD_CLIP,
        help=f"MAD winsorization multiple for ivol and delta_ivol. Default: {DEFAULT_MAD_CLIP}",
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
    print("  old_main + z-scored factor score + attention penalty")
    print(f"  Window: {base.START_DATE} ~ {base.END_DATE}")
    print(f"  Target positions: {base.TARGET_POSITIONS}")
    print(f"  Delta weight: {args.delta_weight:.2f}")
    print(f"  IVOL weight: {args.ivol_weight:.2f}")
    print(f"  Attention penalty: {args.penalty_weight:.2f}")
    print(f"  MAD clip: {args.mad_clip:.2f}")
    print(f"  Slippage: {base.SLIPPAGE_BPS_PER_SIDE:.2f} bps/side")
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
    loader.load_hs300()
    loader.compute_returns()

    print("\n[baseline] running old_main...")
    baseline_engine = base.EventDrivenBacktestEngine(loader)
    baseline_engine.run()
    baseline_summary = compute_summary(baseline_engine.daily_nav, baseline_engine.trade_log)

    print("\n[attention_penalty012] running current version...")
    vanilla_engine = AttentionVariantBacktestEngine(
        loader,
        attention_panel,
        variant_name="attention_penalty_w012",
        backup_multiple=args.backup_multiple,
        backup_extra=args.backup_extra,
        penalty_weight=args.penalty_weight,
    )
    vanilla_engine.run()
    vanilla_summary = compute_summary(vanilla_engine.daily_nav, vanilla_engine.trade_log)

    print("\n[attention_score012] running z-scored score version...")
    score_engine = AttentionScoreBacktestEngine(
        loader,
        attention_panel,
        penalty_weight=args.penalty_weight,
        delta_weight=args.delta_weight,
        ivol_weight=args.ivol_weight,
        mad_clip=args.mad_clip,
        backup_multiple=args.backup_multiple,
        backup_extra=args.backup_extra,
    )
    score_engine.run()
    score_summary = compute_summary(score_engine.daily_nav, score_engine.trade_log)
    print_summary(score_summary)

    compare_rows = [
        {"strategy": "baseline", **baseline_summary},
        {
            "strategy": "attention_penalty012",
            **vanilla_summary,
            "baseline_total_return": baseline_summary["total_return"],
            "baseline_annual_return": baseline_summary["annual_return"],
            "baseline_max_drawdown": baseline_summary["max_drawdown"],
            "baseline_sharpe": baseline_summary["sharpe"],
            "total_return_diff": vanilla_summary["total_return"] - baseline_summary["total_return"],
            "annual_return_diff": vanilla_summary["annual_return"] - baseline_summary["annual_return"],
            "max_drawdown_diff": vanilla_summary["max_drawdown"] - baseline_summary["max_drawdown"],
            "sharpe_diff": vanilla_summary["sharpe"] - baseline_summary["sharpe"],
        },
        {
            "strategy": "attention_score012",
            **score_summary,
            "baseline_total_return": baseline_summary["total_return"],
            "baseline_annual_return": baseline_summary["annual_return"],
            "baseline_max_drawdown": baseline_summary["max_drawdown"],
            "baseline_sharpe": baseline_summary["sharpe"],
            "total_return_diff": score_summary["total_return"] - baseline_summary["total_return"],
            "annual_return_diff": score_summary["annual_return"] - baseline_summary["annual_return"],
            "max_drawdown_diff": score_summary["max_drawdown"] - baseline_summary["max_drawdown"],
            "sharpe_diff": score_summary["sharpe"] - baseline_summary["sharpe"],
            "vanilla_total_return": vanilla_summary["total_return"],
            "vanilla_annual_return": vanilla_summary["annual_return"],
            "vanilla_max_drawdown": vanilla_summary["max_drawdown"],
            "vanilla_sharpe": vanilla_summary["sharpe"],
            "vanilla_total_return_diff": score_summary["total_return"] - vanilla_summary["total_return"],
            "vanilla_annual_return_diff": score_summary["annual_return"] - vanilla_summary["annual_return"],
            "vanilla_max_drawdown_diff": score_summary["max_drawdown"] - vanilla_summary["max_drawdown"],
            "vanilla_sharpe_diff": score_summary["sharpe"] - vanilla_summary["sharpe"],
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

    if score_engine.trade_log:
        pd.DataFrame(score_engine.trade_log).to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"Trade log exported: {trades_path} ({len(score_engine.trade_log)} rows)")

    if score_engine.daily_status:
        pd.DataFrame(score_engine.daily_status).to_csv(status_path, index=False, encoding="utf-8-sig")
        print(f"Daily status exported: {status_path}")

    if score_engine.daily_nav:
        pd.DataFrame(score_engine.daily_nav, columns=["date", "nav"]).to_csv(nav_path, index=False, encoding="utf-8-sig")
        print(f"NAV exported: {nav_path}")

    plot_compare(
        {
            "baseline": pd.DataFrame(baseline_engine.daily_nav, columns=["date", "nav"]),
            "attention_penalty012": pd.DataFrame(vanilla_engine.daily_nav, columns=["date", "nav"]),
            "attention_score012": pd.DataFrame(score_engine.daily_nav, columns=["date", "nav"]),
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
            "delta_weight": float(args.delta_weight),
            "ivol_weight": float(args.ivol_weight),
            "mad_clip": float(args.mad_clip),
            "attention_cache": os.path.abspath(args.attention_cache),
            "backup_multiple": int(args.backup_multiple),
            "backup_extra": int(args.backup_extra),
        },
        "attention_dates": {
            "start": attention_panel.dates.min().strftime("%Y-%m-%d"),
            "end": attention_panel.dates.max().strftime("%Y-%m-%d"),
        },
        "baseline": baseline_summary,
        "attention_penalty012": vanilla_summary,
        "attention_score012": score_summary,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Summary exported: {summary_path}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
