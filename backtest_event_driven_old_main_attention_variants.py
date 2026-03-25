import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest_event_driven_old_main as base
from intraday_behavior_factor_ic_validation import (
    BASE_DIR as FACTOR_BASE_DIR,
    DEFAULT_END as DEFAULT_FACTOR_END,
    INTRADAY_DIR,
    build_intraday_behavior_matrices,
    is_equity_code,
    load_master_intraday_calendar,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-03-06"
DEFAULT_OUTPUT_PREFIX = "event_driven_old_main_attention_variants_full"
DEFAULT_CACHE_NAME = "intraday_attention_panel_20200102_20260318.npz"


def compute_summary(daily_nav, trade_log):
    nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"])
    if nav_df.empty:
        return {}

    nav = nav_df["nav"].values
    total_ret = float(nav[-1] / nav[0] - 1.0)
    n_years = (nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days / 365.25
    annual_ret = float((1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0)
    daily_rets = np.diff(nav) / nav[:-1]
    sharpe = float(
        (np.mean(daily_rets) - base.RISK_FREE_RATE / 252.0) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)
    )
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak
    max_dd = float(drawdown.min())

    return {
        "start": nav_df["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": nav_df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "backtest_days": int(len(nav)),
        "final_nav": float(nav[-1]),
        "total_return": total_ret,
        "annual_return": annual_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "total_trades": int(len(trade_log)),
        "buy_trades": int(sum(1 for t in trade_log if t["action"] == "BUY")),
        "sell_trades": int(sum(1 for t in trade_log if t["action"] == "SELL")),
    }


def plot_nav_compare(nav_map, output_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    color_map = {
        "baseline": "#1f77b4",
        "attention_filter_q80": "#d62728",
        "attention_penalty_w010": "#ff7f0e",
        "attention_cooldown_q90_5d": "#2ca02c",
    }
    for name, nav_df in nav_map.items():
        ax.plot(
            nav_df["date"],
            nav_df["nav"] / base.INITIAL_CAPITAL,
            linewidth=1.5,
            color=color_map.get(name, None),
            label=name,
        )

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_title(
        "Old-main with intraday attention variants\n"
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


class AttentionPanel:
    def __init__(self, codes, dates, attention_up, attention_down, crowding_pct):
        self.codes = np.array(codes)
        self.dates = pd.DatetimeIndex(pd.to_datetime(dates))
        self.code_to_idx = {code: idx for idx, code in enumerate(self.codes.tolist())}

        self.attention_up = pd.DataFrame(attention_up, index=self.dates, columns=self.codes, dtype=np.float32)
        self.attention_down = pd.DataFrame(attention_down, index=self.dates, columns=self.codes, dtype=np.float32)
        self.crowding_pct = pd.DataFrame(crowding_pct, index=self.dates, columns=self.codes, dtype=np.float32)
        self.trade_crowding_pct = self.crowding_pct.shift(1)

    def get_crowding(self, date, code):
        date = pd.Timestamp(date)
        if code not in self.trade_crowding_pct.columns:
            return np.nan
        if date in self.trade_crowding_pct.index:
            return self.trade_crowding_pct.at[date, code]
        # For next-day order generation after the latest available intraday date,
        # use the latest realized crowding signal directly.
        if len(self.dates) > 0 and date > self.dates.max():
            return self.crowding_pct.at[self.dates.max(), code]
        return np.nan

    def get_cooldown_active(self, date, code, threshold, cooldown_days):
        date = pd.Timestamp(date)
        if date not in self.crowding_pct.index or code not in self.crowding_pct.columns:
            return False

        series = self.crowding_pct[code]
        loc = self.crowding_pct.index.get_indexer([date])[0]
        if loc <= 0:
            return False
        start = max(loc - cooldown_days, 0)
        prev_window = series.iloc[start:loc]
        if len(prev_window) == 0:
            return False
        return bool((prev_window >= threshold).any())


def build_attention_panel(cache_path, start_date, end_date, intraday_dir):
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        return AttentionPanel(
            codes=data["codes"].astype(str),
            dates=pd.to_datetime(data["dates"].astype(str), format="%Y%m%d"),
            attention_up=data["attention_up"],
            attention_down=data["attention_down"],
            crowding_pct=data["crowding_pct"],
        )

    intraday_files = sorted(
        path for path in os.listdir(intraday_dir) if path.endswith(".csv") and is_equity_code(os.path.splitext(path)[0])
    )
    intraday_files = [os.path.join(intraday_dir, name) for name in intraday_files]
    if not intraday_files:
        raise FileNotFoundError(f"No 15m files found under {intraday_dir}")

    master_ts, master_day_int, master_days = load_master_intraday_calendar(intraday_files[0])
    factor_days = pd.to_datetime(master_days.astype(str), format="%Y%m%d")
    day_mask = (factor_days >= pd.Timestamp(start_date)) & (factor_days <= pd.Timestamp(min(end_date, DEFAULT_FACTOR_END)))
    selected_days = factor_days[day_mask]
    selected_day_int = master_days[day_mask.to_numpy() if hasattr(day_mask, "to_numpy") else day_mask]
    if len(selected_days) == 0:
        raise RuntimeError("No intraday attention dates found inside requested window.")

    intraday = build_intraday_behavior_matrices(intraday_files, master_ts, master_day_int, master_days)
    day_sel = np.searchsorted(master_days, selected_day_int)
    attention_up = intraday["attention_up"][day_sel, :]
    attention_down = intraday["attention_down"][day_sel, :]

    up_df = pd.DataFrame(attention_up, index=selected_days, columns=intraday["codes"], dtype=np.float32)
    down_df = pd.DataFrame(attention_down, index=selected_days, columns=intraday["codes"], dtype=np.float32)
    crowding_pct = ((up_df.rank(axis=1, pct=True) + down_df.rank(axis=1, pct=True)) / 2.0).astype(np.float32)

    np.savez_compressed(
        cache_path,
        codes=np.array(intraday["codes"], dtype="U6"),
        dates=selected_days.strftime("%Y%m%d").to_numpy(dtype="U8"),
        attention_up=attention_up.astype(np.float32),
        attention_down=attention_down.astype(np.float32),
        crowding_pct=crowding_pct.to_numpy(dtype=np.float32),
    )

    return AttentionPanel(
        codes=np.array(intraday["codes"], dtype="U6"),
        dates=selected_days,
        attention_up=attention_up.astype(np.float32),
        attention_down=attention_down.astype(np.float32),
        crowding_pct=crowding_pct.to_numpy(dtype=np.float32),
    )


class AttentionTriggerEngine(base.EventTriggerEngine):
    def __init__(self, factor_engine, backup_multiple=3, backup_extra=20):
        super().__init__(factor_engine)
        self.backup_multiple = max(int(backup_multiple), 1)
        self.backup_extra = max(int(backup_extra), 0)

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        if n_needed <= 0:
            return []

        eligible_pool = self._get_eligible_pool(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(["delta_ivol", "ivol"], ascending=[True, True])
        backup_n = max(n_needed, n_needed * self.backup_multiple, n_needed + self.backup_extra)
        return candidates.head(backup_n)["code"].tolist()


class AttentionVariantBacktestEngine(base.EventDrivenBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        variant_name,
        backup_multiple=3,
        backup_extra=20,
        filter_threshold=0.80,
        penalty_weight=0.10,
        cooldown_threshold=0.90,
        cooldown_days=5,
    ):
        super().__init__(data_loader)
        self.attention_panel = attention_panel
        self.variant_name = variant_name
        self.filter_threshold = float(filter_threshold)
        self.penalty_weight = float(penalty_weight)
        self.cooldown_threshold = float(cooldown_threshold)
        self.cooldown_days = int(cooldown_days)
        self.trigger_engine = AttentionTriggerEngine(
            self.factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )

    def _select_buys(self, date, candidate_codes, n_slots):
        if not candidate_codes or n_slots <= 0:
            return [], {}, self._empty_stats(len(candidate_codes), n_slots)

        date = pd.Timestamp(date)
        last_idx = max(len(candidate_codes) - 1, 1)
        meta_by_code = {}

        if self.variant_name == "attention_filter_q80":
            kept = []
            filtered = []
            for idx, code in enumerate(candidate_codes):
                crowding = self.attention_panel.get_crowding(date, code)
                keep = pd.isna(crowding) or crowding < self.filter_threshold
                meta_by_code[code] = {
                    "base_rank": int(idx + 1),
                    "crowding_pct": float(crowding) if pd.notna(crowding) else np.nan,
                    "kept": bool(keep),
                }
                if keep:
                    kept.append(code)
                else:
                    filtered.append(code)

            selected = kept[:n_slots]
            if len(selected) < n_slots:
                selected.extend(filtered[: n_slots - len(selected)])

            stats = {
                "candidate_count": int(len(candidate_codes)),
                "slot_count": int(n_slots),
                "selected_count": int(len(selected)),
                "avg_selected_crowding": float(np.nanmean([meta_by_code[c]["crowding_pct"] for c in selected]))
                if selected
                else np.nan,
                "blocked_count": int(len(filtered)),
            }
            return selected, meta_by_code, stats

        if self.variant_name.startswith("attention_penalty"):
            scored = []
            for idx, code in enumerate(candidate_codes):
                base_score = 1.0 - idx / last_idx
                crowding = self.attention_panel.get_crowding(date, code)
                penalty = 0.0 if pd.isna(crowding) else self.penalty_weight * float(crowding)
                combined_score = base_score - penalty
                meta_by_code[code] = {
                    "base_rank": int(idx + 1),
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
                "blocked_count": 0,
            }
            return selected, meta_by_code, stats

        if self.variant_name == "attention_cooldown_q90_5d":
            allowed = []
            blocked = []
            for idx, code in enumerate(candidate_codes):
                cooldown = self.attention_panel.get_cooldown_active(
                    date,
                    code,
                    threshold=self.cooldown_threshold,
                    cooldown_days=self.cooldown_days,
                )
                crowding = self.attention_panel.get_crowding(date, code)
                meta_by_code[code] = {
                    "base_rank": int(idx + 1),
                    "crowding_pct": float(crowding) if pd.notna(crowding) else np.nan,
                    "cooldown": bool(cooldown),
                }
                if cooldown:
                    blocked.append(code)
                else:
                    allowed.append(code)

            selected = allowed[:n_slots]
            stats = {
                "candidate_count": int(len(candidate_codes)),
                "slot_count": int(n_slots),
                "selected_count": int(len(selected)),
                "avg_selected_crowding": float(np.nanmean([meta_by_code[c]["crowding_pct"] for c in selected]))
                if selected
                else np.nan,
                "blocked_count": int(len(blocked)),
            }
            return selected, meta_by_code, stats

        raise ValueError(f"Unknown attention variant: {self.variant_name}")

    @staticmethod
    def _empty_stats(candidate_count, n_slots):
        return {
            "candidate_count": int(candidate_count),
            "slot_count": int(n_slots),
            "selected_count": 0,
            "avg_selected_crowding": np.nan,
            "blocked_count": 0,
        }

    def run(self):
        print("\n" + "=" * 60)
        print(f"  Start backtest: {self.variant_name}")
        print("=" * 60)

        df = self.dl.df_hfq
        all_dates_raw = sorted(df["date"].unique())
        all_dates = [
            pd.Timestamp(d)
            for d in all_dates_raw
            if pd.Timestamp(base.START_DATE) <= pd.Timestamp(d) <= pd.Timestamp(base.END_DATE)
        ]
        if not all_dates:
            print("No data in the backtest window.")
            return

        print(f"Backtest window: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")
        print(f"Trading days: {len(all_dates)}")
        print("Building price lookup tables...")

        df_raw = self.dl.df_raw
        close_pivot = df_raw.pivot_table(index="date", columns="code", values="close", aggfunc="first")
        close_pivot.ffill(inplace=True)
        open_pivot = df_raw.pivot_table(index="date", columns="code", values="open", aggfunc="first")
        last_data_date = df_raw.dropna(subset=["close"]).groupby("code")["date"].max().to_dict()

        date_series = pd.Series(all_dates)
        refresh_dates = set(date_series.groupby(date_series.dt.to_period(base.FACTOR_REFRESH_FREQ)).max().tolist())

        holdings = {}
        cash = base.INITIAL_CAPITAL
        factor_cache = None
        nav = base.INITIAL_CAPITAL

        first_factor_date = None
        for d in all_dates:
            if d in refresh_dates:
                first_factor_date = d
                break
        if first_factor_date is None:
            first_factor_date = all_dates[0]

        initialized = False
        pending_sells = set()
        pending_buys = []
        pending_buy_slots = 0
        pending_sell_reasons = {}

        print("\n--- Running daily loop ---")
        for t_idx in base.tqdm(range(len(all_dates)), desc=self.variant_name):
            today = all_dates[t_idx]
            prev_date = all_dates[t_idx - 1] if t_idx > 0 else None
            today_stats = self._empty_stats(len(pending_buys), pending_buy_slots)

            delist_codes = set()
            for code in list(holdings.keys()):
                code_last = last_data_date.get(code)
                if code_last is not None and today > code_last + pd.Timedelta(days=base.DELIST_NO_DATA_DAYS):
                    delist_codes.add(code)
            if delist_codes:
                for code in delist_codes:
                    liq_price = np.nan
                    if code in close_pivot.columns:
                        valid_prices = close_pivot.loc[:today, code].dropna()
                        if len(valid_prices) > 0:
                            liq_price = valid_prices.iloc[-1]
                    if pd.isna(liq_price) or liq_price <= 0:
                        liq_price = 0.0
                    shares = holdings[code]["shares"]
                    sell_value = shares * liq_price
                    cost = sell_value * (base.COMMISSION_RATE + base.STAMP_DUTY_RATE)
                    cash += sell_value - cost
                    self.trade_log.append(
                        {
                            "date": today.strftime("%Y-%m-%d"),
                            "code": code,
                            "action": "SELL",
                            "price": round(float(liq_price), 4),
                            "shares": shares,
                            "value": round(float(sell_value), 2),
                            "reason": "delist_liquidation",
                        }
                    )
                    del holdings[code]
                    pending_sells.discard(code)
                    pending_sell_reasons.pop(code, None)

            if pending_sells or pending_buys:
                for code in list(pending_sells):
                    if code not in holdings:
                        pending_sells.discard(code)
                        continue

                    sell_price = np.nan
                    prev_close = np.nan
                    if code in open_pivot.columns and today in open_pivot.index:
                        sell_price = open_pivot.loc[today, code]
                    if code in close_pivot.columns and prev_date is not None and prev_date in close_pivot.index:
                        prev_close = close_pivot.loc[prev_date, code]

                    if pd.isna(sell_price) or sell_price <= 0:
                        continue

                    limit_pct = base._get_limit_pct(code)
                    if pd.notna(prev_close) and prev_close > 0 and sell_price <= prev_close * (1.0 - limit_pct * 0.95):
                        continue

                    sell_price = sell_price * (1.0 - base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
                    if pd.isna(sell_price) or sell_price <= 0:
                        continue

                    shares = holdings[code]["shares"]
                    sell_value = shares * sell_price
                    cost = sell_value * (base.COMMISSION_RATE + base.STAMP_DUTY_RATE)
                    cash += sell_value - cost

                    reason = pending_sell_reasons.get(code, "unknown")
                    self.trade_log.append(
                        {
                            "date": today.strftime("%Y-%m-%d"),
                            "code": code,
                            "action": "SELL",
                            "price": round(float(sell_price), 4),
                            "shares": shares,
                            "value": round(float(sell_value), 2),
                            "reason": reason,
                        }
                    )

                    del holdings[code]
                    pending_sells.discard(code)
                    pending_sell_reasons.pop(code, None)

                if pending_buys and pending_buy_slots > 0 and cash > 0:
                    selected_buys, buy_meta, today_stats = self._select_buys(today, pending_buys, pending_buy_slots)
                    if selected_buys:
                        standard_position_value = nav / base.TARGET_POSITIONS
                        allocatable_per_stock = min(cash / len(selected_buys), standard_position_value)

                        executed_buys = []
                        for code in selected_buys:
                            buy_price = np.nan
                            prev_close_buy = np.nan
                            if code in open_pivot.columns and today in open_pivot.index:
                                buy_price = open_pivot.loc[today, code]
                            if code in close_pivot.columns and prev_date is not None and prev_date in close_pivot.index:
                                prev_close_buy = close_pivot.loc[prev_date, code]

                            if pd.isna(buy_price) or buy_price <= 0:
                                continue
                            limit_pct = base._get_limit_pct(code)
                            if pd.notna(prev_close_buy) and prev_close_buy > 0:
                                if buy_price >= prev_close_buy * (1.0 + limit_pct * 0.95):
                                    continue
                                if buy_price <= prev_close_buy * (1.0 - limit_pct * 0.95):
                                    continue

                            buy_price = buy_price * (1.0 + base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
                            if pd.isna(buy_price) or buy_price <= 0:
                                continue

                            executed_buys.append((code, float(buy_price)))

                        for code, buy_price in executed_buys:
                            raw_shares = allocatable_per_stock / buy_price
                            trade_shares = int(raw_shares // 100) * 100
                            if trade_shares < 100:
                                continue

                            trade_value = trade_shares * buy_price
                            cost = trade_value * base.COMMISSION_RATE
                            actual_invested = trade_value + cost
                            if cash >= actual_invested:
                                holdings[code] = {"shares": trade_shares}
                                cash -= actual_invested
                                meta = buy_meta.get(code, {})
                                crowd = meta.get("crowding_pct", np.nan)
                                self.trade_log.append(
                                    {
                                        "date": today.strftime("%Y-%m-%d"),
                                        "code": code,
                                        "action": "BUY",
                                        "price": round(buy_price, 4),
                                        "shares": trade_shares,
                                        "value": round(float(trade_value), 2),
                                        "reason": f"{self.variant_name}(crowding={crowd:.3f})",
                                    }
                                )

                pending_buys = []
                pending_buy_slots = 0

            total_holdings_value = 0.0
            for code, holding in holdings.items():
                c_today = np.nan
                if code in close_pivot.columns and today in close_pivot.index:
                    c_today = close_pivot.loc[today, code]
                if pd.notna(c_today) and c_today > 0:
                    total_holdings_value += holding["shares"] * c_today
                else:
                    prev_prices = close_pivot.loc[:today, code].dropna()
                    if len(prev_prices) > 0:
                        total_holdings_value += holding["shares"] * prev_prices.iloc[-1]

            nav = total_holdings_value + cash
            self.daily_nav.append((today, nav))

            need_factor_refresh = (today in refresh_dates) or (not initialized and today >= first_factor_date)
            if need_factor_refresh:
                valid_codes = self.filter.filter_universe(today)
                if len(valid_codes) >= 50:
                    factor_cache = self.factor_engine.compute_factors(valid_codes, today)
                    if len(factor_cache) > 0 and not initialized:
                        pending_buy_slots = base.TARGET_POSITIONS
                        buy_candidates = self.trigger_engine.get_buy_candidates(set(), pending_buy_slots, factor_cache)
                        if buy_candidates:
                            pending_buys = buy_candidates
                            initialized = True

            if initialized and factor_cache is not None and len(holdings) > 0:
                sell_codes, sell_reasons = self.trigger_engine.check_sell_triggers(list(holdings.keys()), today, factor_cache)
                if sell_codes:
                    pending_sells.update(sell_codes)
                    pending_sell_reasons.update(sell_reasons)

                current_count = len(holdings) - len(pending_sells)
                n_needed = base.TARGET_POSITIONS - current_count
                if n_needed > 0:
                    current_holding_codes = set(holdings.keys()) - pending_sells
                    pending_buy_slots = n_needed
                    buy_candidates = self.trigger_engine.get_buy_candidates(current_holding_codes, n_needed, factor_cache)
                    if buy_candidates:
                        pending_buys = buy_candidates
                    else:
                        pending_buy_slots = 0

            self.daily_status.append(
                {
                    "date": today.strftime("%Y-%m-%d"),
                    "nav": round(float(nav), 2),
                    "n_holdings": len(holdings),
                    "n_sold": len(pending_sells),
                    "n_buy_slots": int(pending_buy_slots),
                    "n_buy_backups": int(len(pending_buys)),
                    "attention_candidates": int(today_stats["candidate_count"]),
                    "attention_selected": int(today_stats["selected_count"]),
                    "attention_blocked": int(today_stats["blocked_count"]),
                    "attention_avg_selected_crowding": today_stats["avg_selected_crowding"],
                }
            )

            if initialized and len(holdings) == 0 and len(pending_buys) == 0:
                if factor_cache is not None:
                    pending_buy_slots = base.TARGET_POSITIONS
                    buy_candidates = self.trigger_engine.get_buy_candidates(set(), pending_buy_slots, factor_cache)
                    if buy_candidates:
                        pending_buys = buy_candidates
                    else:
                        pending_buy_slots = 0

        print("\nBacktest completed.")
        print(f"  Total trades: {len(self.trade_log)}")
        n_sells = sum(1 for t in self.trade_log if t["action"] == "SELL")
        n_buys = sum(1 for t in self.trade_log if t["action"] == "BUY")
        print(f"  Sells: {n_sells}, Buys: {n_buys}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test intraday attention variants on old_main")
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
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Output prefix. Default: {DEFAULT_OUTPUT_PREFIX}",
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
    print("  Old-main attention variant test")
    print(f"  Window: {base.START_DATE} ~ {base.END_DATE}")
    print(f"  Attention cache: {args.attention_cache}")
    print("  Variants: filter_q80, penalty_w010, cooldown_q90_5d")
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

    variant_specs = [
        ("attention_filter_q80", {"filter_threshold": 0.80}),
        ("attention_penalty_w010", {"penalty_weight": 0.10}),
        ("attention_cooldown_q90_5d", {"cooldown_threshold": 0.90, "cooldown_days": 5}),
    ]

    results = [{"strategy": "baseline", **baseline_summary}]
    nav_map = {"baseline": pd.DataFrame(baseline_engine.daily_nav, columns=["date", "nav"])}

    for variant_name, kwargs in variant_specs:
        print(f"\n[{variant_name}] running variant backtest...")
        engine = AttentionVariantBacktestEngine(
            loader,
            attention_panel,
            variant_name=variant_name,
            backup_multiple=args.backup_multiple,
            backup_extra=args.backup_extra,
            **kwargs,
        )
        engine.run()
        summary = compute_summary(engine.daily_nav, engine.trade_log)
        summary["baseline_total_return"] = baseline_summary["total_return"]
        summary["baseline_annual_return"] = baseline_summary["annual_return"]
        summary["baseline_max_drawdown"] = baseline_summary["max_drawdown"]
        summary["baseline_sharpe"] = baseline_summary["sharpe"]
        summary["total_return_diff"] = summary["total_return"] - baseline_summary["total_return"]
        summary["annual_return_diff"] = summary["annual_return"] - baseline_summary["annual_return"]
        summary["max_drawdown_diff"] = summary["max_drawdown"] - baseline_summary["max_drawdown"]
        summary["sharpe_diff"] = summary["sharpe"] - baseline_summary["sharpe"]
        results.append({"strategy": variant_name, **summary})
        nav_map[variant_name] = pd.DataFrame(engine.daily_nav, columns=["date", "nav"])

    result_df = pd.DataFrame(results)

    summary_path = os.path.join(BASE_DIR, f"{args.output_prefix}_summary.csv")
    plot_path = os.path.join(BASE_DIR, f"{args.output_prefix}_nav_compare.png")
    meta_path = os.path.join(BASE_DIR, f"{args.output_prefix}_meta.json")

    result_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nSummary exported: {summary_path}")

    plot_nav_compare(nav_map, plot_path)
    print(f"Compare plot exported: {plot_path}")

    payload = {
        "params": {
            "start": base.START_DATE,
            "end": base.END_DATE,
            "positions": base.TARGET_POSITIONS,
            "slippage_bps": base.SLIPPAGE_BPS_PER_SIDE,
            "attention_cache": os.path.abspath(args.attention_cache),
            "backup_multiple": args.backup_multiple,
            "backup_extra": args.backup_extra,
        },
        "attention_dates": {
            "start": attention_panel.dates.min().strftime("%Y-%m-%d"),
            "end": attention_panel.dates.max().strftime("%Y-%m-%d"),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Meta exported: {meta_path}")


if __name__ == "__main__":
    main()
