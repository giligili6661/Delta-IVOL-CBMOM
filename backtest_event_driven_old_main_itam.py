import argparse
import os
import glob
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from event_driven_tiam_factor_validation import INTRADAY_15M_HFQ_DIR, build_tiam_panel

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HFQ_DIR = os.path.join(BASE_DIR, "data_stock_daily_hfq")
RAW_DIR = os.path.join(BASE_DIR, "data_stock_daily_unadj")

START_DATE = "2019-01-01"
END_DATE = "2025-12-31"
INITIAL_CAPITAL = 1_000_000
COMMISSION_RATE = 0.0003
STAMP_DUTY_RATE = 0.0005
RISK_FREE_RATE = 0.0
SLIPPAGE_BPS_PER_SIDE = 5

IVOL_WINDOW = 252
IVOL_MIN_OBS = 200
SHORT_IVOL_WINDOW = 5
IVOL_SPIKE_MULT = 2.0
MIN_LIST_DAYS = 252
MIN_PRICE = 2.0
LONG_PCT = 0.20
TARGET_POSITIONS = 20
MA_PERIOD = 60
FACTOR_REFRESH_FREQ = "M"
BUY_SCORE_DELTA_WEIGHT = 0.50
BUY_SCORE_CBMOM_WEIGHT = 0.50
DELIST_NO_DATA_DAYS = 20
ITAM_TOP_PCT = 0.10
ITAM_SELL_TOP_PCT = 0.50
DEFAULT_ITAM_PANEL_PATH = os.path.join(BASE_DIR, "event_driven_tiam_factor_validation_tiam_panel.csv")


def load_itam_panel(panel_path=DEFAULT_ITAM_PANEL_PATH, intraday_dir=INTRADAY_15M_HFQ_DIR, start=None, end=None):
    if os.path.exists(panel_path):
        print(f"[ITAM] Loading cached panel: {panel_path}")
        panel = pd.read_csv(panel_path, dtype={"code": str})
    else:
        print(f"[ITAM] Building panel from 15m data: {intraday_dir}")
        panel = build_tiam_panel(intraday_dir, start=start, end=end)
        panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
        print(f"[ITAM] Saved panel: {panel_path}")

    panel["code"] = panel["code"].astype(str).str.zfill(6)
    rename_map = {}
    if "tiam" in panel.columns:
        rename_map["tiam"] = "itam"
    if "tiam_n_days" in panel.columns:
        rename_map["tiam_n_days"] = "itam_n_days"
    panel = panel.rename(columns=rename_map)
    if "itam" not in panel.columns:
        raise RuntimeError("ITAM panel is missing the required 'itam' column.")
    if "itam_n_days" not in panel.columns:
        panel["itam_n_days"] = np.nan
    return panel[["month", "code", "itam", "itam_n_days"]].copy()


def _get_limit_pct(code: str) -> float:
    """Return the daily price limit percentage based on board type.

    - ChiNext (300xxx, 301xxx): +/-20%
    - STAR Market (688xxx, 689xxx): +/-20%
    - Main board & others: +/-10%
    """
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    return 0.10



class DataLoader:
    def __init__(self, hfq_dir=HFQ_DIR, raw_dir=RAW_DIR):
        self.hfq_dir = hfq_dir
        self.raw_dir = raw_dir
        self.df_hfq = None
        self.df_raw = None
        self.df_ff3 = None
        self.df_hs300 = None

    def load_stock_data(self):
        print("[1/4] Loading HFQ daily data...")
        self.df_hfq = self._load_dir(self.hfq_dir)

        print("[2/4] Loading raw daily data...")
        self.df_raw = self._load_dir(self.raw_dir)

        print(f"  HFQ: {self.df_hfq['code'].nunique()} stocks, {len(self.df_hfq)} rows")
        print(f"  RAW: {self.df_raw['code'].nunique()} stocks, {len(self.df_raw)} rows")

    def _load_dir(self, directory):
        files = glob.glob(os.path.join(directory, "*.csv"))
        dfs = []
        for path in files:
            code = os.path.splitext(os.path.basename(path))[0]
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
                if df.shape[1] < 4:
                    continue
                df = df.iloc[:, :4].copy()
                df.columns = ["date", "close", "open", "turnover"]
                df["code"] = code
                df["date"] = pd.to_datetime(df["date"])
                for col in ["close", "open", "turnover"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                dfs.append(df)
            except Exception:
                continue

        if not dfs:
            raise RuntimeError(f"No csv files loaded from {directory}")

        result = pd.concat(dfs, ignore_index=True)
        result.sort_values(["code", "date"], inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def load_ff3_factors(self):
        ff3_path = os.path.join(BASE_DIR, "ff3_factors_cn.csv")
        if not os.path.exists(ff3_path):
            raise FileNotFoundError(f"FF3 file not found: {ff3_path}")
        print("[3/4] Loading FF3 factors...")
        self.df_ff3 = pd.read_csv(ff3_path, parse_dates=["date"])
        print(f"  FF3 dates: {len(self.df_ff3)}")

    def load_hs300(self):
        hs300_path = os.path.join(BASE_DIR, "math", "hs300_for_hmm.csv")
        if not os.path.exists(hs300_path):
            raise FileNotFoundError(f"HS300 file not found: {hs300_path}")
        print("[4/4] Loading HS300 data...")
        self.df_hs300 = pd.read_csv(hs300_path, parse_dates=["date"])
        self.df_hs300.sort_values("date", inplace=True)
        self.df_hs300.reset_index(drop=True, inplace=True)
        print(f"  HS300 dates: {len(self.df_hs300)}")

    def compute_returns(self):
        print("Computing daily returns...")
        self.df_hfq.sort_values(["code", "date"], inplace=True)
        self.df_hfq["ret"] = self.df_hfq.groupby("code")["close"].pct_change()


class UniverseFilter:
    def __init__(self, df_hfq, df_raw):
        self.df_hfq = df_hfq
        self.df_raw = df_raw
        self.ipo_dates = df_hfq.groupby("code")["date"].min().to_dict()

    def filter_universe(self, date):
        valid_codes = set(self.df_hfq["code"].unique())

        st_codes = self._detect_st(date)
        valid_codes -= st_codes

        ipo_cutoff = date - pd.Timedelta(days=365)
        new_codes = {c for c, d in self.ipo_dates.items() if d > ipo_cutoff}
        valid_codes -= new_codes

        lookback_start = date - pd.Timedelta(days=40)
        mask = (self.df_hfq["date"] >= lookback_start) & (self.df_hfq["date"] <= date)
        recent_data = self.df_hfq[mask]
        trade_counts = recent_data.groupby("code")["date"].count()
        suspended = set(trade_counts[trade_counts < 15].index)
        missing = valid_codes - set(trade_counts.index)
        valid_codes -= suspended
        valid_codes -= missing

        raw_mask = (self.df_raw["date"] <= date) & (self.df_raw["date"] >= date - pd.Timedelta(days=7))
        raw_eod = self.df_raw[raw_mask].sort_values("date").groupby("code").last()
        penny = set(raw_eod[raw_eod["close"] < MIN_PRICE].index)
        valid_codes -= penny

        return list(valid_codes)

    def _detect_st(self, date, lookback=20):
        mask = (self.df_hfq["date"] <= date) & (self.df_hfq["date"] >= date - pd.Timedelta(days=lookback * 2))
        subset = self.df_hfq[mask].copy()
        subset.sort_values(["code", "date"], inplace=True)
        subset["ret"] = subset.groupby("code")["close"].pct_change()
        subset = subset.dropna(subset=["ret"])

        recent = subset.groupby("code").tail(lookback)
        max_ret = recent.groupby("code")["ret"].max()
        min_ret = recent.groupby("code")["ret"].min()

        st_codes = set()
        for code in max_ret.index:
            mx = max_ret.get(code, 0.10)
            mn = min_ret.get(code, -0.10)
            if 0.04 <= mx <= 0.055 and -0.055 <= mn <= -0.04:
                st_codes.add(code)
        return st_codes


class FactorEngine:
    def __init__(self, df_hfq, df_ff3, itam_lookup=None):
        self.df_hfq = df_hfq
        self.df_ff3 = df_ff3
        self.itam_lookup = itam_lookup or {}

    def _attach_itam(self, factor_df, date):
        if factor_df is None or len(factor_df) == 0:
            return pd.DataFrame()

        month_key = str(pd.Timestamp(date).to_period(FACTOR_REFRESH_FREQ))
        itam_month = self.itam_lookup.get(month_key)
        if itam_month is None or len(itam_month) == 0:
            return pd.DataFrame()

        out = factor_df.merge(itam_month[["code", "itam", "itam_n_days"]], on="code", how="inner")
        return out

    def compute_factors(self, valid_codes, date):
        window_start = date - pd.Timedelta(days=int(IVOL_WINDOW * 1.6))
        mask = (
            (self.df_hfq["date"] <= date)
            & (self.df_hfq["date"] >= window_start)
            & (self.df_hfq["code"].isin(valid_codes))
        )
        stock_data = self.df_hfq[mask]

        ff3_mask = (self.df_ff3["date"] <= date) & (self.df_ff3["date"] >= window_start)
        ff3 = self.df_ff3[ff3_mask].set_index("date")

        ret_pivot = stock_data.pivot_table(index="date", columns="code", values="ret", aggfunc="first")
        common_dates = ret_pivot.index.intersection(ff3.index)
        if len(common_dates) < IVOL_MIN_OBS:
            return pd.DataFrame()

        ret_pivot = ret_pivot.loc[common_dates].sort_index()
        ff3_aligned = ff3.loc[common_dates].sort_index()

        if len(ret_pivot) > IVOL_WINDOW:
            ret_pivot = ret_pivot.tail(IVOL_WINDOW)
            ff3_aligned = ff3_aligned.tail(IVOL_WINDOW)

        t_len = len(ret_pivot)
        valid_count = ret_pivot.notna().sum(axis=0)
        eligible_codes = valid_count[valid_count >= IVOL_MIN_OBS].index.tolist()
        if not eligible_codes:
            return pd.DataFrame()

        ret_pivot = ret_pivot[eligible_codes]
        x = np.column_stack(
            [
                np.ones(t_len),
                ff3_aligned["MKT"].values,
                ff3_aligned["SMB"].values,
                ff3_aligned["HML"].values,
            ]
        )

        y_raw = ret_pivot.values
        nan_mask = np.isnan(y_raw)
        has_nans = nan_mask.any()

        sum_ret = np.nansum(y_raw, axis=0)
        sum_abs_ret = np.nansum(np.abs(y_raw), axis=0)
        cbmom_arr_all = np.where(sum_abs_ret > 0, sum_ret / sum_abs_ret, 0.0)

        if not has_nans:
            try:
                beta, _, _, _ = np.linalg.lstsq(x, y_raw, rcond=None)
            except np.linalg.LinAlgError:
                return pd.DataFrame()

            resid = y_raw - x @ beta
            ivol_arr = np.std(resid, axis=0, ddof=1)
            half = t_len // 2
            ivol_past_arr = np.std(resid[:half], axis=0, ddof=1)
            ivol_recent_arr = np.std(resid[half:], axis=0, ddof=1)
            delta_ivol_arr = ivol_recent_arr - ivol_past_arr

            factor_df = pd.DataFrame(
                {
                    "code": eligible_codes,
                    "ivol": ivol_arr,
                    "delta_ivol": delta_ivol_arr,
                    "cbmom": cbmom_arr_all,
                }
            )
            return self._attach_itam(factor_df, date)

        pattern_map = {}
        for j in range(y_raw.shape[1]):
            key = nan_mask[:, j].tobytes()
            pattern_map.setdefault(key, []).append(j)

        ivol_arr = np.full(len(eligible_codes), np.nan)
        delta_ivol_arr = np.full(len(eligible_codes), np.nan)

        for key_bytes, col_indices in pattern_map.items():
            row_mask = ~nan_mask[:, col_indices[0]]
            n_valid = row_mask.sum()
            if n_valid < IVOL_MIN_OBS:
                continue

            x_sub = x[row_mask]
            y_sub = y_raw[np.ix_(row_mask, col_indices)]
            try:
                beta_sub, _, _, _ = np.linalg.lstsq(x_sub, y_sub, rcond=None)
            except np.linalg.LinAlgError:
                continue

            resid_sub = y_sub - x_sub @ beta_sub
            ivol_arr[col_indices] = np.std(resid_sub, axis=0, ddof=1)
            half = n_valid // 2
            ivol_past = np.std(resid_sub[:half], axis=0, ddof=1)
            ivol_recent = np.std(resid_sub[half:], axis=0, ddof=1)
            delta_ivol_arr[col_indices] = ivol_recent - ivol_past

        computed_mask = ~np.isnan(ivol_arr)
        factor_df = pd.DataFrame(
            {
                "code": np.array(eligible_codes)[computed_mask],
                "ivol": ivol_arr[computed_mask],
                "delta_ivol": delta_ivol_arr[computed_mask],
                "cbmom": cbmom_arr_all[computed_mask],
            }
        )
        return self._attach_itam(factor_df, date)

    def compute_short_ivol(self, codes, date):
        if not codes:
            return {}

        window_start = date - pd.Timedelta(days=SHORT_IVOL_WINDOW * 3)
        mask = (
            (self.df_hfq["date"] <= date)
            & (self.df_hfq["date"] >= window_start)
            & (self.df_hfq["code"].isin(codes))
        )
        stock_data = self.df_hfq[mask]

        ff3_mask = (self.df_ff3["date"] <= date) & (self.df_ff3["date"] >= window_start)
        ff3 = self.df_ff3[ff3_mask].set_index("date")

        ret_pivot = stock_data.pivot_table(index="date", columns="code", values="ret", aggfunc="first")
        common_dates = ret_pivot.index.intersection(ff3.index)
        if len(common_dates) < SHORT_IVOL_WINDOW:
            return {}

        ret_pivot = ret_pivot.loc[common_dates].sort_index().tail(SHORT_IVOL_WINDOW)
        ff3_aligned = ff3.loc[common_dates].sort_index().tail(SHORT_IVOL_WINDOW)
        t_len = len(ret_pivot)
        if t_len < SHORT_IVOL_WINDOW:
            return {}

        x = np.column_stack(
            [
                np.ones(t_len),
                ff3_aligned["MKT"].values,
                ff3_aligned["SMB"].values,
                ff3_aligned["HML"].values,
            ]
        )

        result = {}
        for code in codes:
            if code not in ret_pivot.columns:
                continue
            y = ret_pivot[code].values
            if np.isnan(y).any():
                continue
            try:
                beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = y - x @ beta
            result[code] = np.std(resid, ddof=1)
        return result


class CircuitBreaker:
    def __init__(self, df_hs300):
        self.df_hs300 = df_hs300.copy()
        self.df_hs300.sort_values("date", inplace=True)
        self.df_hs300.set_index("date", inplace=True)
        self.df_hs300["ma60"] = self.df_hs300["close"].rolling(window=MA_PERIOD, min_periods=MA_PERIOD).mean()

    def is_frozen(self, date):
        available = self.df_hs300.loc[:date]
        if len(available) == 0:
            return False, np.nan, np.nan

        latest = available.iloc[-1]
        close_price = latest["close"]
        ma60 = latest["ma60"]
        if pd.isna(ma60):
            return False, close_price, np.nan
        return close_price < ma60, close_price, ma60


class EventTriggerEngine:
    def __init__(self, factor_engine):
        self.factor_engine = factor_engine

    def _get_high_itam_codes(self, factor_cache, top_pct):
        if factor_cache is None or len(factor_cache) == 0:
            return set()

        cutoff = max(int(len(factor_cache) * top_pct), 1)
        return set(factor_cache.sort_values("itam", ascending=False).head(cutoff)["code"].tolist())

    def _get_eligible_pool(self, factor_cache):
        if factor_cache is None or len(factor_cache) == 0:
            return pd.DataFrame()

        cutoff_low = max(int(len(factor_cache) * LONG_PCT), 1)
        high_itam_codes = self._get_high_itam_codes(factor_cache, ITAM_TOP_PCT)
        low_ivol = factor_cache.sort_values("ivol").head(cutoff_low)
        low_ivol = low_ivol[low_ivol["code"].isin(high_itam_codes)]
        converging = low_ivol[low_ivol["delta_ivol"] < 0]
        positive = converging[converging["cbmom"] > 0]
        return positive

    def check_sell_triggers(self, holding_codes, date, factor_cache):
        sell_codes = set()
        sell_reasons = {}

        if factor_cache is None or len(factor_cache) == 0:
            return sell_codes, sell_reasons

        keep_itam_codes = self._get_high_itam_codes(factor_cache, ITAM_SELL_TOP_PCT)
        for code in holding_codes:
            code_data = factor_cache[factor_cache["code"] == code]
            if len(code_data) == 0:
                sell_codes.add(code)
                sell_reasons[code] = "data_missing"
                continue

            if code not in keep_itam_codes:
                itam_score = code_data["itam"].values[0]
                sell_codes.add(code)
                sell_reasons[code] = (
                    f"ITAM_faded(score={itam_score:.4f}, out_of_top_{int(ITAM_SELL_TOP_PCT * 100)}pct)"
                )
                continue

            cbmom_score = code_data["cbmom"].values[0]
            if cbmom_score < 0:
                sell_codes.add(code)
                sell_reasons[code] = f"CBMOM_negative(score={cbmom_score:.4f})"
                continue

        remaining = [c for c in holding_codes if c not in sell_codes]
        if remaining:
            short_ivol = self.factor_engine.compute_short_ivol(remaining, date)
            for code in remaining:
                if code not in short_ivol:
                    continue
                code_data = factor_cache[factor_cache["code"] == code]
                if len(code_data) == 0:
                    continue
                long_avg_ivol = code_data["ivol"].values[0]
                if long_avg_ivol > 0 and short_ivol[code] > IVOL_SPIKE_MULT * long_avg_ivol:
                    sell_codes.add(code)
                    ratio = short_ivol[code] / long_avg_ivol
                    sell_reasons[code] = (
                        f"IVOL_spike(5d={short_ivol[code]:.4f}, long={long_avg_ivol:.4f}, ratio={ratio:.1f}x)"
                    )

        return sell_codes, sell_reasons

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        """
        Build buy candidates from the current factor cache:
          1. IVOL in the lowest 20% of the market
          2. ITAM in the highest 10% of the market
          3. Delta IVOL < 0
          4. CBMOM > 0
          5. rank by Delta IVOL ascending, then IVOL ascending
        Return: list of codes
        """
        if n_needed <= 0:
            return []

        eligible_pool = self._get_eligible_pool(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(["delta_ivol", "ivol", "itam"], ascending=[True, True, False])
        return candidates.head(n_needed)["code"].tolist()


class EventDrivenBacktestEngine:
    def __init__(self, data_loader, itam_lookup=None):
        self.dl = data_loader
        self.filter = UniverseFilter(data_loader.df_hfq, data_loader.df_raw)
        self.factor_engine = FactorEngine(data_loader.df_hfq, data_loader.df_ff3, itam_lookup=itam_lookup)
        self.trigger_engine = EventTriggerEngine(self.factor_engine)
        self.circuit_breaker = CircuitBreaker(data_loader.df_hs300)
        self.daily_nav = []
        self.trade_log = []
        self.daily_status = []

    def run(self):
        print("\n" + "=" * 60)
        print("  Start backtest: event-driven strategy")
        print("=" * 60)

        df = self.dl.df_hfq
        all_dates_raw = sorted(df["date"].unique())
        all_dates = [
            pd.Timestamp(d)
            for d in all_dates_raw
            if pd.Timestamp(START_DATE) <= pd.Timestamp(d) <= pd.Timestamp(END_DATE)
        ]
        if not all_dates:
            print("No data in the backtest window.")
            return

        print(f"Backtest window: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")
        print(f"Trading days: {len(all_dates)}")

        print("Building price lookup tables...")
        # --- raw (unadjusted) prices for trade execution & NAV valuation ---
        df_raw = self.dl.df_raw
        close_pivot = df_raw.pivot_table(index="date", columns="code", values="close", aggfunc="first")
        close_pivot.ffill(inplace=True)
        open_pivot = df_raw.pivot_table(index="date", columns="code", values="open", aggfunc="first")

        # Build a lookup: last date that each code has valid raw data
        last_data_date = df_raw.dropna(subset=["close"]).groupby("code")["date"].max().to_dict()


        date_series = pd.Series(all_dates)
        month_ends = set(date_series.groupby(date_series.dt.to_period(FACTOR_REFRESH_FREQ)).max().tolist())

        holdings = {}
        cash = INITIAL_CAPITAL
        factor_cache = None
        nav = INITIAL_CAPITAL

        first_factor_date = None
        for d in all_dates:
            if d in month_ends:
                first_factor_date = d
                break
        if first_factor_date is None:
            first_factor_date = all_dates[0]

        initialized = False
        pending_sells = set()
        pending_buys = []
        pending_sell_reasons = {}

        print("\n--- Running daily loop ---")
        for t_idx in tqdm(range(len(all_dates)), desc="Backtest progress"):
            today = all_dates[t_idx]
            prev_date = all_dates[t_idx - 1] if t_idx > 0 else None

            # ------ Delisting forced liquidation check ------
            delist_codes = set()
            for code in list(holdings.keys()):
                code_last = last_data_date.get(code)
                if code_last is not None and today > code_last + pd.Timedelta(days=DELIST_NO_DATA_DAYS):
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
                    cost = sell_value * (COMMISSION_RATE + STAMP_DUTY_RATE)
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

                    # Board-specific limit-down check
                    limit_pct = _get_limit_pct(code)
                    if pd.notna(prev_close) and prev_close > 0 and sell_price <= prev_close * (1.0 - limit_pct * 0.95):
                        continue

                    sell_price = sell_price * (1.0 - SLIPPAGE_BPS_PER_SIDE / 10000.0)
                    if pd.isna(sell_price) or sell_price <= 0:
                        continue

                    shares = holdings[code]["shares"]
                    sell_value = shares * sell_price
                    cost = sell_value * (COMMISSION_RATE + STAMP_DUTY_RATE)
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


                if pending_buys and cash > 0:
                    standard_position_value = nav / TARGET_POSITIONS
                    allocatable_per_stock = min(cash / len(pending_buys), standard_position_value)

                    executed_buys = []
                    for code in pending_buys:
                        buy_price = np.nan
                        prev_close_buy = np.nan
                        if code in open_pivot.columns and today in open_pivot.index:
                            buy_price = open_pivot.loc[today, code]
                        if code in close_pivot.columns and prev_date is not None and prev_date in close_pivot.index:
                            prev_close_buy = close_pivot.loc[prev_date, code]

                        if pd.isna(buy_price) or buy_price <= 0:
                            continue
                        # Board-specific limit-up / limit-down check
                        limit_pct = _get_limit_pct(code)
                        if pd.notna(prev_close_buy) and prev_close_buy > 0:
                            if buy_price >= prev_close_buy * (1.0 + limit_pct * 0.95):
                                continue
                            if buy_price <= prev_close_buy * (1.0 - limit_pct * 0.95):
                                continue

                        buy_price = buy_price * (1.0 + SLIPPAGE_BPS_PER_SIDE / 10000.0)
                        if pd.isna(buy_price) or buy_price <= 0:
                            continue

                        executed_buys.append((code, float(buy_price)))

                    for code, buy_price in executed_buys:
                        # Round down to nearest 100 shares (1 lot)
                        raw_shares = allocatable_per_stock / buy_price
                        trade_shares = int(raw_shares // 100) * 100
                        if trade_shares < 100:
                            continue
                        trade_value = trade_shares * buy_price
                        cost = trade_value * COMMISSION_RATE
                        actual_invested = trade_value + cost
                        if cash >= actual_invested:
                            holdings[code] = {"shares": trade_shares}
                            cash -= actual_invested
                            self.trade_log.append(
                                {
                                    "date": today.strftime("%Y-%m-%d"),
                                    "code": code,
                                    "action": "BUY",
                                    "price": round(buy_price, 4),
                                    "shares": trade_shares,
                                    "value": round(float(trade_value), 2),
                                    "reason": "fill",
                                }
                            )

                pending_buys = []

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

            need_factor_refresh = (today in month_ends) or (not initialized and today >= first_factor_date)
            if need_factor_refresh:
                valid_codes = self.filter.filter_universe(today)
                if len(valid_codes) >= 50:
                    factor_cache = self.factor_engine.compute_factors(valid_codes, today)
                    if len(factor_cache) > 0 and not initialized:
                        buy_candidates = self.trigger_engine.get_buy_candidates(set(), TARGET_POSITIONS, factor_cache)
                        if buy_candidates:
                            pending_buys = buy_candidates
                            initialized = True

            frozen_status, hs300_c, hs300_m = self.circuit_breaker.is_frozen(today)

            if initialized and factor_cache is not None and len(holdings) > 0:
                sell_codes, sell_reasons = self.trigger_engine.check_sell_triggers(list(holdings.keys()), today, factor_cache)
                if sell_codes:
                    pending_sells.update(sell_codes)
                    pending_sell_reasons.update(sell_reasons)

                current_count = len(holdings) - len(pending_sells)
                n_needed = TARGET_POSITIONS - current_count
                if n_needed > 0:
                    current_holding_codes = set(holdings.keys()) - pending_sells
                    buy_candidates = self.trigger_engine.get_buy_candidates(current_holding_codes, n_needed, factor_cache)
                    if buy_candidates:
                        pending_buys = buy_candidates

            self.daily_status.append(
                {
                    "date": today.strftime("%Y-%m-%d"),
                    "nav": round(float(nav), 2),
                    "n_holdings": len(holdings),
                    "n_sold": len(pending_sells),
                    "n_bought": len(pending_buys),
                    "frozen": frozen_status,
                    "hs300": round(float(hs300_c), 2) if pd.notna(hs300_c) else np.nan,
                    "hs300_ma60": round(float(hs300_m), 2) if pd.notna(hs300_m) else np.nan,
                }
            )

            if initialized and len(holdings) == 0 and len(pending_buys) == 0:
                if factor_cache is not None:
                    buy_candidates = self.trigger_engine.get_buy_candidates(set(), TARGET_POSITIONS, factor_cache)
                    if buy_candidates:
                        pending_buys = buy_candidates

        print("\nBacktest completed.")
        print(f"  Total trades: {len(self.trade_log)}")
        n_sells = sum(1 for t in self.trade_log if t["action"] == "SELL")
        n_buys = sum(1 for t in self.trade_log if t["action"] == "BUY")
        print(f"  Sells: {n_sells}, Buys: {n_buys}")


class PerformanceEvaluator:
    def __init__(self, daily_nav, trade_log, daily_status=None):
        self.nav_df = pd.DataFrame(daily_nav, columns=["date", "nav"])
        self.trade_log = trade_log
        self.daily_status = pd.DataFrame(daily_status) if daily_status is not None else pd.DataFrame()

    def evaluate(self):
        print("\n" + "=" * 60)
        print("  Performance")
        print("=" * 60)

        if len(self.nav_df) == 0:
            print("No NAV data.")
            return

        nav = self.nav_df["nav"].values
        total_ret = nav[-1] / nav[0] - 1.0
        n_years = (self.nav_df["date"].iloc[-1] - self.nav_df["date"].iloc[0]).days / 365.25
        annual_ret = (1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0
        daily_rets = np.diff(nav) / nav[:-1]
        sharpe = (np.mean(daily_rets) - RISK_FREE_RATE / 252.0) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)
        peak = np.maximum.accumulate(nav)
        drawdown = (nav - peak) / peak
        max_dd = drawdown.min()

        print(f"\nInitial capital: {INITIAL_CAPITAL:,.0f}")
        print(f"Final NAV:       {nav[-1]:,.0f}")
        print(f"Total return:    {total_ret:.2%}")
        print(f"Annual return:   {annual_ret:.2%}")
        print(f"Max drawdown:    {max_dd:.2%}")
        print(f"Sharpe:          {sharpe:.2f}")
        print(f"Backtest days:   {len(nav)}")

        if self.trade_log:
            tl = pd.DataFrame(self.trade_log)
            n_sells = len(tl[tl["action"] == "SELL"])
            n_buys = len(tl[tl["action"] == "BUY"])
            print("\nTrade stats")
            print(f"  Total trades: {len(tl)}")
            print(f"  Sells:        {n_sells}")
            print(f"  Buys:         {n_buys}")

            sells = tl[tl["action"] == "SELL"]
            if len(sells) > 0:
                itam_sells = len(sells[sells["reason"].str.contains("ITAM", na=False)])
                cbmom_sells = len(sells[sells["reason"].str.contains("CBMOM", na=False)])
                ivol_sells = len(sells[sells["reason"].str.contains("IVOL", na=False)])
                other_sells = n_sells - itam_sells - cbmom_sells - ivol_sells
                print("\nSell reason breakdown")
                print(f"  ITAM:  {itam_sells}")
                print(f"  CBMOM: {cbmom_sells}")
                print(f"  IVOL:  {ivol_sells}")
                print(f"  Other: {other_sells}")

    def plot(self, save_path=None):
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

        ax1 = axes[0]
        if len(self.nav_df) > 0:
            norm_nav = self.nav_df["nav"] / INITIAL_CAPITAL
            ax1.plot(self.nav_df["date"], norm_nav, label="event-driven strategy", color="#1f77b4", linewidth=1.5)
            nav_arr = norm_nav.values
            peak = np.maximum.accumulate(nav_arr)
            dd = (nav_arr - peak) / peak
            ax1_dd = ax1.twinx()
            ax1_dd.fill_between(self.nav_df["date"], dd, 0, alpha=0.15, color="#d62728", label="drawdown")
            ax1_dd.set_ylabel("Drawdown")
            ax1_dd.set_ylim([-0.6, 0])
            ax1_dd.legend(fontsize=9, loc="lower right")

        ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax1.set_ylabel("NAV (start=1.0)")
        ax1.set_title(
            "Event-driven strategy: low IVOL + top ITAM + Delta IVOL<0 + CBMOM>0\n"
            f"{START_DATE} ~ {END_DATE} | daily scan | equal weight | "
            f"Buy ITAM top {int(ITAM_TOP_PCT * 100)}% | Sell keep top {int(ITAM_SELL_TOP_PCT * 100)}%",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend(fontsize=11, loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        if len(self.daily_status) > 0:
            status = self.daily_status.copy()
            status["date"] = pd.to_datetime(status["date"])
            ax2.plot(status["date"], status["n_holdings"], color="#2ca02c", linewidth=1.2)
        ax2.set_ylabel("Holdings")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\nChart saved: {save_path}")
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Event-driven old-main backtest with ITAM filter")
    parser.add_argument(
        "--start",
        default=START_DATE,
        help=f"Backtest start date in YYYY-MM-DD format. Default: {START_DATE}",
    )
    parser.add_argument(
        "--end",
        default=END_DATE,
        help=f"Backtest end date in YYYY-MM-DD format. Default: {END_DATE}",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=TARGET_POSITIONS,
        help=f"Target number of holdings. Default: {TARGET_POSITIONS}",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=SLIPPAGE_BPS_PER_SIDE,
        help=f"Per-side slippage in bps applied to open-price execution. Default: {SLIPPAGE_BPS_PER_SIDE}",
    )
    parser.add_argument(
        "--intraday-dir",
        default=INTRADAY_15M_HFQ_DIR,
        help=f"15m HFQ directory used to build ITAM when cache is missing. Default: {INTRADAY_15M_HFQ_DIR}",
    )
    parser.add_argument(
        "--itam-panel",
        default=DEFAULT_ITAM_PANEL_PATH,
        help=f"Cached ITAM panel path. Default: {DEFAULT_ITAM_PANEL_PATH}",
    )
    parser.add_argument(
        "--itam-top-pct",
        type=float,
        default=ITAM_TOP_PCT,
        help=f"Buy filter percentile for highest ITAM illiquidity. Default: {ITAM_TOP_PCT:.2f}",
    )
    parser.add_argument(
        "--itam-sell-top-pct",
        type=float,
        default=ITAM_SELL_TOP_PCT,
        help=f"Sell keep-zone percentile for highest ITAM illiquidity. Default: {ITAM_SELL_TOP_PCT:.2f}",
    )
    parser.add_argument(
        "--output-prefix",
        default="event_driven_old_main_itam",
        help="Prefix for output figure/csv files.",
    )
    return parser.parse_args()


def set_backtest_window(start_date, end_date):
    global START_DATE, END_DATE

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if start_ts > end_ts:
        raise ValueError(f"start date {start_date} is after end date {end_date}")

    START_DATE = start_ts.strftime("%Y-%m-%d")
    END_DATE = end_ts.strftime("%Y-%m-%d")


def set_target_positions(target_positions):
    global TARGET_POSITIONS

    if int(target_positions) <= 0:
        raise ValueError(f"target positions must be positive, got {target_positions}")

    TARGET_POSITIONS = int(target_positions)


def set_slippage_bps(slippage_bps):
    global SLIPPAGE_BPS_PER_SIDE

    if float(slippage_bps) < 0:
        raise ValueError(f"slippage bps must be non-negative, got {slippage_bps}")

    SLIPPAGE_BPS_PER_SIDE = float(slippage_bps)


def set_itam_top_pct(itam_top_pct):
    global ITAM_TOP_PCT

    pct = float(itam_top_pct)
    if pct <= 0 or pct >= 1:
        raise ValueError(f"itam top pct must be in (0, 1), got {itam_top_pct}")

    ITAM_TOP_PCT = pct


def set_itam_sell_top_pct(itam_sell_top_pct):
    global ITAM_SELL_TOP_PCT

    pct = float(itam_sell_top_pct)
    if pct <= 0 or pct >= 1:
        raise ValueError(f"itam sell top pct must be in (0, 1), got {itam_sell_top_pct}")

    if pct < ITAM_TOP_PCT:
        raise ValueError(
            f"itam sell top pct must be >= itam buy top pct, got sell={itam_sell_top_pct}, buy={ITAM_TOP_PCT}"
        )

    ITAM_SELL_TOP_PCT = pct


def main():
    args = parse_args()
    set_backtest_window(args.start, args.end)
    set_target_positions(args.positions)
    set_slippage_bps(args.slippage_bps)
    set_itam_top_pct(args.itam_top_pct)
    set_itam_sell_top_pct(args.itam_sell_top_pct)

    t0 = datetime.now()
    print("=" * 60)
    print("  Event-driven backtest with ITAM filter")
    print("  low IVOL -> high ITAM -> Delta IVOL < 0 -> CBMOM > 0")
    print("  buy ranking = Delta IVOL ascending | sell = ITAM fade or CBMOM < 0 or IVOL spike")
    print(f"  Window: {START_DATE} ~ {END_DATE}")
    print(f"  Target positions: {TARGET_POSITIONS}")
    print(f"  Slippage: {SLIPPAGE_BPS_PER_SIDE:.2f} bps/side")
    print(f"  ITAM buy percentile:  {ITAM_TOP_PCT:.2%}")
    print(f"  ITAM sell keep zone:  {ITAM_SELL_TOP_PCT:.2%}")
    print(f"  Start time: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    loader = DataLoader()
    loader.load_stock_data()
    loader.load_ff3_factors()
    loader.load_hs300()
    loader.compute_returns()

    itam_panel = load_itam_panel(
        panel_path=args.itam_panel,
        intraday_dir=args.intraday_dir,
        start=pd.Timestamp(START_DATE),
        end=pd.Timestamp(END_DATE),
    )
    if len(itam_panel) == 0:
        raise RuntimeError("ITAM panel is empty in the requested backtest window.")
    itam_lookup = {
        month: frame[["code", "itam", "itam_n_days"]].copy()
        for month, frame in itam_panel.groupby("month", sort=True)
    }
    print(
        f"[ITAM] Coverage: {itam_panel['month'].min()} -> {itam_panel['month'].max()} | "
        f"rows={len(itam_panel)}"
    )

    engine = EventDrivenBacktestEngine(loader, itam_lookup=itam_lookup)
    engine.run()

    evaluator = PerformanceEvaluator(engine.daily_nav, engine.trade_log, engine.daily_status)
    evaluator.evaluate()

    save_path = os.path.join(BASE_DIR, f"{args.output_prefix}_backtest_result.png")
    evaluator.plot(save_path=save_path)

    if engine.trade_log:
        trade_path = os.path.join(BASE_DIR, f"{args.output_prefix}_trades.csv")
        pd.DataFrame(engine.trade_log).to_csv(trade_path, index=False, encoding="utf-8-sig")
        print(f"Trade log exported: {trade_path} ({len(engine.trade_log)} rows)")

    if engine.daily_status:
        status_path = os.path.join(BASE_DIR, f"{args.output_prefix}_daily_status.csv")
        pd.DataFrame(engine.daily_status).to_csv(status_path, index=False, encoding="utf-8-sig")
        print(f"Daily status exported: {status_path}")

    if engine.daily_nav:
        nav_path = os.path.join(BASE_DIR, f"{args.output_prefix}_nav.csv")
        pd.DataFrame(engine.daily_nav, columns=["date", "nav"]).to_csv(nav_path, index=False, encoding="utf-8-sig")
        print(f"NAV exported: {nav_path}")

    t1 = datetime.now()
    print(f"\nTotal runtime: {(t1 - t0).total_seconds():.1f}s")


if __name__ == "__main__":
    main()
