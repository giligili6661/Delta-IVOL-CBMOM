import argparse
import json
import os
import time
from bisect import bisect_left
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import akshare as ak

import backtest_event_driven_old_main as base
from csi_size_universe import (
    build_csi300_reconstructed_panels,
    build_csi1000_reconstructed_panels,
    build_csi2000_reconstructed_panels,
)
from backtest_event_driven_old_main_attention_penalty012 import (
    DEFAULT_END,
    DEFAULT_MAX_NEW_BUYS,
    DEFAULT_PENALTY_WEIGHT,
    DEFAULT_START,
    DailyCappedAttentionPenaltyBacktestEngine,
)
from backtest_event_driven_old_main_attention_variants import (
    AttentionPanel,
    AttentionTriggerEngine,
    DEFAULT_CACHE_NAME,
    INTRADAY_DIR,
    build_attention_panel,
    compute_summary,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOCK_DATA_NAME = "__calendar__"
DEFAULT_OUTPUT_PREFIX = "backtrader_old_main_attention_penalty012_maxbuy10_full"
DEFAULT_ATTENTION_TSMEAN_WINDOW = 1
DEFAULT_SHORT_IVOL_TSMEAN_WINDOW = 1
DEFAULT_BUY_VOL_SIGNAL = "long"
DEFAULT_BUY_SHORT_IVOL_FILTER_PCT = 0.50
DEFAULT_REFRESH_TRADING_STEP = 0
DEFAULT_SHORT_IVOL_WINSOR_PCT = 0.01
DEFAULT_RA_MIN_OBS = 200
DEFAULT_RA_KDE_GRID_SIZE = 81
DEFAULT_RA_KDE_GRID_MAX = 4.0
DEFAULT_RA_KDE_BANDWIDTH_FLOOR = 0.15
DEFAULT_UNIVERSE_INDEX = "all"
DEFAULT_CSI300_RECON_EFFECTIVE_PANEL = os.path.join(BASE_DIR, "tmp", "csi300_hist_reconstructed_effective_panel.csv")
DEFAULT_CSI300_RECON_MONTHLY_PANEL = os.path.join(BASE_DIR, "tmp", "csi300_hist_reconstructed_monthly_panel.csv")
DEFAULT_CSI300_RECON_METRIC_CACHE = os.path.join(BASE_DIR, "tmp", "csi300_hist_reconstructed_metric_cache.csv")
DEFAULT_CSI1000_RECON_EFFECTIVE_PANEL = os.path.join(BASE_DIR, "tmp", "csi1000_hist_reconstructed_effective_panel.csv")
DEFAULT_CSI1000_RECON_MONTHLY_PANEL = os.path.join(BASE_DIR, "tmp", "csi1000_hist_reconstructed_monthly_panel.csv")
DEFAULT_CSI1000_RECON_METRIC_CACHE = os.path.join(BASE_DIR, "tmp", "csi1000_hist_reconstructed_metric_cache.csv")
DEFAULT_CSI2000_RECON_EFFECTIVE_PANEL = os.path.join(BASE_DIR, "tmp", "csi2000_hist_reconstructed_effective_panel.csv")
DEFAULT_CSI2000_RECON_MONTHLY_PANEL = os.path.join(BASE_DIR, "tmp", "csi2000_hist_reconstructed_monthly_panel.csv")
DEFAULT_CSI2000_RECON_METRIC_CACHE = os.path.join(BASE_DIR, "tmp", "csi2000_hist_reconstructed_metric_cache.csv")
DEFAULT_BUY_EXECUTION_MODE = "open"
DEFAULT_BUY_15M_EXEC_DIR = os.path.join(BASE_DIR, "data_stock_15m_unadj")
DEFAULT_BUY_15M_IVOL_LOOKBACK_DAYS = 20
DEFAULT_BUY_15M_IVOL_TRIGGER_RATIO = 0.85
DEFAULT_BUY_15M_IVOL_MIN_BARS = 4
DEFAULT_BUY_15M_FALLBACK_MODE = "vwap"
DEFAULT_BUY_15M_CACHE_SIZE = 256
DEFAULT_INTRADAY_ATTENTION_CACHE = os.path.join(BASE_DIR, "intraday_attention_early2_panel_20200102_20260318.npz")
DEFAULT_INTRADAY_ATTENTION_EARLY_BARS = 2
DEFAULT_INTRADAY_ATTENTION_BUY_THRESHOLD = 0.50
DEFAULT_INTRADAY_ATTENTION_SELL_THRESHOLD = 0.50
DEFAULT_INTRADAY_ATTENTION_EXEC_MODE = "hybrid"
DEFAULT_INTRADAY_FACTOR_LOOKBACK_DAYS = 20
DEFAULT_INTRADAY_FACTOR_EARLY_BARS = 2
DEFAULT_INTRADAY_FACTOR_BUY_THRESHOLD = 0.50
DEFAULT_INTRADAY_FACTOR_SELL_THRESHOLD = 0.50
DEFAULT_INTRADAY_FACTOR_EXEC_MODE = "hybrid"


class ChinaAStockCommInfo(bt.CommInfoBase):
    params = (
        ("commission", base.COMMISSION_RATE),
        ("stamp_duty", base.STAMP_DUTY_RATE),
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        rate = self.p.commission
        if size < 0:
            rate += self.p.stamp_duty
        return abs(size) * price * rate


def bar_vwap(row):
    volume = row.get("volume", np.nan)
    amount = row.get("amount", np.nan)
    if pd.notna(volume) and pd.notna(amount) and volume > 0 and amount > 0:
        return float(amount / volume)
    vals = [row.get("open", np.nan), row.get("high", np.nan), row.get("low", np.nan), row.get("close", np.nan)]
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def bar_twap(row):
    vals = [row.get("open", np.nan), row.get("high", np.nan), row.get("low", np.nan), row.get("close", np.nan)]
    vals = [float(v) for v in vals if pd.notna(v)]
    return float(np.mean(vals)) if vals else np.nan


def select_exec_price(vwap_price, twap_price, mode):
    if mode == "vwap":
        return vwap_price
    if mode == "twap":
        return twap_price
    if pd.notna(vwap_price) and pd.notna(twap_price):
        return float((vwap_price + twap_price) / 2.0)
    if pd.notna(vwap_price):
        return float(vwap_price)
    if pd.notna(twap_price):
        return float(twap_price)
    return np.nan


def fetch_csindex_constituent_snapshot(symbol, cache_dir=None):
    symbol = str(symbol).strip()
    cache_dir = cache_dir or os.path.join(BASE_DIR, "tmp")
    os.makedirs(cache_dir, exist_ok=True)

    df = ak.index_stock_cons_csindex(symbol=symbol)
    if df is None or df.empty:
        raise RuntimeError(f"No constituent snapshot returned for CSIndex symbol={symbol}")

    out = df.copy()
    out["成分券代码"] = out["成分券代码"].astype(str).str.zfill(6)
    snapshot_date = pd.to_datetime(out["日期"]).max().strftime("%Y%m%d")
    cache_path = os.path.join(cache_dir, f"csindex_{symbol}_snapshot_{snapshot_date}.csv")
    out.to_csv(cache_path, index=False, encoding="utf-8-sig")

    codes = tuple(sorted(out["成分券代码"].astype(str).unique().tolist()))
    return {
        "symbol": symbol,
        "snapshot_date": snapshot_date,
        "codes": codes,
        "cache_path": cache_path,
    }


class NoPennyUniverseFilter(base.UniverseFilter):
    """UniverseFilter that skips the penny-stock (close < MIN_PRICE) filter."""

    def filter_universe(self, date):
        valid_codes = set(self.df_hfq["code"].unique())

        st_codes = self.get_st_codes(date)
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

        # NOTE: penny stock filter (close < MIN_PRICE) intentionally skipped
        return list(valid_codes)


class RestrictedUniverseFilter:
    def __init__(self, base_filter, allowed_codes, label):
        self.base_filter = base_filter
        self.allowed_codes = set(str(code).zfill(6) for code in allowed_codes)
        self.label = str(label)

    def filter_universe(self, date):
        base_codes = self.base_filter.filter_universe(date)
        return [code for code in base_codes if code in self.allowed_codes]

    def get_st_codes(self, date):
        return self.base_filter.get_st_codes(date)

    def is_st(self, date, code):
        return self.base_filter.is_st(date, code)


class TimeVaryingUniverseFilter:
    def __init__(self, base_filter, effective_entries, label):
        self.base_filter = base_filter
        entries = []
        for eff_date, codes in (effective_entries or tuple()):
            ts = pd.Timestamp(eff_date).normalize()
            code_set = set(str(code).zfill(6) for code in codes)
            entries.append((ts, code_set))
        entries.sort(key=lambda item: item[0])
        self.effective_dates = [item[0] for item in entries]
        self.code_sets = [item[1] for item in entries]
        self.label = str(label)

    def _allowed_codes(self, date):
        if not self.effective_dates:
            return set()
        ts = pd.Timestamp(date).normalize()
        idx = bisect_left(self.effective_dates, ts)
        if idx < len(self.effective_dates) and self.effective_dates[idx] == ts:
            return self.code_sets[idx]
        idx -= 1
        if idx < 0:
            return set()
        return self.code_sets[idx]

    def filter_universe(self, date):
        base_codes = self.base_filter.filter_universe(date)
        allowed = self._allowed_codes(date)
        return [code for code in base_codes if code in allowed]

    def get_st_codes(self, date):
        return self.base_filter.get_st_codes(date)

    def is_st(self, date, code):
        return self.base_filter.is_st(date, code)


class ApproxRADensityMixin:
    _ra_grid = np.linspace(
        -DEFAULT_RA_KDE_GRID_MAX,
        DEFAULT_RA_KDE_GRID_MAX,
        DEFAULT_RA_KDE_GRID_SIZE,
        dtype=np.float64,
    )
    _ra_grid_step = float(_ra_grid[1] - _ra_grid[0])
    _gaussian_norm = float(np.sqrt(2.0 * np.pi))

    @classmethod
    def _resolve_mode_sign(cls, density_col, mode_idx, standardized_resid):
        mode_value = float(cls._ra_grid[mode_idx])
        if abs(mode_value) > cls._ra_grid_step * 0.5:
            return float(np.sign(mode_value))

        left_peak = float(np.max(density_col[:mode_idx])) if mode_idx > 0 else -np.inf
        right_peak = float(np.max(density_col[mode_idx + 1 :])) if mode_idx + 1 < len(density_col) else -np.inf
        if right_peak > left_peak:
            return 1.0
        if left_peak > right_peak:
            return -1.0

        skew_proxy = float(np.nanmean(np.power(standardized_resid, 3)))
        return float(np.sign(skew_proxy))

    @classmethod
    def _compute_approx_ra_from_resid(cls, resid_matrix, chunk_size=256):
        resid_matrix = np.asarray(resid_matrix, dtype=np.float64)
        if resid_matrix.ndim != 2 or resid_matrix.size == 0:
            return np.array([], dtype=np.float64)

        n_obs, n_codes = resid_matrix.shape
        ra_arr = np.full(n_codes, np.nan, dtype=np.float64)
        if n_obs < DEFAULT_RA_MIN_OBS:
            return ra_arr

        bandwidth = max(
            1.06 * (float(n_obs) ** (-1.0 / 5.0)),
            DEFAULT_RA_KDE_BANDWIDTH_FLOOR,
        )

        for start in range(0, n_codes, max(int(chunk_size), 1)):
            end = min(start + max(int(chunk_size), 1), n_codes)
            resid_block = resid_matrix[:, start:end]
            std_block = np.std(resid_block, axis=0, ddof=1)
            valid_mask = np.isfinite(std_block) & (std_block > 1e-12)
            if not valid_mask.any():
                continue

            valid_cols = np.flatnonzero(valid_mask)
            standardized = resid_block[:, valid_cols] / std_block[valid_cols]
            kde_arg = (cls._ra_grid[:, None, None] - standardized[None, :, :]) / bandwidth
            density = np.exp(-0.5 * kde_arg * kde_arg).mean(axis=1) / (bandwidth * cls._gaussian_norm)
            density = np.maximum(density, 1e-12)
            mode_idx = np.argmax(density, axis=0)

            block_ra = np.zeros(len(valid_cols), dtype=np.float64)
            for local_idx, grid_idx in enumerate(mode_idx):
                mode_sign = cls._resolve_mode_sign(density[:, local_idx], int(grid_idx), standardized[:, local_idx])
                if mode_sign == 0:
                    continue
                span = min(int(grid_idx), density.shape[0] - 1 - int(grid_idx))
                if span <= 0:
                    continue

                forward = np.sqrt(density[grid_idx + 1 : grid_idx + span + 1, local_idx])
                backward = np.sqrt(density[grid_idx - 1 : grid_idx - span - 1 : -1, local_idx])
                if len(forward) == 0 or len(backward) == 0:
                    continue
                block_ra[local_idx] = mode_sign * float(np.sum((forward - backward) ** 2) * cls._ra_grid_step)

            sub_arr = ra_arr[start:end].copy()
            sub_arr[valid_cols] = block_ra
            ra_arr[start:end] = sub_arr

        return ra_arr


class IntradayIvolBuyExecutor:
    def __init__(
        self,
        intraday_dir,
        lookback_days=DEFAULT_BUY_15M_IVOL_LOOKBACK_DAYS,
        trigger_ratio=DEFAULT_BUY_15M_IVOL_TRIGGER_RATIO,
        min_bars=DEFAULT_BUY_15M_IVOL_MIN_BARS,
        fallback_mode=DEFAULT_BUY_15M_FALLBACK_MODE,
        cache_size=DEFAULT_BUY_15M_CACHE_SIZE,
    ):
        self.intraday_dir = os.path.abspath(intraday_dir)
        self.lookback_days = max(int(lookback_days), 1)
        self.trigger_ratio = float(trigger_ratio)
        self.min_bars = max(int(min_bars), 3)
        self.fallback_mode = str(fallback_mode)
        self.cache_size = max(int(cache_size), 1)
        self._cache = OrderedDict()

    def _evict_if_needed(self):
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _load_code_payload(self, code):
        if code in self._cache:
            self._cache.move_to_end(code)
            return self._cache[code]

        path = os.path.join(self.intraday_dir, f"{code}.csv")
        if not os.path.exists(path):
            self._cache[code] = None
            self._evict_if_needed()
            return None

        try:
            df = pd.read_csv(path, encoding="utf-8-sig", usecols=[0, 1, 2, 3, 4, 5, 6])
        except Exception:
            self._cache[code] = None
            self._evict_if_needed()
            return None

        if df.empty or df.shape[1] < 7:
            self._cache[code] = None
            self._evict_if_needed()
            return None

        df.columns = ["time", "open", "high", "low", "close", "volume", "amount"]
        df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["time"])
        if df.empty:
            self._cache[code] = None
            self._evict_if_needed()
            return None

        df["time"] = df["time"].astype(np.int64)
        df = df.sort_values("time").copy()
        df["date_key"] = (df["time"] // 1000000).astype(str)
        df["bar_key"] = (df["time"] % 1000000).astype(int)
        day_map = {day: day_df.reset_index(drop=True) for day, day_df in df.groupby("date_key", sort=True)}
        day_keys = sorted(day_map)
        payload = {"day_map": day_map, "day_keys": day_keys}

        self._cache[code] = payload
        self._evict_if_needed()
        return payload

    @staticmethod
    def _prefix_ivol(day_df, end_idx):
        if day_df is None or len(day_df) <= end_idx:
            return np.nan
        prefix = day_df.iloc[: end_idx + 1]
        close_arr = prefix["close"].to_numpy(dtype=np.float64, copy=False)
        if len(close_arr) < 3:
            return np.nan

        prev_close = close_arr[:-1]
        cur_close = close_arr[1:]
        mask = np.isfinite(prev_close) & np.isfinite(cur_close) & (prev_close > 0) & (cur_close > 0)
        if int(mask.sum()) < 2:
            return np.nan

        rets = np.log(cur_close[mask] / prev_close[mask])
        if len(rets) < 2:
            return np.nan
        return float(np.std(rets, ddof=1))

    def _build_tail_candidate(self, today_df):
        if today_df is None or today_df.empty:
            return None

        tail_df = today_df[today_df["bar_key"] == 150000]
        tail_row = tail_df.iloc[-1] if not tail_df.empty else today_df.iloc[-1]
        tail_raw_price = select_exec_price(
            bar_vwap(tail_row.to_dict()),
            bar_twap(tail_row.to_dict()),
            self.fallback_mode,
        )
        if pd.isna(tail_raw_price) or tail_raw_price <= 0:
            return None

        return {
            "kind": "fallback",
            "signal_bar_key": int(tail_row["bar_key"]),
            "exec_bar_key": int(tail_row["bar_key"]),
            "raw_price": float(tail_raw_price),
            "ivol_ratio": np.nan,
            "current_ivol": np.nan,
            "baseline_ivol": np.nan,
        }

    def list_buy_candidates(self, code, date):
        payload = self._load_code_payload(code)
        if not payload:
            return []

        date_key = pd.Timestamp(date).strftime("%Y%m%d")
        day_map = payload["day_map"]
        day_keys = payload["day_keys"]
        today_df = day_map.get(date_key)
        if today_df is None or today_df.empty:
            return []

        current_pos = bisect_left(day_keys, date_key)
        history_keys = day_keys[max(0, current_pos - self.lookback_days) : current_pos]
        candidates = []

        max_eval_idx = len(today_df) - 2
        for eval_idx in range(self.min_bars - 1, max_eval_idx + 1):
            current_ivol = self._prefix_ivol(today_df, eval_idx)
            if pd.isna(current_ivol) or current_ivol <= 0:
                continue

            hist_ivols = []
            for hist_key in history_keys:
                hist_df = day_map.get(hist_key)
                hist_ivol = self._prefix_ivol(hist_df, eval_idx)
                if pd.notna(hist_ivol) and hist_ivol > 0:
                    hist_ivols.append(float(hist_ivol))

            if len(hist_ivols) < max(3, min(self.lookback_days, 5)):
                continue

            baseline_ivol = float(np.nanmedian(hist_ivols))
            if not np.isfinite(baseline_ivol) or baseline_ivol <= 0:
                continue

            ratio = current_ivol / baseline_ivol
            if not np.isfinite(ratio) or ratio > self.trigger_ratio:
                continue

            exec_row = today_df.iloc[eval_idx + 1]
            raw_exec_price = float(exec_row["open"]) if pd.notna(exec_row["open"]) else np.nan
            if pd.isna(raw_exec_price) or raw_exec_price <= 0:
                continue

            signal_row = today_df.iloc[eval_idx]
            candidates.append(
                {
                    "kind": "triggered",
                    "signal_bar_key": int(signal_row["bar_key"]),
                    "exec_bar_key": int(exec_row["bar_key"]),
                    "raw_price": raw_exec_price,
                    "ivol_ratio": float(ratio),
                    "current_ivol": float(current_ivol),
                    "baseline_ivol": float(baseline_ivol),
                }
            )

        tail_candidate = self._build_tail_candidate(today_df)
        if tail_candidate is not None:
            candidates.append(tail_candidate)

        return candidates

    def list_tail_buy_candidates(self, code, date):
        payload = self._load_code_payload(code)
        if not payload:
            return []

        date_key = pd.Timestamp(date).strftime("%Y%m%d")
        today_df = payload["day_map"].get(date_key)
        tail_candidate = self._build_tail_candidate(today_df)
        return [tail_candidate] if tail_candidate is not None else []


class IntradayFactor3Executor:
    def __init__(
        self,
        intraday_dir,
        lookback_days=DEFAULT_INTRADAY_FACTOR_LOOKBACK_DAYS,
        early_bars=DEFAULT_INTRADAY_FACTOR_EARLY_BARS,
        exec_mode=DEFAULT_INTRADAY_FACTOR_EXEC_MODE,
        cache_size=DEFAULT_BUY_15M_CACHE_SIZE,
    ):
        self.intraday_dir = os.path.abspath(intraday_dir)
        self.lookback_days = max(int(lookback_days), 3)
        self.early_bars = max(int(early_bars), 2)
        self.exec_mode = str(exec_mode)
        self.cache_size = max(int(cache_size), 1)
        self._cache = OrderedDict()

    def _evict_if_needed(self):
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _load_code_payload(self, code):
        if code in self._cache:
            self._cache.move_to_end(code)
            return self._cache[code]

        path = os.path.join(self.intraday_dir, f"{code}.csv")
        if not os.path.exists(path):
            self._cache[code] = None
            self._evict_if_needed()
            return None

        try:
            df = pd.read_csv(path, encoding="utf-8-sig", usecols=[0, 1, 2, 3, 4, 5, 6])
        except Exception:
            self._cache[code] = None
            self._evict_if_needed()
            return None

        if df.empty or df.shape[1] < 7:
            self._cache[code] = None
            self._evict_if_needed()
            return None

        df.columns = ["time", "open", "high", "low", "close", "volume", "amount"]
        df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["time"])
        if df.empty:
            self._cache[code] = None
            self._evict_if_needed()
            return None

        df["time"] = df["time"].astype(np.int64)
        df = df.sort_values("time").copy()
        df["date_key"] = (df["time"] // 1000000).astype(str)
        df["bar_key"] = (df["time"] % 1000000).astype(int)
        day_map = {day: day_df.reset_index(drop=True) for day, day_df in df.groupby("date_key", sort=True)}
        day_keys = sorted(day_map)
        payload = {"day_map": day_map, "day_keys": day_keys}

        self._cache[code] = payload
        self._evict_if_needed()
        return payload

    @staticmethod
    def _extract_early_returns(day_df, early_bars):
        if day_df is None or day_df.empty:
            return np.array([], dtype=np.float64)

        close_arr = day_df["close"].to_numpy(dtype=np.float64, copy=False)
        if len(close_arr) < 3:
            return np.array([], dtype=np.float64)

        rets = []
        for row_idx in range(1, len(close_arr)):
            prev_close = close_arr[row_idx - 1]
            curr_close = close_arr[row_idx]
            if not (np.isfinite(prev_close) and np.isfinite(curr_close) and prev_close > 0 and curr_close > 0):
                continue
            rets.append(float(np.log(curr_close / prev_close)))
            if len(rets) >= early_bars:
                break
        return np.asarray(rets, dtype=np.float64)

    @staticmethod
    def _compute_ivol(rets):
        if rets is None or len(rets) < 2:
            return np.nan
        return float(np.std(rets, ddof=1))

    @staticmethod
    def _compute_cbmom(rets):
        if rets is None or len(rets) == 0:
            return np.nan
        denom = float(np.sum(np.abs(rets)))
        if not np.isfinite(denom) or denom <= 0:
            return 0.0
        return float(np.sum(rets) / denom)

    @staticmethod
    def _score_from_history(value, hist_values, prefer_low):
        if not np.isfinite(value):
            return np.nan
        arr = np.asarray(hist_values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        if prefer_low:
            return float(np.mean(arr >= value))
        return float(np.mean(arr <= value))

    def _get_exec_prices(self, day_df):
        early_price = np.nan
        late_price = np.nan
        if day_df is None or day_df.empty:
            return early_price, late_price

        bar_1015_df = day_df[day_df["bar_key"] == 101500]
        bar_1030_df = day_df[day_df["bar_key"] == 103000]
        pieces = []
        if not bar_1015_df.empty:
            pieces.append((bar_1015_df.iloc[-1].to_dict(), 10.0 / 15.0))
        if not bar_1030_df.empty:
            pieces.append((bar_1030_df.iloc[-1].to_dict(), 5.0 / 15.0))
        if pieces:
            twap_num = 0.0
            twap_den = 0.0
            vwap_num = 0.0
            vwap_den = 0.0
            for row_dict, ratio in pieces:
                twap_price = bar_twap(row_dict)
                vwap_price = bar_vwap(row_dict)
                volume = float(row_dict.get("volume", np.nan))
                if pd.notna(twap_price):
                    twap_num += ratio * twap_price
                    twap_den += ratio
                if pd.notna(vwap_price):
                    weight = ratio * (volume if pd.notna(volume) and volume > 0 else 1.0)
                    vwap_num += weight * vwap_price
                    vwap_den += weight
            early_price = select_exec_price(
                vwap_num / vwap_den if vwap_den > 0 else np.nan,
                twap_num / twap_den if twap_den > 0 else np.nan,
                self.exec_mode,
            )

        bar_1500_df = day_df[day_df["bar_key"] == 150000]
        if not bar_1500_df.empty:
            row_dict = bar_1500_df.iloc[-1].to_dict()
            late_price = select_exec_price(bar_vwap(row_dict), bar_twap(row_dict), self.exec_mode)

        return early_price, late_price

    def get_snapshot(self, code, date):
        payload = self._load_code_payload(code)
        if not payload:
            return None

        date_key = pd.Timestamp(date).strftime("%Y%m%d")
        day_map = payload["day_map"]
        day_keys = payload["day_keys"]
        today_df = day_map.get(date_key)
        if today_df is None or today_df.empty:
            return None

        current_pos = bisect_left(day_keys, date_key)
        history_keys = day_keys[max(0, current_pos - self.lookback_days) : current_pos]
        today_rets = self._extract_early_returns(today_df, self.early_bars)
        current_ivol = self._compute_ivol(today_rets)
        current_cbmom = self._compute_cbmom(today_rets)

        hist_ivols = []
        hist_cbmoms = []
        for hist_key in history_keys:
            hist_df = day_map.get(hist_key)
            hist_rets = self._extract_early_returns(hist_df, self.early_bars)
            hist_ivol = self._compute_ivol(hist_rets)
            hist_cbmom = self._compute_cbmom(hist_rets)
            if np.isfinite(hist_ivol):
                hist_ivols.append(hist_ivol)
            if np.isfinite(hist_cbmom):
                hist_cbmoms.append(hist_cbmom)

        baseline_ivol = float(np.nanmedian(hist_ivols)) if hist_ivols else np.nan
        ivol_ratio = current_ivol / baseline_ivol if np.isfinite(current_ivol) and np.isfinite(baseline_ivol) and baseline_ivol > 0 else np.nan
        delta_ivol = current_ivol - baseline_ivol if np.isfinite(current_ivol) and np.isfinite(baseline_ivol) else np.nan

        ivol_buy_score = self._score_from_history(current_ivol, hist_ivols, prefer_low=True)
        ivol_sell_score = self._score_from_history(current_ivol, hist_ivols, prefer_low=False)
        cbmom_buy_score = self._score_from_history(current_cbmom, hist_cbmoms, prefer_low=False)
        cbmom_sell_score = self._score_from_history(current_cbmom, hist_cbmoms, prefer_low=True)
        delta_buy_score = 1.0 if np.isfinite(delta_ivol) and delta_ivol <= 0 else (0.0 if np.isfinite(delta_ivol) else np.nan)
        delta_sell_score = 1.0 if np.isfinite(delta_ivol) and delta_ivol >= 0 else (0.0 if np.isfinite(delta_ivol) else np.nan)

        buy_components = [score for score in [ivol_buy_score, delta_buy_score, cbmom_buy_score] if np.isfinite(score)]
        sell_components = [score for score in [ivol_sell_score, delta_sell_score, cbmom_sell_score] if np.isfinite(score)]
        buy_score = float(np.mean(buy_components)) if buy_components else np.nan
        sell_score = float(np.mean(sell_components)) if sell_components else np.nan
        exec_early_price, exec_late_price = self._get_exec_prices(today_df)

        return {
            "code": code,
            "ivol": current_ivol,
            "baseline_ivol": baseline_ivol,
            "ivol_ratio": ivol_ratio,
            "delta_ivol": delta_ivol,
            "cbmom": current_cbmom,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "ivol_buy_score": ivol_buy_score,
            "ivol_sell_score": ivol_sell_score,
            "delta_buy_score": delta_buy_score,
            "delta_sell_score": delta_sell_score,
            "cbmom_buy_score": cbmom_buy_score,
            "cbmom_sell_score": cbmom_sell_score,
            "exec_early_price": exec_early_price,
            "exec_late_price": exec_late_price,
        }


class IntradayAttentionTimingPanel:
    def __init__(
        self,
        codes,
        dates,
        attention_up,
        attention_down,
        crowding_pct,
        exec_1020_vwap,
        exec_1020_twap,
        exec_1450_vwap,
        exec_1450_twap,
    ):
        self.codes = np.array(codes)
        self.dates = pd.DatetimeIndex(pd.to_datetime(dates))
        self.attention_up = pd.DataFrame(attention_up, index=self.dates, columns=self.codes, dtype=np.float32)
        self.attention_down = pd.DataFrame(attention_down, index=self.dates, columns=self.codes, dtype=np.float32)
        self.crowding_pct = pd.DataFrame(crowding_pct, index=self.dates, columns=self.codes, dtype=np.float32)
        self.exec_1020_vwap = pd.DataFrame(exec_1020_vwap, index=self.dates, columns=self.codes, dtype=np.float32)
        self.exec_1020_twap = pd.DataFrame(exec_1020_twap, index=self.dates, columns=self.codes, dtype=np.float32)
        self.exec_1450_vwap = pd.DataFrame(exec_1450_vwap, index=self.dates, columns=self.codes, dtype=np.float32)
        self.exec_1450_twap = pd.DataFrame(exec_1450_twap, index=self.dates, columns=self.codes, dtype=np.float32)

    def get_metric(self, date, code, field):
        date = pd.Timestamp(date)
        table = getattr(self, field)
        if date not in table.index or code not in table.columns:
            return np.nan
        return table.at[date, code]

    def get_exec_price(self, date, code, exec_mode, late=False):
        date = pd.Timestamp(date)
        if late:
            vwap_price = self.get_metric(date, code, "exec_1450_vwap")
            twap_price = self.get_metric(date, code, "exec_1450_twap")
        else:
            vwap_price = self.get_metric(date, code, "exec_1020_vwap")
            twap_price = self.get_metric(date, code, "exec_1020_twap")
        return select_exec_price(vwap_price, twap_price, exec_mode)


def build_intraday_attention_timing_panel(
    cache_path,
    calendar_dates,
    signal_dir,
    exec_dir,
    early_bars=DEFAULT_INTRADAY_ATTENTION_EARLY_BARS,
    workers=8,
):
    req_dates = pd.DatetimeIndex(pd.to_datetime(calendar_dates))
    if req_dates.empty:
        raise RuntimeError("No calendar dates provided for intraday attention timing panel.")

    early_bars = max(int(early_bars), 1)
    req_start = req_dates.min()
    req_end = req_dates.max()

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        dates = pd.to_datetime(data["dates"].astype(str), format="%Y%m%d")
        cached_early_bars_arr = data["early_bars"] if "early_bars" in data.files else np.array([1], dtype=np.int16)
        cached_early_bars = int(np.asarray(cached_early_bars_arr).reshape(-1)[0])
        if cached_early_bars == early_bars and len(dates) > 0 and dates.min() <= req_start and dates.max() >= req_end:
            keep = (dates >= req_start) & (dates <= req_end)
            return IntradayAttentionTimingPanel(
                codes=data["codes"].astype(str),
                dates=dates[keep],
                attention_up=data["attention_up"][keep],
                attention_down=data["attention_down"][keep],
                crowding_pct=data["crowding_pct"][keep],
                exec_1020_vwap=data["exec_1020_vwap"][keep],
                exec_1020_twap=data["exec_1020_twap"][keep],
                exec_1450_vwap=data["exec_1450_vwap"][keep],
                exec_1450_twap=data["exec_1450_twap"][keep],
            )
        print(
            f"intraday attention cache exists but does not match requested window/bars "
            f"({req_start:%Y-%m-%d} ~ {req_end:%Y-%m-%d}, early_bars={early_bars}); rebuilding..."
        )

    signal_files = {
        os.path.splitext(name)[0]: os.path.join(signal_dir, name)
        for name in os.listdir(signal_dir)
        if name.endswith(".csv")
    }
    exec_files = {
        os.path.splitext(name)[0]: os.path.join(exec_dir, name)
        for name in os.listdir(exec_dir)
        if name.endswith(".csv")
    }
    codes = sorted(
        code
        for code in exec_files
        if code in signal_files and len(code) == 6 and code.isdigit() and not code.startswith(("39", "88"))
    )
    if not codes:
        raise FileNotFoundError(
            f"No overlapping 15m signal/exec files found under {os.path.abspath(signal_dir)} and {os.path.abspath(exec_dir)}"
        )

    date_keys = req_dates.strftime("%Y%m%d").tolist()
    date_indexer = {key: idx for idx, key in enumerate(date_keys)}
    n_days = len(req_dates)
    n_codes = len(codes)

    early_returns = np.full((early_bars, n_days, n_codes), np.nan, dtype=np.float32)
    exec_1020_vwap = np.full((n_days, n_codes), np.nan, dtype=np.float32)
    exec_1020_twap = np.full((n_days, n_codes), np.nan, dtype=np.float32)
    exec_1450_vwap = np.full((n_days, n_codes), np.nan, dtype=np.float32)
    exec_1450_twap = np.full((n_days, n_codes), np.nan, dtype=np.float32)

    def prepare(df):
        df = df.copy()
        df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("Int64")
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["time"])
        if df.empty:
            return df
        df["time"] = df["time"].astype(np.int64)
        df = df.sort_values("time").copy()
        df["date_key"] = (df["time"] // 1000000).astype(str)
        df["bar_key"] = (df["time"] % 1000000).astype(int)
        return df

    def process_one(code):
        signal_path = signal_files.get(code)
        exec_path = exec_files.get(code)
        if signal_path is None or exec_path is None:
            return code, None

        try:
            sig_df = pd.read_csv(signal_path, encoding="utf-8-sig", usecols=[0, 1, 2, 3, 4, 5, 6])
            exe_df = pd.read_csv(exec_path, encoding="utf-8-sig", usecols=[0, 1, 2, 3, 4, 5, 6])
        except Exception:
            return code, None

        if sig_df.shape[1] < 7 or exe_df.shape[1] < 7:
            return code, None

        sig_df.columns = ["time", "open", "high", "low", "close", "volume", "amount"]
        exe_df.columns = ["time", "open", "high", "low", "close", "volume", "amount"]

        sig_df = prepare(sig_df)
        exe_df = prepare(exe_df)
        if sig_df.empty or exe_df.empty:
            return code, None

        result = {
            "early_returns": np.full((early_bars, n_days), np.nan, dtype=np.float32),
            "exec_1020_vwap": np.full(n_days, np.nan, dtype=np.float32),
            "exec_1020_twap": np.full(n_days, np.nan, dtype=np.float32),
            "exec_1450_vwap": np.full(n_days, np.nan, dtype=np.float32),
            "exec_1450_twap": np.full(n_days, np.nan, dtype=np.float32),
        }

        sig_days = {day: day_df.reset_index(drop=True) for day, day_df in sig_df.groupby("date_key", sort=True)}
        exe_map = {(row.date_key, int(row.bar_key)): row for row in exe_df.itertuples(index=False)}

        for date_key, idx in date_indexer.items():
            day_df = sig_days.get(date_key)
            if day_df is not None and not day_df.empty:
                close_arr = day_df["close"].to_numpy(dtype=np.float64, copy=False)
                valid_rets = []
                for row_idx in range(1, len(close_arr)):
                    prev_close = close_arr[row_idx - 1]
                    curr_close = close_arr[row_idx]
                    if not (np.isfinite(prev_close) and np.isfinite(curr_close) and prev_close > 0 and curr_close > 0):
                        continue
                    valid_rets.append(np.float32(np.log(curr_close / prev_close)))
                    if len(valid_rets) >= early_bars:
                        break
                for early_idx, ret in enumerate(valid_rets):
                    result["early_returns"][early_idx, idx] = ret

            bar_1015 = exe_map.get((date_key, 101500))
            bar_1030 = exe_map.get((date_key, 103000))
            pieces = []
            if bar_1015 is not None:
                pieces.append((bar_1015, 10.0 / 15.0))
            if bar_1030 is not None:
                pieces.append((bar_1030, 5.0 / 15.0))
            if pieces:
                twap_num = 0.0
                twap_den = 0.0
                vwap_num = 0.0
                vwap_den = 0.0
                for bar, ratio in pieces:
                    bt = bar_twap(bar._asdict())
                    bv = bar_vwap(bar._asdict())
                    vol = float(bar.volume) if pd.notna(bar.volume) else np.nan
                    if pd.notna(bt):
                        twap_num += ratio * bt
                        twap_den += ratio
                    if pd.notna(bv):
                        weight = ratio * (vol if pd.notna(vol) and vol > 0 else 1.0)
                        vwap_num += weight * bv
                        vwap_den += weight
                if twap_den > 0:
                    result["exec_1020_twap"][idx] = np.float32(twap_num / twap_den)
                if vwap_den > 0:
                    result["exec_1020_vwap"][idx] = np.float32(vwap_num / vwap_den)

            bar_1500 = exe_map.get((date_key, 150000))
            if bar_1500 is not None:
                result["exec_1450_twap"][idx] = np.float32(bar_twap(bar_1500._asdict()))
                result["exec_1450_vwap"][idx] = np.float32(bar_vwap(bar_1500._asdict()))

        return code, result

    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as executor:
        futures = {executor.submit(process_one, code): (col_idx, code) for col_idx, code in enumerate(codes)}
        for future in as_completed(futures):
            col_idx, _ = futures[future]
            _, result = future.result()
            if result is None:
                continue
            early_returns[:, :, col_idx] = result["early_returns"]
            exec_1020_vwap[:, col_idx] = result["exec_1020_vwap"]
            exec_1020_twap[:, col_idx] = result["exec_1020_twap"]
            exec_1450_vwap[:, col_idx] = result["exec_1450_vwap"]
            exec_1450_twap[:, col_idx] = result["exec_1450_twap"]

    top_hits = np.zeros((n_days, n_codes), dtype=np.uint8)
    low_hits = np.zeros((n_days, n_codes), dtype=np.uint8)
    valid_hits = np.zeros((n_days, n_codes), dtype=np.uint8)

    for early_idx in range(early_bars):
        mat = early_returns[early_idx]
        for day_idx in range(n_days):
            values = mat[day_idx]
            mask = np.isfinite(values)
            n_valid = int(mask.sum())
            if n_valid == 0:
                continue

            valid_hits[day_idx, mask] = np.minimum(valid_hits[day_idx, mask] + 1, 255)
            if n_valid < 20:
                continue

            k = max(int(np.floor(n_valid * 0.05 + 1e-12)), 1)
            valid_idx = np.flatnonzero(mask)
            valid_vals = values[mask]

            top_local = np.argpartition(valid_vals, n_valid - k)[-k:]
            low_local = np.argpartition(valid_vals, k - 1)[:k]
            top_hits[day_idx, valid_idx[top_local]] = np.minimum(top_hits[day_idx, valid_idx[top_local]] + 1, 255)
            low_hits[day_idx, valid_idx[low_local]] = np.minimum(low_hits[day_idx, valid_idx[low_local]] + 1, 255)

    attention_up = np.full((n_days, n_codes), np.nan, dtype=np.float32)
    attention_down = np.full((n_days, n_codes), np.nan, dtype=np.float32)
    valid_mask = valid_hits > 0
    attention_up[valid_mask] = (top_hits[valid_mask] / valid_hits[valid_mask]).astype(np.float32)
    attention_down[valid_mask] = (low_hits[valid_mask] / valid_hits[valid_mask]).astype(np.float32)

    up_df = pd.DataFrame(attention_up, index=req_dates, columns=codes, dtype=np.float32)
    down_df = pd.DataFrame(attention_down, index=req_dates, columns=codes, dtype=np.float32)
    crowding_pct = ((up_df.rank(axis=1, pct=True) + down_df.rank(axis=1, pct=True)) / 2.0).to_numpy(dtype=np.float32)

    np.savez_compressed(
        cache_path,
        codes=np.array(codes, dtype="U6"),
        dates=req_dates.strftime("%Y%m%d").to_numpy(dtype="U8"),
        early_bars=np.array([early_bars], dtype=np.int16),
        attention_up=attention_up.astype(np.float32),
        attention_down=attention_down.astype(np.float32),
        crowding_pct=crowding_pct.astype(np.float32),
        exec_1020_vwap=exec_1020_vwap.astype(np.float32),
        exec_1020_twap=exec_1020_twap.astype(np.float32),
        exec_1450_vwap=exec_1450_vwap.astype(np.float32),
        exec_1450_twap=exec_1450_twap.astype(np.float32),
    )
    print(
        f"intraday attention timing panel cached: {cache_path} | "
        f"early_bars={early_bars} | codes={len(codes)} | dates={len(req_dates)}"
    )

    return IntradayAttentionTimingPanel(
        codes=np.array(codes, dtype="U6"),
        dates=req_dates,
        attention_up=attention_up.astype(np.float32),
        attention_down=attention_down.astype(np.float32),
        crowding_pct=crowding_pct.astype(np.float32),
        exec_1020_vwap=exec_1020_vwap.astype(np.float32),
        exec_1020_twap=exec_1020_twap.astype(np.float32),
        exec_1450_vwap=exec_1450_vwap.astype(np.float32),
        exec_1450_twap=exec_1450_twap.astype(np.float32),
    )


class SmoothedAttentionPanel(AttentionPanel):
    def __init__(self, base_panel, tsmean_window=1):
        self.codes = base_panel.codes
        self.dates = base_panel.dates
        self.code_to_idx = base_panel.code_to_idx
        self.attention_up = base_panel.attention_up
        self.attention_down = base_panel.attention_down
        self.crowding_pct = base_panel.crowding_pct
        self.tsmean_window = max(int(tsmean_window), 1)

        if self.tsmean_window > 1:
            self.smoothed_crowding_pct = self.crowding_pct.rolling(
                window=self.tsmean_window,
                min_periods=1,
            ).mean()
        else:
            self.smoothed_crowding_pct = self.crowding_pct.copy()

        self.trade_crowding_pct = self.smoothed_crowding_pct.shift(1)

    def get_cooldown_active(self, date, code, threshold, cooldown_days):
        date = pd.Timestamp(date)
        if date not in self.smoothed_crowding_pct.index or code not in self.smoothed_crowding_pct.columns:
            return False

        series = self.smoothed_crowding_pct[code]
        loc = self.smoothed_crowding_pct.index.get_indexer([date])[0]
        if loc <= 0:
            return False

        start = max(loc - cooldown_days, 0)
        prev_window = series.iloc[start:loc]
        if len(prev_window) == 0:
            return False
        return bool((prev_window >= threshold).any())


class TsMeanFactorEngine(base.FactorEngine):
    def __init__(self, df_hfq, df_ff3):
        super().__init__(df_hfq, df_ff3)
        self.available_dates = pd.DatetimeIndex(sorted(pd.to_datetime(df_hfq["date"].unique())))

    def get_recent_trading_dates(self, date, window):
        date = pd.Timestamp(date)
        window = max(int(window), 1)
        end_idx = self.available_dates.searchsorted(date, side="right")
        if end_idx <= 0:
            return pd.DatetimeIndex([])
        start_idx = max(0, end_idx - window)
        return self.available_dates[start_idx:end_idx]

    def compute_smoothed_short_ivol_ratio(self, codes, date, factor_cache, tsmean_window):
        if not codes:
            return {}

        tsmean_window = max(int(tsmean_window), 1)
        factor_map = factor_cache.set_index("code")["ivol"].to_dict()
        eval_dates = self.get_recent_trading_dates(date, tsmean_window)
        if len(eval_dates) == 0:
            return {}

        ratio_history = {code: [] for code in codes}
        for eval_date in eval_dates:
            short_ivol_map = self.compute_short_ivol(codes, eval_date)
            for code in codes:
                long_ivol = factor_map.get(code, np.nan)
                short_ivol = short_ivol_map.get(code, np.nan)
                if pd.isna(long_ivol) or long_ivol <= 0:
                    continue
                if pd.isna(short_ivol):
                    continue
                ratio_history[code].append(float(short_ivol) / float(long_ivol))

        return {
            code: float(np.mean(values))
            for code, values in ratio_history.items()
            if values
        }


class ShortIvolBuyFactorEngine(TsMeanFactorEngine):
    def compute_factors(self, valid_codes, date):
        window_start = date - pd.Timedelta(days=int(base.IVOL_WINDOW * 1.6))
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
        if len(common_dates) < base.IVOL_MIN_OBS:
            return pd.DataFrame()

        ret_pivot = ret_pivot.loc[common_dates].sort_index()
        ff3_aligned = ff3.loc[common_dates].sort_index()

        if len(ret_pivot) > base.IVOL_WINDOW:
            ret_pivot = ret_pivot.tail(base.IVOL_WINDOW)
            ff3_aligned = ff3_aligned.tail(base.IVOL_WINDOW)

        t_len = len(ret_pivot)
        valid_count = ret_pivot.notna().sum(axis=0)
        eligible_codes = valid_count[valid_count >= base.IVOL_MIN_OBS].index.tolist()
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

            short_ivol_arr = np.full(len(eligible_codes), np.nan)
            short_len = min(base.SHORT_IVOL_WINDOW, resid.shape[0])
            if short_len >= base.SHORT_IVOL_WINDOW:
                x_short = x[-short_len:]
                y_short = y_raw[-short_len:]
                try:
                    beta_short, _, _, _ = np.linalg.lstsq(x_short, y_short, rcond=None)
                    resid_short = y_short - x_short @ beta_short
                    short_ivol_arr = np.std(resid_short, axis=0, ddof=1)
                except np.linalg.LinAlgError:
                    short_ivol_arr = np.full(len(eligible_codes), np.nan)

            return pd.DataFrame(
                {
                    "code": eligible_codes,
                    "ivol": ivol_arr,
                    "delta_ivol": delta_ivol_arr,
                    "cbmom": cbmom_arr_all,
                    "short_ivol_buy": short_ivol_arr,
                }
            )

        pattern_map = {}
        for j in range(y_raw.shape[1]):
            key = nan_mask[:, j].tobytes()
            pattern_map.setdefault(key, []).append(j)

        ivol_arr = np.full(len(eligible_codes), np.nan)
        delta_ivol_arr = np.full(len(eligible_codes), np.nan)
        short_ivol_arr = np.full(len(eligible_codes), np.nan)

        for key_bytes, col_indices in pattern_map.items():
            row_mask = ~nan_mask[:, col_indices[0]]
            n_valid = row_mask.sum()
            if n_valid < base.IVOL_MIN_OBS:
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

            short_len = min(base.SHORT_IVOL_WINDOW, n_valid)
            if short_len < base.SHORT_IVOL_WINDOW:
                continue

            x_short = x_sub[-short_len:]
            y_short = y_sub[-short_len:]
            try:
                beta_short, _, _, _ = np.linalg.lstsq(x_short, y_short, rcond=None)
            except np.linalg.LinAlgError:
                continue

            resid_short = y_short - x_short @ beta_short
            short_ivol_arr[col_indices] = np.std(resid_short, axis=0, ddof=1)

        computed_mask = ~np.isnan(ivol_arr)
        return pd.DataFrame(
            {
                "code": np.array(eligible_codes)[computed_mask],
                "ivol": ivol_arr[computed_mask],
                "delta_ivol": delta_ivol_arr[computed_mask],
                "cbmom": cbmom_arr_all[computed_mask],
                "short_ivol_buy": short_ivol_arr[computed_mask],
            }
        )


class ApproxRAFactorEngine(TsMeanFactorEngine, ApproxRADensityMixin):
    def compute_factors(self, valid_codes, date):
        window_start = date - pd.Timedelta(days=int(base.IVOL_WINDOW * 1.6))
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
        if len(common_dates) < base.IVOL_MIN_OBS:
            return pd.DataFrame()

        ret_pivot = ret_pivot.loc[common_dates].sort_index()
        ff3_aligned = ff3.loc[common_dates].sort_index()

        if len(ret_pivot) > base.IVOL_WINDOW:
            ret_pivot = ret_pivot.tail(base.IVOL_WINDOW)
            ff3_aligned = ff3_aligned.tail(base.IVOL_WINDOW)

        t_len = len(ret_pivot)
        valid_count = ret_pivot.notna().sum(axis=0)
        eligible_codes = valid_count[valid_count >= base.IVOL_MIN_OBS].index.tolist()
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
            ra_arr = self._compute_approx_ra_from_resid(resid)

            return pd.DataFrame(
                {
                    "code": eligible_codes,
                    "ivol": ivol_arr,
                    "delta_ivol": delta_ivol_arr,
                    "cbmom": cbmom_arr_all,
                    "ra_approx": ra_arr,
                }
            )

        pattern_map = {}
        for j in range(y_raw.shape[1]):
            key = nan_mask[:, j].tobytes()
            pattern_map.setdefault(key, []).append(j)

        ivol_arr = np.full(len(eligible_codes), np.nan)
        delta_ivol_arr = np.full(len(eligible_codes), np.nan)
        ra_arr = np.full(len(eligible_codes), np.nan)

        for key_bytes, col_indices in pattern_map.items():
            row_mask = ~nan_mask[:, col_indices[0]]
            n_valid = row_mask.sum()
            if n_valid < base.IVOL_MIN_OBS:
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
            ra_arr[col_indices] = self._compute_approx_ra_from_resid(resid_sub)

        computed_mask = ~np.isnan(ivol_arr)
        return pd.DataFrame(
            {
                "code": np.array(eligible_codes)[computed_mask],
                "ivol": ivol_arr[computed_mask],
                "delta_ivol": delta_ivol_arr[computed_mask],
                "cbmom": cbmom_arr_all[computed_mask],
                "ra_approx": ra_arr[computed_mask],
            }
        )


class TechnicalCompositeFactorEngine(TsMeanFactorEngine):
    TECH_SIGNAL_NAMES = (
        "ma_1_9",
        "ma_1_12",
        "ma_2_9",
        "ma_2_12",
        "ma_3_9",
        "ma_3_12",
        "mom_9",
        "mom_12",
        "vol_1_9",
        "vol_1_12",
        "vol_2_9",
        "vol_2_12",
        "vol_3_9",
        "vol_3_12",
    )
    TECH_SMOOTH_MONTHS = 60
    TECH_MIN_TRAIN_MONTHS = 24
    TECH_MIN_STOCKS = 200

    def __init__(self, df_hfq, df_ff3):
        super().__init__(df_hfq, df_ff3)
        self._tech_snapshot_scores = self._build_monthly_technical_scores()
        self._tech_snapshot_dates = pd.DatetimeIndex(self._tech_snapshot_scores.index)

    @staticmethod
    def _binary_signal(left_df, right_df):
        valid = left_df.notna() & right_df.notna()
        out = pd.DataFrame(np.nan, index=left_df.index, columns=left_df.columns, dtype=np.float32)
        if valid.to_numpy().any():
            out[valid] = (left_df[valid] >= right_df[valid]).astype(np.float32)
        return out

    def _build_monthly_technical_scores(self):
        monthly = self.df_hfq[["date", "code", "close", "turnover"]].dropna(subset=["date", "code", "close"]).copy()
        if monthly.empty:
            return pd.DataFrame()

        monthly["month"] = monthly["date"].dt.to_period("M")
        monthly = (
            monthly.groupby(["code", "month"], sort=True)
            .agg(
                snapshot_date=("date", "max"),
                close=("close", "last"),
                turnover=("turnover", "sum"),
            )
            .reset_index()
        )
        if monthly.empty:
            return pd.DataFrame()

        month_snapshot_dates = (
            monthly.groupby("month", sort=True)["snapshot_date"]
            .max()
            .sort_index()
        )

        close_wide = monthly.pivot(index="month", columns="code", values="close").sort_index()
        close_wide = close_wide.sort_index(axis=1)
        turnover_wide = (
            monthly.pivot(index="month", columns="code", values="turnover")
            .reindex(index=close_wide.index, columns=close_wide.columns)
            .sort_index()
        )

        signal_map = {}
        for short_len in (1, 2, 3):
            for long_len in (9, 12):
                ma_short = close_wide.rolling(short_len, min_periods=short_len).mean()
                ma_long = close_wide.rolling(long_len, min_periods=long_len).mean()
                signal_map[f"ma_{short_len}_{long_len}"] = self._binary_signal(ma_short, ma_long)

        for lag in (9, 12):
            signal_map[f"mom_{lag}"] = self._binary_signal(close_wide, close_wide.shift(lag))

        month_diff = close_wide.diff()
        direction = pd.DataFrame(np.nan, index=close_wide.index, columns=close_wide.columns, dtype=np.float32)
        valid_diff = month_diff.notna()
        if valid_diff.to_numpy().any():
            direction[valid_diff] = np.where(month_diff[valid_diff] >= 0, 1.0, -1.0).astype(np.float32)

        signed_turnover = turnover_wide * direction
        obv_proxy = signed_turnover.fillna(0.0).cumsum()
        for short_len in (1, 2, 3):
            for long_len in (9, 12):
                vol_short = obv_proxy.rolling(short_len, min_periods=short_len).mean()
                vol_long = obv_proxy.rolling(long_len, min_periods=long_len).mean()
                signal_map[f"vol_{short_len}_{long_len}"] = self._binary_signal(vol_short, vol_long)

        forward_ret = close_wide.shift(-1) / close_wide - 1.0
        codes = close_wide.columns
        coef_history = []
        prediction_rows = []

        for snapshot_month in close_wide.index:
            current_features = pd.DataFrame(
                {name: signal_map[name].loc[snapshot_month] for name in self.TECH_SIGNAL_NAMES},
                index=codes,
            )

            pred_scores = pd.Series(np.nan, index=codes, dtype=np.float32)
            if len(coef_history) >= self.TECH_MIN_TRAIN_MONTHS:
                coef_window = np.asarray(coef_history[-self.TECH_SMOOTH_MONTHS :], dtype=np.float64)
                coef_smooth = np.nanmean(coef_window, axis=0)
                valid_pred = current_features.notna().all(axis=1)
                if valid_pred.any():
                    x_pred = current_features.loc[valid_pred, self.TECH_SIGNAL_NAMES].to_numpy(dtype=np.float64)
                    pred_scores.loc[valid_pred] = (
                        coef_smooth[0] + x_pred @ coef_smooth[1:]
                    ).astype(np.float32)
            prediction_rows.append(pred_scores.rename(pd.Timestamp(month_snapshot_dates.loc[snapshot_month])))

            train_df = current_features.copy()
            train_df["forward_ret"] = forward_ret.loc[snapshot_month]
            train_df = train_df.dropna(subset=["forward_ret"])
            train_df = train_df.dropna(subset=list(self.TECH_SIGNAL_NAMES))
            if len(train_df) < self.TECH_MIN_STOCKS:
                continue

            x_train = np.column_stack(
                [
                    np.ones(len(train_df), dtype=np.float64),
                    train_df.loc[:, self.TECH_SIGNAL_NAMES].to_numpy(dtype=np.float64),
                ]
            )
            y_train = train_df["forward_ret"].to_numpy(dtype=np.float64)
            try:
                beta, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
            except np.linalg.LinAlgError:
                continue
            coef_history.append(beta.astype(np.float64))

        if not prediction_rows:
            return pd.DataFrame()

        score_df = pd.DataFrame(prediction_rows, columns=codes, dtype=np.float32)
        score_df.index = pd.DatetimeIndex(score_df.index)
        score_df.sort_index(inplace=True)
        score_df.dropna(axis=0, how="all", inplace=True)
        return score_df

    def _get_snapshot_scores(self, date):
        if self._tech_snapshot_scores.empty:
            return None
        ts = pd.Timestamp(date).normalize()
        pos = self._tech_snapshot_dates.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None
        snapshot_date = self._tech_snapshot_dates[pos]
        return self._tech_snapshot_scores.loc[snapshot_date]

    def compute_factors(self, valid_codes, date):
        base_df = super().compute_factors(valid_codes, date)
        if base_df is None or len(base_df) == 0:
            return base_df

        snapshot_scores = self._get_snapshot_scores(date)
        if snapshot_scores is None:
            base_df["tech_composite"] = np.nan
            return base_df

        score_map = snapshot_scores.to_dict()
        base_df["tech_composite"] = base_df["code"].map(score_map)
        return base_df


class TsMeanAttentionTriggerEngine(AttentionTriggerEngine):
    def __init__(
        self,
        factor_engine,
        backup_multiple=3,
        backup_extra=20,
        short_ivol_tsmean_window=1,
    ):
        super().__init__(factor_engine, backup_multiple=backup_multiple, backup_extra=backup_extra)
        self.short_ivol_tsmean_window = max(int(short_ivol_tsmean_window), 1)

    def check_sell_triggers(self, holding_codes, date, factor_cache):
        sell_codes = set()
        sell_reasons = {}

        if factor_cache is None or len(factor_cache) == 0:
            return sell_codes, sell_reasons

        for code in holding_codes:
            code_data = factor_cache[factor_cache["code"] == code]
            if len(code_data) == 0:
                sell_codes.add(code)
                sell_reasons[code] = "data_missing"
                continue

            cbmom_score = code_data["cbmom"].values[0]
            if cbmom_score < 0:
                sell_codes.add(code)
                sell_reasons[code] = f"CBMOM_negative(score={cbmom_score:.4f})"
                continue

        remaining = [c for c in holding_codes if c not in sell_codes]
        if remaining:
            ratio_map = self.factor_engine.compute_smoothed_short_ivol_ratio(
                remaining,
                date,
                factor_cache,
                tsmean_window=self.short_ivol_tsmean_window,
            )
            current_short_ivol = self.factor_engine.compute_short_ivol(remaining, date)

            for code in remaining:
                ratio = ratio_map.get(code, np.nan)
                if pd.isna(ratio):
                    continue

                code_data = factor_cache[factor_cache["code"] == code]
                if len(code_data) == 0:
                    continue

                long_avg_ivol = code_data["ivol"].values[0]
                if long_avg_ivol > 0 and ratio > base.IVOL_SPIKE_MULT:
                    sell_codes.add(code)
                    short_now = current_short_ivol.get(code, np.nan)
                    if pd.notna(short_now):
                        sell_reasons[code] = (
                            f"IVOL_spike_tsmean({self.short_ivol_tsmean_window}d_ratio={ratio:.3f}, "
                            f"5d={short_now:.4f}, long={long_avg_ivol:.4f})"
                        )
                    else:
                        sell_reasons[code] = (
                            f"IVOL_spike_tsmean({self.short_ivol_tsmean_window}d_ratio={ratio:.3f}, "
                            f"long={long_avg_ivol:.4f})"
                        )

        return sell_codes, sell_reasons


class ShortIvolBuyAttentionTriggerEngine(TsMeanAttentionTriggerEngine):
    def _get_eligible_pool(self, factor_cache):
        if factor_cache is None or len(factor_cache) == 0:
            return pd.DataFrame()

        eligible = factor_cache.dropna(subset=["short_ivol_buy"]).copy()
        if len(eligible) == 0:
            return pd.DataFrame()

        cutoff_low = max(int(len(eligible) * base.LONG_PCT), 1)
        low_short_ivol = eligible.sort_values("short_ivol_buy").head(cutoff_low)
        converging = low_short_ivol[low_short_ivol["delta_ivol"] < 0]
        positive = converging[converging["cbmom"] > 0]
        return positive

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        if n_needed <= 0:
            return []

        eligible_pool = self._get_eligible_pool(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(["delta_ivol", "short_ivol_buy"], ascending=[True, True])
        backup_n = max(n_needed, n_needed * self.backup_multiple, n_needed + self.backup_extra)
        return candidates.head(backup_n)["code"].tolist()


class LongIvolShortFilterAttentionTriggerEngine(TsMeanAttentionTriggerEngine):
    def __init__(
        self,
        factor_engine,
        backup_multiple=3,
        backup_extra=20,
        short_ivol_tsmean_window=1,
        buy_short_ivol_filter_pct=DEFAULT_BUY_SHORT_IVOL_FILTER_PCT,
    ):
        super().__init__(
            factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            short_ivol_tsmean_window=short_ivol_tsmean_window,
        )
        self.buy_short_ivol_filter_pct = float(buy_short_ivol_filter_pct)

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        if n_needed <= 0:
            return []

        eligible_pool = super()._get_eligible_pool(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(["delta_ivol", "ivol"], ascending=[True, True])
        backup_n = max(n_needed, n_needed * self.backup_multiple, n_needed + self.backup_extra)

        with_short = candidates.dropna(subset=["short_ivol_buy"]).copy()
        if len(with_short) == 0:
            return candidates.head(backup_n)["code"].tolist()

        filter_pct = min(max(self.buy_short_ivol_filter_pct, 0.05), 1.0)
        short_cutoff = max(int(np.ceil(len(with_short) * filter_pct)), backup_n)
        preferred_codes = set(
            with_short.sort_values(["short_ivol_buy", "delta_ivol", "ivol"], ascending=[True, True, True])
            .head(short_cutoff)["code"]
            .tolist()
        )

        primary = candidates[candidates["code"].isin(preferred_codes)].sort_values(
            ["delta_ivol", "ivol"],
            ascending=[True, True],
        )
        if len(primary) >= backup_n:
            return primary.head(backup_n)["code"].tolist()

        fallback = candidates[~candidates["code"].isin(primary["code"])]
        merged = pd.concat([primary, fallback], ignore_index=False)
        return merged.head(backup_n)["code"].tolist()


class AttentionBuyTriggerEngine(TsMeanAttentionTriggerEngine):
    def _get_eligible_pool(self, factor_cache):
        if factor_cache is None or len(factor_cache) == 0:
            return pd.DataFrame()

        eligible = factor_cache.copy()
        eligible = eligible[eligible["delta_ivol"] < 0]
        eligible = eligible[eligible["cbmom"] > 0]
        return eligible

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        if n_needed <= 0:
            return []

        eligible_pool = self._get_eligible_pool(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(["delta_ivol"], ascending=[True])
        backup_n = max(n_needed, n_needed * self.backup_multiple, n_needed + self.backup_extra)
        return candidates.head(backup_n)["code"].tolist()


class ApproxRABuyAttentionTriggerEngine(TsMeanAttentionTriggerEngine):
    def _get_eligible_pool(self, factor_cache):
        if factor_cache is None or len(factor_cache) == 0:
            return pd.DataFrame()

        eligible = factor_cache.dropna(subset=["ra_approx"]).copy()
        if len(eligible) == 0:
            return pd.DataFrame()

        cutoff_low = max(int(len(eligible) * base.LONG_PCT), 1)
        low_ra = eligible.sort_values("ra_approx").head(cutoff_low)
        converging = low_ra[low_ra["delta_ivol"] < 0]
        positive = converging[converging["cbmom"] > 0]
        return positive

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        if n_needed <= 0:
            return []

        eligible_pool = self._get_eligible_pool(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(["delta_ivol", "ra_approx"], ascending=[True, True])
        backup_n = max(n_needed, n_needed * self.backup_multiple, n_needed + self.backup_extra)
        return candidates.head(backup_n)["code"].tolist()


class TechnicalCompositeBuyAttentionTriggerEngine(TsMeanAttentionTriggerEngine):
    def _get_eligible_pool(self, factor_cache):
        if factor_cache is None or len(factor_cache) == 0:
            return pd.DataFrame()

        eligible = factor_cache.dropna(subset=["tech_composite"]).copy()
        if len(eligible) == 0:
            return pd.DataFrame()

        cutoff_high = max(int(len(eligible) * base.LONG_PCT), 1)
        high_tech = eligible.sort_values("tech_composite", ascending=False).head(cutoff_high)
        converging = high_tech[high_tech["delta_ivol"] < 0]
        positive = converging[converging["cbmom"] > 0]
        return positive

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        if n_needed <= 0:
            return []

        eligible_pool = self._get_eligible_pool(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(
            ["tech_composite", "delta_ivol", "ivol"],
            ascending=[False, True, True],
        )
        backup_n = max(n_needed, n_needed * self.backup_multiple, n_needed + self.backup_extra)
        return candidates.head(backup_n)["code"].tolist()


class ApproxRAOnlyTriggerEngine:
    def __init__(self, factor_engine, backup_multiple=3, backup_extra=20):
        self.factor_engine = factor_engine
        self.backup_multiple = max(int(backup_multiple), 1)
        self.backup_extra = max(int(backup_extra), 0)

    @staticmethod
    def _eligible_codes_by_ra(factor_cache):
        if factor_cache is None or len(factor_cache) == 0:
            return pd.DataFrame()

        eligible = factor_cache.dropna(subset=["ra_approx"]).copy()
        if len(eligible) == 0:
            return pd.DataFrame()

        cutoff_low = max(int(len(eligible) * base.LONG_PCT), 1)
        return eligible.sort_values("ra_approx").head(cutoff_low)

    def check_sell_triggers(self, holding_codes, date, factor_cache):
        sell_codes = set()
        sell_reasons = {}

        if not holding_codes:
            return sell_codes, sell_reasons

        eligible_pool = self._eligible_codes_by_ra(factor_cache)
        if len(eligible_pool) == 0:
            for code in holding_codes:
                sell_codes.add(code)
                sell_reasons[code] = "RA_only_data_missing"
            return sell_codes, sell_reasons

        keep_codes = set(eligible_pool["code"].astype(str).tolist())
        ra_map = factor_cache.set_index("code")["ra_approx"].to_dict() if "ra_approx" in factor_cache.columns else {}
        for code in holding_codes:
            ra_val = ra_map.get(code, np.nan)
            if code not in keep_codes:
                sell_codes.add(code)
                if pd.notna(ra_val):
                    sell_reasons[code] = f"RA_only_exit(ra={float(ra_val):.6f})"
                else:
                    sell_reasons[code] = "RA_only_exit(ra=nan)"

        return sell_codes, sell_reasons

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        if n_needed <= 0:
            return []

        eligible_pool = self._eligible_codes_by_ra(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(["ra_approx"], ascending=[True])
        backup_n = max(n_needed, n_needed * self.backup_multiple, n_needed + self.backup_extra)
        return candidates.head(backup_n)["code"].tolist()


class ShortIvolCrossSectionalPreprocessor:
    def __init__(self, df_raw):
        self.df_raw = df_raw
        self.raw_date_indices = df_raw.groupby("date").indices
        self._size_proxy_cache = {}
        self.industry_map = self._load_industry_map()

    @staticmethod
    def _normalize_code(series):
        return (
            series.astype(str)
            .str.replace(".0", "", regex=False)
            .str.strip()
            .str.zfill(6)
        )

    def _load_industry_map(self):
        controller_path = os.path.join(BASE_DIR, "stock_sw_industry_controller.csv")
        if os.path.exists(controller_path):
            try:
                df = pd.read_csv(controller_path, encoding="utf-8-sig")
                if {"股票代码", "申万三级行业"}.issubset(df.columns):
                    df = df[["股票代码", "申万三级行业"]].copy()
                    df["code"] = self._normalize_code(df["股票代码"])
                    df["industry"] = df["申万三级行业"].astype(str).str.strip()
                    df = df[(df["industry"] != "") & (df["industry"] != "缺失") & (df["industry"] != "nan")]
                    return df.drop_duplicates("code").set_index("code")["industry"].to_dict()
            except Exception:
                pass

        fallback_path = os.path.join(BASE_DIR, "sw_industry_map.csv")
        if os.path.exists(fallback_path):
            try:
                df = pd.read_csv(fallback_path, encoding="utf-8-sig")
                if {"code", "industry"}.issubset(df.columns):
                    df = df[["code", "industry"]].copy()
                    df["code"] = self._normalize_code(df["code"])
                    df["industry"] = df["industry"].astype(str).str.strip()
                    df = df[(df["industry"] != "") & (df["industry"] != "nan")]
                    return df.drop_duplicates("code").set_index("code")["industry"].to_dict()
            except Exception:
                pass
        return {}

    def get_size_proxy_log(self, date):
        key = pd.Timestamp(date).normalize()
        cached = self._size_proxy_cache.get(key)
        if cached is not None:
            return cached

        idx = self.raw_date_indices.get(key)
        if idx is None:
            series = pd.Series(dtype=np.float64)
            self._size_proxy_cache[key] = series
            return series

        day_df = self.df_raw.iloc[idx][["code", "close", "turnover"]].copy()
        close = pd.to_numeric(day_df["close"], errors="coerce")
        turnover_frac = pd.to_numeric(day_df["turnover"], errors="coerce") / 100.0
        size_proxy = np.where(
            np.isfinite(close) & np.isfinite(turnover_frac) & (close > 0) & (turnover_frac > 0),
            np.log(close) - np.log(turnover_frac),
            np.nan,
        )
        series = pd.Series(size_proxy, index=day_df["code"].astype(str).values, dtype=np.float64)
        self._size_proxy_cache[key] = series
        return series

    @staticmethod
    def _zscore(series):
        std = float(series.std(ddof=1))
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=series.index, dtype=np.float64)
        return (series - float(series.mean())) / std

    def preprocess(self, factor_cache, date, factor_col="short_ivol_buy"):
        if factor_cache is None or len(factor_cache) == 0:
            return pd.DataFrame() if factor_cache is None else factor_cache.copy()

        out = factor_cache.copy()
        raw = pd.to_numeric(out[factor_col], errors="coerce")
        finite_raw = raw[np.isfinite(raw)]
        neu_col = f"{factor_col}_neu"
        winsor_col = f"{factor_col}_winsor"
        z_col = f"{factor_col}_z"
        out[winsor_col] = np.nan
        out[z_col] = np.nan
        out[neu_col] = np.nan
        if len(finite_raw) < 20:
            return out

        lower = float(finite_raw.quantile(DEFAULT_SHORT_IVOL_WINSOR_PCT))
        upper = float(finite_raw.quantile(1.0 - DEFAULT_SHORT_IVOL_WINSOR_PCT))
        winsor = raw.clip(lower=lower, upper=upper)
        out[winsor_col] = winsor

        valid_winsor = winsor[np.isfinite(winsor)]
        if len(valid_winsor) < 20:
            return out

        zscore = pd.Series(np.nan, index=out.index, dtype=np.float64)
        zscore.loc[valid_winsor.index] = self._zscore(valid_winsor)
        out[z_col] = zscore

        size_proxy = out["code"].map(self.get_size_proxy_log(date))
        industry = out["code"].map(self.industry_map)
        out["size_log_proxy"] = size_proxy
        out["industry_bucket"] = industry

        reg_df = out[[z_col, "size_log_proxy", "industry_bucket"]].copy()
        reg_df = reg_df[
            np.isfinite(reg_df[z_col])
            & np.isfinite(reg_df["size_log_proxy"])
            & reg_df["industry_bucket"].notna()
        ]
        if len(reg_df) < 20:
            return out

        reg_df["size_z"] = self._zscore(reg_df["size_log_proxy"])
        counts = reg_df["industry_bucket"].value_counts()
        reg_df["industry_bucket"] = reg_df["industry_bucket"].where(
            reg_df["industry_bucket"].map(counts) >= 5,
            "__OTHER__",
        )
        dummies = pd.get_dummies(reg_df["industry_bucket"], prefix="ind", drop_first=True, dtype=float)
        x_parts = [np.ones(len(reg_df), dtype=np.float64), reg_df["size_z"].to_numpy(dtype=np.float64, copy=False)]
        if not dummies.empty:
            x_parts.append(dummies.to_numpy(dtype=np.float64, copy=False))
        x = np.column_stack(x_parts)
        y = reg_df[z_col].to_numpy(dtype=np.float64, copy=False)
        try:
            beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            return out

        resid = y - x @ beta
        resid_series = pd.Series(resid, index=reg_df.index, dtype=np.float64)
        out.loc[reg_df.index, neu_col] = self._zscore(resid_series)
        return out


class PreprocessedShortIvolBuyAttentionTriggerEngine(TsMeanAttentionTriggerEngine):
    def __init__(
        self,
        factor_engine,
        preprocessor,
        backup_multiple=3,
        backup_extra=20,
        short_ivol_tsmean_window=1,
    ):
        super().__init__(
            factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            short_ivol_tsmean_window=short_ivol_tsmean_window,
        )
        self.preprocessor = preprocessor
        self.current_date = None

    def _build_processed_pool(self, factor_cache):
        if self.current_date is None:
            return pd.DataFrame()
        processed = self.preprocessor.preprocess(factor_cache, self.current_date, factor_col="short_ivol_buy")
        if processed is None or len(processed) == 0:
            return pd.DataFrame()

        eligible = processed.dropna(subset=["short_ivol_buy_neu"]).copy()
        if len(eligible) == 0:
            return pd.DataFrame()

        cutoff_low = max(int(len(eligible) * base.LONG_PCT), 1)
        low_short_ivol = eligible.sort_values("short_ivol_buy_neu").head(cutoff_low)
        converging = low_short_ivol[low_short_ivol["delta_ivol"] < 0]
        positive = converging[converging["cbmom"] > 0]
        return positive

    def get_buy_candidates(self, current_holdings, n_needed, factor_cache):
        if n_needed <= 0:
            return []

        eligible_pool = self._build_processed_pool(factor_cache)
        if len(eligible_pool) == 0:
            return []

        candidates = eligible_pool[~eligible_pool["code"].isin(current_holdings)].copy()
        if len(candidates) == 0:
            return []

        candidates = candidates.sort_values(["short_ivol_buy_neu", "delta_ivol"], ascending=[True, True])
        backup_n = max(n_needed, n_needed * self.backup_multiple, n_needed + self.backup_extra)
        return candidates.head(backup_n)["code"].tolist()


class TsMeanDailyCappedAttentionPenaltyBacktestEngine(DailyCappedAttentionPenaltyBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        max_new_buys_per_day=DEFAULT_MAX_NEW_BUYS,
        backup_multiple=3,
        backup_extra=20,
        attention_tsmean_window=DEFAULT_ATTENTION_TSMEAN_WINDOW,
        short_ivol_tsmean_window=DEFAULT_SHORT_IVOL_TSMEAN_WINDOW,
    ):
        self.attention_tsmean_window = max(int(attention_tsmean_window), 1)
        self.short_ivol_tsmean_window = max(int(short_ivol_tsmean_window), 1)
        smoothed_panel = SmoothedAttentionPanel(attention_panel, tsmean_window=self.attention_tsmean_window)

        super().__init__(
            data_loader,
            smoothed_panel,
            penalty_weight=penalty_weight,
            max_new_buys_per_day=max_new_buys_per_day,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )

        self.attention_panel_raw = attention_panel
        self.attention_panel = smoothed_panel
        self.factor_engine = TsMeanFactorEngine(data_loader.df_hfq, data_loader.df_ff3)
        self.trigger_engine = TsMeanAttentionTriggerEngine(
            self.factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            short_ivol_tsmean_window=self.short_ivol_tsmean_window,
        )

        suffix = "all" if self.max_new_buys_per_day is None else str(self.max_new_buys_per_day)
        self.variant_name = (
            f"attention_penalty_w012_tsmean_att{self.attention_tsmean_window}"
            f"_ivol{self.short_ivol_tsmean_window}_maxbuy{suffix}"
        )


class AttentionBuyDailyCappedAttentionPenaltyBacktestEngine(DailyCappedAttentionPenaltyBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        max_new_buys_per_day=DEFAULT_MAX_NEW_BUYS,
        backup_multiple=3,
        backup_extra=20,
        attention_tsmean_window=DEFAULT_ATTENTION_TSMEAN_WINDOW,
        short_ivol_tsmean_window=DEFAULT_SHORT_IVOL_TSMEAN_WINDOW,
    ):
        self.attention_tsmean_window = max(int(attention_tsmean_window), 1)
        self.short_ivol_tsmean_window = max(int(short_ivol_tsmean_window), 1)
        smoothed_panel = attention_panel
        if self.attention_tsmean_window > 1:
            smoothed_panel = SmoothedAttentionPanel(attention_panel, tsmean_window=self.attention_tsmean_window)

        super().__init__(
            data_loader,
            smoothed_panel,
            penalty_weight=penalty_weight,
            max_new_buys_per_day=max_new_buys_per_day,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )

        self.attention_panel_raw = attention_panel
        self.attention_panel = smoothed_panel
        self.factor_engine = TsMeanFactorEngine(data_loader.df_hfq, data_loader.df_ff3)
        self.trigger_engine = AttentionBuyTriggerEngine(
            self.factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            short_ivol_tsmean_window=self.short_ivol_tsmean_window,
        )

        suffix = "all" if self.max_new_buys_per_day is None else str(self.max_new_buys_per_day)
        parts = ["attention_penalty_w012", "buyattention"]
        if self.attention_tsmean_window > 1:
            parts.append(f"atttsmean{self.attention_tsmean_window}")
        if self.short_ivol_tsmean_window > 1:
            parts.append(f"sellivoltsmean{self.short_ivol_tsmean_window}")
        parts.append(f"maxbuy{suffix}")
        self.variant_name = "_".join(parts)


class ApproxRABuyDailyCappedAttentionPenaltyBacktestEngine(DailyCappedAttentionPenaltyBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        max_new_buys_per_day=DEFAULT_MAX_NEW_BUYS,
        backup_multiple=3,
        backup_extra=20,
        attention_tsmean_window=DEFAULT_ATTENTION_TSMEAN_WINDOW,
        short_ivol_tsmean_window=DEFAULT_SHORT_IVOL_TSMEAN_WINDOW,
    ):
        self.attention_tsmean_window = max(int(attention_tsmean_window), 1)
        self.short_ivol_tsmean_window = max(int(short_ivol_tsmean_window), 1)
        smoothed_panel = attention_panel
        if self.attention_tsmean_window > 1:
            smoothed_panel = SmoothedAttentionPanel(attention_panel, tsmean_window=self.attention_tsmean_window)

        super().__init__(
            data_loader,
            smoothed_panel,
            penalty_weight=penalty_weight,
            max_new_buys_per_day=max_new_buys_per_day,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )

        self.attention_panel_raw = attention_panel
        self.attention_panel = smoothed_panel
        self.factor_engine = ApproxRAFactorEngine(data_loader.df_hfq, data_loader.df_ff3)
        self.trigger_engine = ApproxRABuyAttentionTriggerEngine(
            self.factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            short_ivol_tsmean_window=self.short_ivol_tsmean_window,
        )

        suffix = "all" if self.max_new_buys_per_day is None else str(self.max_new_buys_per_day)
        parts = ["attention_penalty_w012", "buyraapprox"]
        if self.attention_tsmean_window > 1:
            parts.append(f"atttsmean{self.attention_tsmean_window}")
        if self.short_ivol_tsmean_window > 1:
            parts.append(f"sellivoltsmean{self.short_ivol_tsmean_window}")
        parts.append("kde12m")
        parts.append(f"maxbuy{suffix}")
        self.variant_name = "_".join(parts)


class TechnicalCompositeBuyDailyCappedAttentionPenaltyBacktestEngine(DailyCappedAttentionPenaltyBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        max_new_buys_per_day=DEFAULT_MAX_NEW_BUYS,
        backup_multiple=3,
        backup_extra=20,
        attention_tsmean_window=DEFAULT_ATTENTION_TSMEAN_WINDOW,
        short_ivol_tsmean_window=DEFAULT_SHORT_IVOL_TSMEAN_WINDOW,
    ):
        self.attention_tsmean_window = max(int(attention_tsmean_window), 1)
        self.short_ivol_tsmean_window = max(int(short_ivol_tsmean_window), 1)
        smoothed_panel = attention_panel
        if self.attention_tsmean_window > 1:
            smoothed_panel = SmoothedAttentionPanel(attention_panel, tsmean_window=self.attention_tsmean_window)

        super().__init__(
            data_loader,
            smoothed_panel,
            penalty_weight=penalty_weight,
            max_new_buys_per_day=max_new_buys_per_day,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )

        self.attention_panel_raw = attention_panel
        self.attention_panel = smoothed_panel
        self.factor_engine = TechnicalCompositeFactorEngine(data_loader.df_hfq, data_loader.df_ff3)
        self.trigger_engine = TechnicalCompositeBuyAttentionTriggerEngine(
            self.factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            short_ivol_tsmean_window=self.short_ivol_tsmean_window,
        )

        suffix = "all" if self.max_new_buys_per_day is None else str(self.max_new_buys_per_day)
        parts = ["attention_penalty_w012", "buytechcomposite"]
        if self.attention_tsmean_window > 1:
            parts.append(f"atttsmean{self.attention_tsmean_window}")
        if self.short_ivol_tsmean_window > 1:
            parts.append(f"sellivoltsmean{self.short_ivol_tsmean_window}")
        parts.append("sols60m")
        parts.append(f"maxbuy{suffix}")
        self.variant_name = "_".join(parts)


class ApproxRAOnlyDailyCappedBacktestEngine(DailyCappedAttentionPenaltyBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        max_new_buys_per_day=DEFAULT_MAX_NEW_BUYS,
        backup_multiple=3,
        backup_extra=20,
        attention_tsmean_window=DEFAULT_ATTENTION_TSMEAN_WINDOW,
        short_ivol_tsmean_window=DEFAULT_SHORT_IVOL_TSMEAN_WINDOW,
    ):
        self.attention_tsmean_window = max(int(attention_tsmean_window), 1)
        self.short_ivol_tsmean_window = max(int(short_ivol_tsmean_window), 1)
        super().__init__(
            data_loader,
            attention_panel,
            penalty_weight=penalty_weight,
            max_new_buys_per_day=max_new_buys_per_day,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )

        self.factor_engine = ApproxRAFactorEngine(data_loader.df_hfq, data_loader.df_ff3)
        self.trigger_engine = ApproxRAOnlyTriggerEngine(
            self.factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )
        suffix = "all" if self.max_new_buys_per_day is None else str(self.max_new_buys_per_day)
        self.variant_name = f"ra_only_kde12m_maxbuy{suffix}"

    def _select_buys(self, date, candidate_codes, n_slots):
        effective_slots = n_slots
        if self.max_new_buys_per_day is not None and self.max_new_buys_per_day > 0:
            effective_slots = min(n_slots, self.max_new_buys_per_day)
        if not candidate_codes or effective_slots <= 0:
            return [], {}, self._empty_stats(len(candidate_codes), effective_slots)

        selected = list(candidate_codes[:effective_slots])
        meta_by_code = {
            code: {
                "base_rank": int(idx + 1),
                "crowding_pct": np.nan,
            }
            for idx, code in enumerate(candidate_codes)
        }
        stats = {
            "candidate_count": int(len(candidate_codes)),
            "slot_count": int(effective_slots),
            "selected_count": int(len(selected)),
            "avg_selected_crowding": np.nan,
            "blocked_count": 0,
        }
        return selected, meta_by_code, stats


class PreprocessedShortIvolBuyDailyCappedAttentionPenaltyBacktestEngine(DailyCappedAttentionPenaltyBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        max_new_buys_per_day=DEFAULT_MAX_NEW_BUYS,
        backup_multiple=3,
        backup_extra=20,
        attention_tsmean_window=DEFAULT_ATTENTION_TSMEAN_WINDOW,
        short_ivol_tsmean_window=DEFAULT_SHORT_IVOL_TSMEAN_WINDOW,
    ):
        self.attention_tsmean_window = max(int(attention_tsmean_window), 1)
        self.short_ivol_tsmean_window = max(int(short_ivol_tsmean_window), 1)
        smoothed_panel = attention_panel
        if self.attention_tsmean_window > 1:
            smoothed_panel = SmoothedAttentionPanel(attention_panel, tsmean_window=self.attention_tsmean_window)

        super().__init__(
            data_loader,
            smoothed_panel,
            penalty_weight=penalty_weight,
            max_new_buys_per_day=max_new_buys_per_day,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )

        self.attention_panel_raw = attention_panel
        self.attention_panel = smoothed_panel
        self.factor_engine = ShortIvolBuyFactorEngine(data_loader.df_hfq, data_loader.df_ff3)
        self.short_ivol_preprocessor = ShortIvolCrossSectionalPreprocessor(data_loader.df_raw)
        self.trigger_engine = PreprocessedShortIvolBuyAttentionTriggerEngine(
            self.factor_engine,
            self.short_ivol_preprocessor,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            short_ivol_tsmean_window=self.short_ivol_tsmean_window,
        )

        suffix = "all" if self.max_new_buys_per_day is None else str(self.max_new_buys_per_day)
        parts = ["attention_penalty_w012", "buyshortivolprep"]
        if self.attention_tsmean_window > 1:
            parts.append(f"atttsmean{self.attention_tsmean_window}")
        if self.short_ivol_tsmean_window > 1:
            parts.append(f"sellivoltsmean{self.short_ivol_tsmean_window}")
        parts.append("xs010199z_neusizeind")
        parts.append(f"maxbuy{suffix}")
        self.variant_name = "_".join(parts)


class LongIvolShortFilterDailyCappedAttentionPenaltyBacktestEngine(DailyCappedAttentionPenaltyBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        max_new_buys_per_day=DEFAULT_MAX_NEW_BUYS,
        backup_multiple=3,
        backup_extra=20,
        attention_tsmean_window=DEFAULT_ATTENTION_TSMEAN_WINDOW,
        short_ivol_tsmean_window=DEFAULT_SHORT_IVOL_TSMEAN_WINDOW,
        buy_short_ivol_filter_pct=DEFAULT_BUY_SHORT_IVOL_FILTER_PCT,
    ):
        self.attention_tsmean_window = max(int(attention_tsmean_window), 1)
        self.short_ivol_tsmean_window = max(int(short_ivol_tsmean_window), 1)
        smoothed_panel = attention_panel
        if self.attention_tsmean_window > 1:
            smoothed_panel = SmoothedAttentionPanel(attention_panel, tsmean_window=self.attention_tsmean_window)

        super().__init__(
            data_loader,
            smoothed_panel,
            penalty_weight=penalty_weight,
            max_new_buys_per_day=max_new_buys_per_day,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )

        self.attention_panel_raw = attention_panel
        self.attention_panel = smoothed_panel
        self.buy_short_ivol_filter_pct = float(buy_short_ivol_filter_pct)
        self.factor_engine = ShortIvolBuyFactorEngine(data_loader.df_hfq, data_loader.df_ff3)
        self.trigger_engine = LongIvolShortFilterAttentionTriggerEngine(
            self.factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            short_ivol_tsmean_window=self.short_ivol_tsmean_window,
            buy_short_ivol_filter_pct=self.buy_short_ivol_filter_pct,
        )

        suffix = "all" if self.max_new_buys_per_day is None else str(self.max_new_buys_per_day)
        parts = ["attention_penalty_w012", "longivol_shortfilter"]
        if self.attention_tsmean_window > 1:
            parts.append(f"atttsmean{self.attention_tsmean_window}")
        if self.short_ivol_tsmean_window > 1:
            parts.append(f"sellivoltsmean{self.short_ivol_tsmean_window}")
        parts.append(f"buyshortpct{int(round(self.buy_short_ivol_filter_pct * 100))}")
        parts.append(f"maxbuy{suffix}")
        self.variant_name = "_".join(parts)


class ShortIvolBuyDailyCappedAttentionPenaltyBacktestEngine(DailyCappedAttentionPenaltyBacktestEngine):
    def __init__(
        self,
        data_loader,
        attention_panel,
        penalty_weight=DEFAULT_PENALTY_WEIGHT,
        max_new_buys_per_day=DEFAULT_MAX_NEW_BUYS,
        backup_multiple=3,
        backup_extra=20,
        attention_tsmean_window=DEFAULT_ATTENTION_TSMEAN_WINDOW,
        short_ivol_tsmean_window=DEFAULT_SHORT_IVOL_TSMEAN_WINDOW,
    ):
        self.attention_tsmean_window = max(int(attention_tsmean_window), 1)
        self.short_ivol_tsmean_window = max(int(short_ivol_tsmean_window), 1)
        smoothed_panel = attention_panel
        if self.attention_tsmean_window > 1:
            smoothed_panel = SmoothedAttentionPanel(attention_panel, tsmean_window=self.attention_tsmean_window)

        super().__init__(
            data_loader,
            smoothed_panel,
            penalty_weight=penalty_weight,
            max_new_buys_per_day=max_new_buys_per_day,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
        )

        self.attention_panel_raw = attention_panel
        self.attention_panel = smoothed_panel
        self.factor_engine = ShortIvolBuyFactorEngine(data_loader.df_hfq, data_loader.df_ff3)
        self.trigger_engine = ShortIvolBuyAttentionTriggerEngine(
            self.factor_engine,
            backup_multiple=backup_multiple,
            backup_extra=backup_extra,
            short_ivol_tsmean_window=self.short_ivol_tsmean_window,
        )

        suffix = "all" if self.max_new_buys_per_day is None else str(self.max_new_buys_per_day)
        parts = ["attention_penalty_w012", "buyshortivol"]
        if self.attention_tsmean_window > 1:
            parts.append(f"atttsmean{self.attention_tsmean_window}")
        if self.short_ivol_tsmean_window > 1:
            parts.append(f"sellivoltsmean{self.short_ivol_tsmean_window}")
        parts.append(f"maxbuy{suffix}")
        self.variant_name = "_".join(parts)


class AttentionPenaltyBacktraderStrategy(bt.Strategy):
    params = (
        ("loader", None),
        ("attention_panel", None),
        ("penalty_weight", DEFAULT_PENALTY_WEIGHT),
        ("max_new_buys", DEFAULT_MAX_NEW_BUYS),
        ("backup_multiple", 3),
        ("backup_extra", 20),
        ("attention_tsmean_window", DEFAULT_ATTENTION_TSMEAN_WINDOW),
        ("short_ivol_tsmean_window", DEFAULT_SHORT_IVOL_TSMEAN_WINDOW),
        ("buy_vol_signal", DEFAULT_BUY_VOL_SIGNAL),
        ("buy_short_ivol_filter_pct", DEFAULT_BUY_SHORT_IVOL_FILTER_PCT),
        ("buy_execution_mode", DEFAULT_BUY_EXECUTION_MODE),
        ("buy_exec_15m_dir", DEFAULT_BUY_15M_EXEC_DIR),
        ("buy_15m_ivol_lookback_days", DEFAULT_BUY_15M_IVOL_LOOKBACK_DAYS),
        ("buy_15m_ivol_trigger_ratio", DEFAULT_BUY_15M_IVOL_TRIGGER_RATIO),
        ("buy_15m_ivol_min_bars", DEFAULT_BUY_15M_IVOL_MIN_BARS),
        ("buy_15m_fallback_mode", DEFAULT_BUY_15M_FALLBACK_MODE),
        ("buy_15m_cache_size", DEFAULT_BUY_15M_CACHE_SIZE),
        ("intraday_factor_lookback_days", DEFAULT_INTRADAY_FACTOR_LOOKBACK_DAYS),
        ("intraday_factor_early_bars", DEFAULT_INTRADAY_FACTOR_EARLY_BARS),
        ("intraday_factor_buy_threshold", DEFAULT_INTRADAY_FACTOR_BUY_THRESHOLD),
        ("intraday_factor_sell_threshold", DEFAULT_INTRADAY_FACTOR_SELL_THRESHOLD),
        ("intraday_factor_exec_mode", DEFAULT_INTRADAY_FACTOR_EXEC_MODE),
        ("intraday_attention_panel", None),
        ("intraday_attention_early_bars", DEFAULT_INTRADAY_ATTENTION_EARLY_BARS),
        ("intraday_attention_buy_threshold", DEFAULT_INTRADAY_ATTENTION_BUY_THRESHOLD),
        ("intraday_attention_sell_threshold", DEFAULT_INTRADAY_ATTENTION_SELL_THRESHOLD),
        ("intraday_attention_exec_mode", DEFAULT_INTRADAY_ATTENTION_EXEC_MODE),
        ("refresh_dates", None),
        ("first_factor_date", None),
        ("last_data_date", None),
        ("universe_index", DEFAULT_UNIVERSE_INDEX),
        ("universe_codes", None),
        ("universe_effective_entries", None),
        ("universe_snapshot_date", None),
        ("universe_cache_path", None),
        ("universe_effective_panel_path", None),
        ("universe_monthly_panel_path", None),
    )

    def __init__(self):
        self.loader = self.p.loader
        self.calendar = self.datas[0]
        self.data_by_code = {data._name: data for data in self.datas[1:]}

        self.buy_vol_signal = str(self.p.buy_vol_signal or DEFAULT_BUY_VOL_SIGNAL).lower()

        if self.buy_vol_signal == "short":
            self.variant_engine = ShortIvolBuyDailyCappedAttentionPenaltyBacktestEngine(
                self.loader,
                self.p.attention_panel,
                penalty_weight=self.p.penalty_weight,
                max_new_buys_per_day=self.p.max_new_buys,
                backup_multiple=self.p.backup_multiple,
                backup_extra=self.p.backup_extra,
                attention_tsmean_window=self.p.attention_tsmean_window,
                short_ivol_tsmean_window=self.p.short_ivol_tsmean_window,
            )
        elif self.buy_vol_signal == "ra_only":
            self.variant_engine = ApproxRAOnlyDailyCappedBacktestEngine(
                self.loader,
                self.p.attention_panel,
                penalty_weight=self.p.penalty_weight,
                max_new_buys_per_day=self.p.max_new_buys,
                backup_multiple=self.p.backup_multiple,
                backup_extra=self.p.backup_extra,
                attention_tsmean_window=self.p.attention_tsmean_window,
                short_ivol_tsmean_window=self.p.short_ivol_tsmean_window,
            )
        elif self.buy_vol_signal == "ra_approx":
            self.variant_engine = ApproxRABuyDailyCappedAttentionPenaltyBacktestEngine(
                self.loader,
                self.p.attention_panel,
                penalty_weight=self.p.penalty_weight,
                max_new_buys_per_day=self.p.max_new_buys,
                backup_multiple=self.p.backup_multiple,
                backup_extra=self.p.backup_extra,
                attention_tsmean_window=self.p.attention_tsmean_window,
                short_ivol_tsmean_window=self.p.short_ivol_tsmean_window,
            )
        elif self.buy_vol_signal == "tech_composite":
            self.variant_engine = TechnicalCompositeBuyDailyCappedAttentionPenaltyBacktestEngine(
                self.loader,
                self.p.attention_panel,
                penalty_weight=self.p.penalty_weight,
                max_new_buys_per_day=self.p.max_new_buys,
                backup_multiple=self.p.backup_multiple,
                backup_extra=self.p.backup_extra,
                attention_tsmean_window=self.p.attention_tsmean_window,
                short_ivol_tsmean_window=self.p.short_ivol_tsmean_window,
            )
        elif self.buy_vol_signal == "short_preprocessed":
            self.variant_engine = PreprocessedShortIvolBuyDailyCappedAttentionPenaltyBacktestEngine(
                self.loader,
                self.p.attention_panel,
                penalty_weight=self.p.penalty_weight,
                max_new_buys_per_day=self.p.max_new_buys,
                backup_multiple=self.p.backup_multiple,
                backup_extra=self.p.backup_extra,
                attention_tsmean_window=self.p.attention_tsmean_window,
                short_ivol_tsmean_window=self.p.short_ivol_tsmean_window,
            )
        elif self.buy_vol_signal == "attention":
            self.variant_engine = AttentionBuyDailyCappedAttentionPenaltyBacktestEngine(
                self.loader,
                self.p.attention_panel,
                penalty_weight=self.p.penalty_weight,
                max_new_buys_per_day=self.p.max_new_buys,
                backup_multiple=self.p.backup_multiple,
                backup_extra=self.p.backup_extra,
                attention_tsmean_window=self.p.attention_tsmean_window,
                short_ivol_tsmean_window=self.p.short_ivol_tsmean_window,
            )
        elif self.buy_vol_signal == "long_short_filter":
            self.variant_engine = LongIvolShortFilterDailyCappedAttentionPenaltyBacktestEngine(
                self.loader,
                self.p.attention_panel,
                penalty_weight=self.p.penalty_weight,
                max_new_buys_per_day=self.p.max_new_buys,
                backup_multiple=self.p.backup_multiple,
                backup_extra=self.p.backup_extra,
                attention_tsmean_window=self.p.attention_tsmean_window,
                short_ivol_tsmean_window=self.p.short_ivol_tsmean_window,
                buy_short_ivol_filter_pct=self.p.buy_short_ivol_filter_pct,
            )
        elif self.p.attention_tsmean_window > 1 or self.p.short_ivol_tsmean_window > 1:
            self.variant_engine = TsMeanDailyCappedAttentionPenaltyBacktestEngine(
                self.loader,
                self.p.attention_panel,
                penalty_weight=self.p.penalty_weight,
                max_new_buys_per_day=self.p.max_new_buys,
                backup_multiple=self.p.backup_multiple,
                backup_extra=self.p.backup_extra,
                attention_tsmean_window=self.p.attention_tsmean_window,
                short_ivol_tsmean_window=self.p.short_ivol_tsmean_window,
            )
        else:
            self.variant_engine = DailyCappedAttentionPenaltyBacktestEngine(
                self.loader,
                self.p.attention_panel,
                penalty_weight=self.p.penalty_weight,
                max_new_buys_per_day=self.p.max_new_buys,
                backup_multiple=self.p.backup_multiple,
                backup_extra=self.p.backup_extra,
            )

        # Replace default UniverseFilter with NoPennyUniverseFilter (skip close < 2 filter)
        no_penny_filter = NoPennyUniverseFilter(self.loader.df_hfq, self.loader.df_raw)
        self.variant_engine.filter = no_penny_filter
        self.filter = no_penny_filter
        self.factor_engine = self.variant_engine.factor_engine
        self.trigger_engine = self.variant_engine.trigger_engine
        self.variant_name = self.variant_engine.variant_name
        self.universe_index = str(self.p.universe_index or DEFAULT_UNIVERSE_INDEX).lower()
        self.universe_codes = tuple(self.p.universe_codes or tuple())
        self.universe_effective_entries = tuple(self.p.universe_effective_entries or tuple())
        if self.universe_effective_entries:
            restricted_filter = TimeVaryingUniverseFilter(
                self.filter,
                self.universe_effective_entries,
                self.universe_index,
            )
            self.filter = restricted_filter
            self.variant_engine.filter = restricted_filter
        elif self.universe_index != "all" and self.universe_codes:
            restricted_filter = RestrictedUniverseFilter(
                self.filter,
                self.universe_codes,
                self.universe_index,
            )
            self.filter = restricted_filter
            self.variant_engine.filter = restricted_filter
        self.buy_execution_mode = str(self.p.buy_execution_mode or DEFAULT_BUY_EXECUTION_MODE).lower()
        self.buy_executor = None
        self.intraday_factor_executor = None
        if self.buy_execution_mode == "intraday_ivol":
            self.variant_name = (
                f"{self.variant_name}_buy15mivol"
                f"{int(round(float(self.p.buy_15m_ivol_trigger_ratio) * 100)):02d}"
                f"lb{int(self.p.buy_15m_ivol_lookback_days)}"
                f"mb{int(self.p.buy_15m_ivol_min_bars)}"
            )
            self.buy_executor = IntradayIvolBuyExecutor(
                intraday_dir=self.p.buy_exec_15m_dir,
                lookback_days=self.p.buy_15m_ivol_lookback_days,
                trigger_ratio=self.p.buy_15m_ivol_trigger_ratio,
                min_bars=self.p.buy_15m_ivol_min_bars,
                fallback_mode=self.p.buy_15m_fallback_mode,
                cache_size=self.p.buy_15m_cache_size,
            )
        elif self.buy_execution_mode == "tail_1450":
            self.variant_name = f"{self.variant_name}_buy1450tail_{self.p.buy_15m_fallback_mode}"
            self.buy_executor = IntradayIvolBuyExecutor(
                intraday_dir=self.p.buy_exec_15m_dir,
                lookback_days=self.p.buy_15m_ivol_lookback_days,
                trigger_ratio=self.p.buy_15m_ivol_trigger_ratio,
                min_bars=self.p.buy_15m_ivol_min_bars,
                fallback_mode=self.p.buy_15m_fallback_mode,
                cache_size=self.p.buy_15m_cache_size,
            )
        elif self.buy_execution_mode == "intraday_attention":
            self.variant_name = (
                f"{self.variant_name}_att15m"
                f"b{int(round(float(self.p.intraday_attention_buy_threshold) * 100)):02d}"
                f"s{int(round(float(self.p.intraday_attention_sell_threshold) * 100)):02d}"
                f"e{int(self.p.intraday_attention_early_bars)}"
            )
        elif self.buy_execution_mode == "buy_intraday_attention":
            self.variant_name = (
                f"{self.variant_name}_buyatt15m"
                f"b{int(round(float(self.p.intraday_attention_buy_threshold) * 100)):02d}"
                f"e{int(self.p.intraday_attention_early_bars)}"
            )
        elif self.buy_execution_mode == "sell_intraday_attention":
            self.variant_name = (
                f"{self.variant_name}_sellatt15m"
                f"s{int(round(float(self.p.intraday_attention_sell_threshold) * 100)):02d}"
                f"e{int(self.p.intraday_attention_early_bars)}"
            )
        elif self.buy_execution_mode == "intraday_factor3":
            self.variant_name = (
                f"{self.variant_name}_factor15m"
                f"b{int(round(float(self.p.intraday_factor_buy_threshold) * 100)):02d}"
                f"s{int(round(float(self.p.intraday_factor_sell_threshold) * 100)):02d}"
                f"e{int(self.p.intraday_factor_early_bars)}"
                f"lb{int(self.p.intraday_factor_lookback_days)}"
            )
            self.intraday_factor_executor = IntradayFactor3Executor(
                intraday_dir=self.p.buy_exec_15m_dir,
                lookback_days=self.p.intraday_factor_lookback_days,
                early_bars=self.p.intraday_factor_early_bars,
                exec_mode=self.p.intraday_factor_exec_mode,
                cache_size=self.p.buy_15m_cache_size,
            )
        self.intraday_attention_panel = self.p.intraday_attention_panel

        self.refresh_dates = {pd.Timestamp(x) for x in (self.p.refresh_dates or [])}
        self.first_factor_date = pd.Timestamp(self.p.first_factor_date)
        self.last_data_date = dict(self.p.last_data_date or {})

        self.daily_nav = []
        self.trade_log = []
        self.daily_status = []

        self.initialized = False
        self.factor_cache = None
        self.pending_sells = set()
        self.pending_buys = []
        self.pending_buy_slots = 0
        self.pending_sell_reasons = {}
        self.pending_intraday_buy_plan = []
        self.last_nav = float(base.INITIAL_CAPITAL)
        self.today_selection_stats = self.variant_engine._empty_stats(0, 0)
        self.last_intraday_buy_stats = self._empty_intraday_buy_stats()
        self.last_intraday_sell_stats = self._empty_intraday_sell_stats()

    def prenext(self):
        self.next()

    def nextstart(self):
        self.next()

    def prenext_open(self):
        self.next_open()

    def nextstart_open(self):
        self.next_open()

    def _current_date(self, data=None):
        data = data or self.calendar
        return pd.Timestamp(bt.num2date(data.datetime[0]).date())

    def _data_has_bar_today(self, data, today):
        if len(data) == 0:
            return False
        return self._current_date(data) == today

    @staticmethod
    def _safe_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return np.nan
        if np.isnan(value):
            return np.nan
        return value

    def _previous_close(self, data):
        if len(data) <= 1:
            return np.nan
        return self._safe_float(data.close[-1])

    def _holding_codes(self):
        codes = []
        for code, data in self.data_by_code.items():
            if self.getposition(data).size > 0:
                codes.append(code)
        return codes

    @staticmethod
    def _dedupe_codes(codes):
        seen = set()
        ordered = []
        for code in codes:
            if code in seen:
                continue
            seen.add(code)
            ordered.append(code)
        return ordered

    def _estimate_buy_cost(self, open_price, shares):
        exec_price = open_price * (1.0 + base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
        trade_value = shares * exec_price
        total_cost = trade_value * (1.0 + base.COMMISSION_RATE)
        return exec_price, trade_value, total_cost

    def _estimate_sell_value(self, open_price, shares):
        exec_price = open_price * (1.0 - base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
        trade_value = shares * exec_price
        net_value = trade_value * (1.0 - base.COMMISSION_RATE - base.STAMP_DUTY_RATE)
        return exec_price, trade_value, net_value

    @staticmethod
    def _empty_intraday_buy_stats():
        return {"planned": 0, "executed": 0, "triggered": 0, "fallback": 0}

    @staticmethod
    def _empty_intraday_sell_stats():
        return {"planned": 0, "executed": 0, "early": 0, "late": 0}

    def _can_trade_buy_price(self, code, prev_close, raw_price):
        if pd.isna(raw_price) or raw_price <= 0:
            return False

        limit_pct = base._get_limit_pct(code)
        if pd.notna(prev_close) and prev_close > 0:
            if raw_price >= prev_close * (1.0 + limit_pct * 0.95):
                return False
            if raw_price <= prev_close * (1.0 - limit_pct * 0.95):
                return False
        return True

    def _can_trade_sell_price(self, code, prev_close, raw_price):
        if pd.isna(raw_price) or raw_price <= 0:
            return False

        limit_pct = base._get_limit_pct(code)
        if pd.notna(prev_close) and prev_close > 0:
            if raw_price <= prev_close * (1.0 - limit_pct * 0.95):
                return False
        return True

    def _get_intraday_attention_snapshot(self, today, code):
        if self.intraday_attention_panel is None:
            return {"crowding": np.nan, "attention_up": np.nan, "attention_down": np.nan}
        return {
            "crowding": self._safe_float(self.intraday_attention_panel.get_metric(today, code, "crowding_pct")),
            "attention_up": self._safe_float(self.intraday_attention_panel.get_metric(today, code, "attention_up")),
            "attention_down": self._safe_float(self.intraday_attention_panel.get_metric(today, code, "attention_down")),
        }

    def _queue_intraday_buy_plan(self, today, selected_buys, buy_meta, allocatable_per_stock):
        self.pending_intraday_buy_plan = []
        planned_buy_codes = set()

        for code in selected_buys:
            data = self.data_by_code.get(code)
            if data is None or not self._data_has_bar_today(data, today):
                continue

            if self.getposition(data).size > 0 or code in planned_buy_codes:
                continue

            crowding = buy_meta.get(code, {}).get("crowding_pct", np.nan)
            self.pending_intraday_buy_plan.append(
                {
                    "date": pd.Timestamp(today),
                    "code": code,
                    "allocatable_value": float(allocatable_per_stock),
                    "crowding": float(crowding) if pd.notna(crowding) else np.nan,
                    "prev_close": self._previous_close(data),
                }
            )
            planned_buy_codes.add(code)

    def _execute_intraday_buys(self, today):
        today = pd.Timestamp(today)
        plans = [plan for plan in self.pending_intraday_buy_plan if pd.Timestamp(plan["date"]) == today]
        self.pending_intraday_buy_plan = []
        stats = self._empty_intraday_buy_stats()
        stats["planned"] = len(plans)
        if not plans or self.buy_executor is None:
            self.last_intraday_buy_stats = stats
            return

        cash_available = float(self.broker.getcash())
        for plan in plans:
            code = plan["code"]
            data = self.data_by_code.get(code)
            if data is None or not self._data_has_bar_today(data, today):
                continue
            if self.getposition(data).size > 0:
                continue

            if self.buy_execution_mode == "tail_1450":
                buy_candidates = self.buy_executor.list_tail_buy_candidates(code, today)
            else:
                buy_candidates = self.buy_executor.list_buy_candidates(code, today)
            if not buy_candidates:
                continue

            chosen = None
            for candidate in buy_candidates:
                raw_price = self._safe_float(candidate.get("raw_price"))
                if self._can_trade_buy_price(code, plan.get("prev_close", np.nan), raw_price):
                    chosen = candidate
                    break
            if chosen is None:
                continue

            raw_exec_price = self._safe_float(chosen.get("raw_price"))
            if pd.isna(raw_exec_price) or raw_exec_price <= 0:
                continue

            exec_price, trade_value, total_cost = self._estimate_buy_cost(raw_exec_price, 100)
            raw_shares = float(plan["allocatable_value"]) / exec_price if exec_price > 0 else 0.0
            trade_shares = int(raw_shares // 100) * 100
            if trade_shares < 100:
                continue

            exec_price, trade_value, total_cost = self._estimate_buy_cost(raw_exec_price, trade_shares)
            if cash_available < total_cost:
                continue

            self.broker.positions[data].fix(trade_shares, exec_price)
            self.broker.add_cash(-total_cost)

            crowding = plan.get("crowding", np.nan)
            if chosen["kind"] == "triggered":
                reason = (
                    f"{self.variant_name}(crowding={crowding:.3f})|buy15m_ivol("
                    f"ratio={chosen.get('ivol_ratio', np.nan):.3f},"
                    f"curr={chosen.get('current_ivol', np.nan):.6f},"
                    f"hist={chosen.get('baseline_ivol', np.nan):.6f},"
                    f"signal_bar={int(chosen.get('signal_bar_key', 0))},"
                    f"exec_bar={int(chosen.get('exec_bar_key', 0))})"
                )
            else:
                reason = (
                    f"{self.variant_name}(crowding={crowding:.3f})|buy15m_tail("
                    f"bar={int(chosen.get('exec_bar_key', 0))},mode={self.buy_executor.fallback_mode})"
                )

            self.trade_log.append(
                {
                    "date": today.strftime("%Y-%m-%d"),
                    "code": code,
                    "action": "BUY",
                    "price": round(float(exec_price), 4),
                    "shares": int(trade_shares),
                    "value": round(float(trade_value), 2),
                    "reason": reason,
                }
            )

            cash_available -= total_cost
            stats["executed"] += 1
            if chosen["kind"] == "triggered":
                stats["triggered"] += 1
            else:
                stats["fallback"] += 1

        self.last_intraday_buy_stats = stats

    def _build_intraday_attention_sell_action(self, today, code):
        data = self.data_by_code.get(code)
        if data is None or not self._data_has_bar_today(data, today):
            return None

        position = self.getposition(data)
        shares = int(position.size)
        if shares <= 0:
            return None

        prev_close = self._previous_close(data)
        signal = self._get_intraday_attention_snapshot(today, code)
        crowding = signal["crowding"]
        prefer_early = pd.notna(crowding) and crowding >= float(self.p.intraday_attention_sell_threshold)
        stage_order = [("early", False), ("late", True)] if prefer_early else [("late", True), ("early", False)]

        for stage_name, late_flag in stage_order:
            raw_price = (
                self._safe_float(self.intraday_attention_panel.get_exec_price(today, code, self.p.intraday_attention_exec_mode, late=late_flag))
                if self.intraday_attention_panel is not None
                else np.nan
            )
            if self._can_trade_sell_price(code, prev_close, raw_price):
                return {
                    "code": code,
                    "data": data,
                    "shares": shares,
                    "stage": stage_name,
                    "raw_price": raw_price,
                    "signal": signal,
                }
        return None

    def _build_intraday_attention_buy_action(self, today, plan):
        code = plan["code"]
        data = self.data_by_code.get(code)
        if data is None or not self._data_has_bar_today(data, today):
            return None
        if self.getposition(data).size > 0:
            return None

        signal = self._get_intraday_attention_snapshot(today, code)
        crowding = signal["crowding"]
        prefer_early = pd.notna(crowding) and crowding <= float(self.p.intraday_attention_buy_threshold)
        stage_order = [("early", False), ("late", True)] if prefer_early else [("late", True), ("early", False)]
        prev_close = plan.get("prev_close", np.nan)

        for stage_name, late_flag in stage_order:
            raw_price = (
                self._safe_float(self.intraday_attention_panel.get_exec_price(today, code, self.p.intraday_attention_exec_mode, late=late_flag))
                if self.intraday_attention_panel is not None
                else np.nan
            )
            if self._can_trade_buy_price(code, prev_close, raw_price):
                action = dict(plan)
                action.update(
                    {
                        "stage": stage_name,
                        "raw_price": raw_price,
                        "signal": signal,
                    }
                )
                return action
        return None

    def _execute_intraday_attention_day(self, today):
        today = pd.Timestamp(today)
        sell_stats = self._empty_intraday_sell_stats()
        buy_stats = self._empty_intraday_buy_stats()

        if self.intraday_attention_panel is None:
            self.last_intraday_sell_stats = sell_stats
            self.last_intraday_buy_stats = buy_stats
            return

        buy_plans = [plan for plan in self.pending_intraday_buy_plan if pd.Timestamp(plan["date"]) == today]
        self.pending_intraday_buy_plan = [plan for plan in self.pending_intraday_buy_plan if pd.Timestamp(plan["date"]) != today]
        sell_codes = [code for code in self._dedupe_codes(list(self.pending_sells)) if code in self.data_by_code]

        sell_stats["planned"] = len(sell_codes)
        buy_stats["planned"] = len(buy_plans)

        sell_actions = []
        for code in sell_codes:
            action = self._build_intraday_attention_sell_action(today, code)
            if action is not None:
                sell_actions.append(action)

        buy_actions = []
        for plan in buy_plans:
            action = self._build_intraday_attention_buy_action(today, plan)
            if action is not None:
                buy_actions.append(action)

        def execute_sell_group(actions, stage_name):
            nonlocal sell_stats
            for action in actions:
                code = action["code"]
                data = action["data"]
                if self.getposition(data).size <= 0:
                    self.pending_sells.discard(code)
                    self.pending_sell_reasons.pop(code, None)
                    continue

                sell_price = action["raw_price"] * (1.0 - base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
                if pd.isna(sell_price) or sell_price <= 0:
                    continue

                shares = int(self.getposition(data).size)
                sell_value = shares * sell_price
                cost = sell_value * (base.COMMISSION_RATE + base.STAMP_DUTY_RATE)
                self.broker.positions[data].fix(0, 0.0)
                self.broker.add_cash(sell_value - cost)

                signal = action["signal"]
                reason = self.pending_sell_reasons.get(code, "unknown")
                reason = (
                    f"{reason}|sell15m_attention("
                    f"stage={stage_name},crowding={signal.get('crowding', np.nan):.3f},"
                    f"up={signal.get('attention_up', np.nan):.3f},down={signal.get('attention_down', np.nan):.3f})"
                )
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

                self.pending_sells.discard(code)
                self.pending_sell_reasons.pop(code, None)
                sell_stats["executed"] += 1
                if stage_name == "early":
                    sell_stats["early"] += 1
                else:
                    sell_stats["late"] += 1

        def execute_buy_group(actions, stage_name):
            nonlocal buy_stats
            remaining = []
            for action in actions:
                data = self.data_by_code.get(action["code"])
                if data is None or self.getposition(data).size > 0:
                    continue
                remaining.append(action)
            if not remaining:
                return

            cash_available = float(self.broker.getcash())
            standard_position_value = self.last_nav / base.TARGET_POSITIONS
            allocatable_per_stock = min(cash_available / len(remaining), standard_position_value)

            for action in remaining:
                code = action["code"]
                data = self.data_by_code.get(code)
                if data is None or self.getposition(data).size > 0:
                    continue

                buy_price = action["raw_price"] * (1.0 + base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
                if pd.isna(buy_price) or buy_price <= 0:
                    continue

                raw_shares = allocatable_per_stock / buy_price if buy_price > 0 else 0.0
                trade_shares = int(raw_shares // 100) * 100
                if trade_shares < 100:
                    continue

                trade_value = trade_shares * buy_price
                total_cost = trade_value * (1.0 + base.COMMISSION_RATE)
                if float(self.broker.getcash()) < total_cost:
                    continue

                self.broker.positions[data].fix(trade_shares, buy_price)
                self.broker.add_cash(-total_cost)

                signal = action["signal"]
                reason = (
                    f"{self.variant_name}(crowding={action.get('crowding', np.nan):.3f})|buy15m_attention("
                    f"stage={stage_name},early_crowding={signal.get('crowding', np.nan):.3f},"
                    f"up={signal.get('attention_up', np.nan):.3f},down={signal.get('attention_down', np.nan):.3f})"
                )
                self.trade_log.append(
                    {
                        "date": today.strftime("%Y-%m-%d"),
                        "code": code,
                        "action": "BUY",
                        "price": round(float(buy_price), 4),
                        "shares": int(trade_shares),
                        "value": round(float(trade_value), 2),
                        "reason": reason,
                    }
                )
                buy_stats["executed"] += 1
                if stage_name == "early":
                    buy_stats["triggered"] += 1
                else:
                    buy_stats["fallback"] += 1

        early_sells = [row for row in sell_actions if row["stage"] == "early"]
        late_sells = [row for row in sell_actions if row["stage"] == "late"]
        early_buys = [row for row in buy_actions if row["stage"] == "early"]
        late_buys = [row for row in buy_actions if row["stage"] == "late"]

        execute_sell_group(early_sells, "early")
        execute_buy_group(early_buys, "early")
        execute_sell_group(late_sells, "late")
        execute_buy_group(late_buys, "late")

        self.last_intraday_sell_stats = sell_stats
        self.last_intraday_buy_stats = buy_stats

    def _execute_intraday_attention_buys_only(self, today):
        today = pd.Timestamp(today)
        buy_stats = self._empty_intraday_buy_stats()
        buy_plans = [plan for plan in self.pending_intraday_buy_plan if pd.Timestamp(plan["date"]) == today]
        self.pending_intraday_buy_plan = [plan for plan in self.pending_intraday_buy_plan if pd.Timestamp(plan["date"]) != today]
        buy_stats["planned"] = len(buy_plans)

        if self.intraday_attention_panel is None or not buy_plans:
            self.last_intraday_buy_stats = buy_stats
            self.last_intraday_sell_stats = self._empty_intraday_sell_stats()
            return

        buy_actions = []
        for plan in buy_plans:
            action = self._build_intraday_attention_buy_action(today, plan)
            if action is not None:
                buy_actions.append(action)

        def execute_buy_group(actions, stage_name):
            nonlocal buy_stats
            remaining = []
            for action in actions:
                data = self.data_by_code.get(action["code"])
                if data is None or self.getposition(data).size > 0:
                    continue
                remaining.append(action)
            if not remaining:
                return

            cash_available = float(self.broker.getcash())
            standard_position_value = self.last_nav / base.TARGET_POSITIONS
            allocatable_per_stock = min(cash_available / len(remaining), standard_position_value)

            for action in remaining:
                code = action["code"]
                data = self.data_by_code.get(code)
                if data is None or self.getposition(data).size > 0:
                    continue

                buy_price = action["raw_price"] * (1.0 + base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
                if pd.isna(buy_price) or buy_price <= 0:
                    continue

                raw_shares = allocatable_per_stock / buy_price if buy_price > 0 else 0.0
                trade_shares = int(raw_shares // 100) * 100
                if trade_shares < 100:
                    continue

                trade_value = trade_shares * buy_price
                total_cost = trade_value * (1.0 + base.COMMISSION_RATE)
                if float(self.broker.getcash()) < total_cost:
                    continue

                self.broker.positions[data].fix(trade_shares, buy_price)
                self.broker.add_cash(-total_cost)

                signal = action["signal"]
                reason = (
                    f"{self.variant_name}(crowding={action.get('crowding', np.nan):.3f})|buy15m_attention("
                    f"stage={stage_name},early_crowding={signal.get('crowding', np.nan):.3f},"
                    f"up={signal.get('attention_up', np.nan):.3f},down={signal.get('attention_down', np.nan):.3f})"
                )
                self.trade_log.append(
                    {
                        "date": today.strftime("%Y-%m-%d"),
                        "code": code,
                        "action": "BUY",
                        "price": round(float(buy_price), 4),
                        "shares": int(trade_shares),
                        "value": round(float(trade_value), 2),
                        "reason": reason,
                    }
                )
                buy_stats["executed"] += 1
                if stage_name == "early":
                    buy_stats["triggered"] += 1
                else:
                    buy_stats["fallback"] += 1

        early_buys = [row for row in buy_actions if row["stage"] == "early"]
        late_buys = [row for row in buy_actions if row["stage"] == "late"]
        execute_buy_group(early_buys, "early")
        execute_buy_group(late_buys, "late")

        self.last_intraday_buy_stats = buy_stats
        self.last_intraday_sell_stats = self._empty_intraday_sell_stats()

    def _execute_intraday_attention_sells_only(self, today):
        today = pd.Timestamp(today)
        sell_stats = self._empty_intraday_sell_stats()
        sell_codes = [code for code in self._dedupe_codes(list(self.pending_sells)) if code in self.data_by_code]
        sell_stats["planned"] = len(sell_codes)

        if self.intraday_attention_panel is None or not sell_codes:
            self.last_intraday_sell_stats = sell_stats
            self.last_intraday_buy_stats = self._empty_intraday_buy_stats()
            return

        sell_actions = []
        for code in sell_codes:
            action = self._build_intraday_attention_sell_action(today, code)
            if action is not None:
                sell_actions.append(action)

        def execute_sell_group(actions, stage_name):
            nonlocal sell_stats
            for action in actions:
                code = action["code"]
                data = action["data"]
                if self.getposition(data).size <= 0:
                    self.pending_sells.discard(code)
                    self.pending_sell_reasons.pop(code, None)
                    continue

                sell_price = action["raw_price"] * (1.0 - base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
                if pd.isna(sell_price) or sell_price <= 0:
                    continue

                shares = int(self.getposition(data).size)
                sell_value = shares * sell_price
                cost = sell_value * (base.COMMISSION_RATE + base.STAMP_DUTY_RATE)
                self.broker.positions[data].fix(0, 0.0)
                self.broker.add_cash(sell_value - cost)

                signal = action["signal"]
                reason = self.pending_sell_reasons.get(code, "unknown")
                reason = (
                    f"{reason}|sell15m_attention("
                    f"stage={stage_name},crowding={signal.get('crowding', np.nan):.3f},"
                    f"up={signal.get('attention_up', np.nan):.3f},down={signal.get('attention_down', np.nan):.3f})"
                )
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

                self.pending_sells.discard(code)
                self.pending_sell_reasons.pop(code, None)
                sell_stats["executed"] += 1
                if stage_name == "early":
                    sell_stats["early"] += 1
                else:
                    sell_stats["late"] += 1

        early_sells = [row for row in sell_actions if row["stage"] == "early"]
        late_sells = [row for row in sell_actions if row["stage"] == "late"]
        execute_sell_group(early_sells, "early")
        execute_sell_group(late_sells, "late")

        self.last_intraday_sell_stats = sell_stats
        self.last_intraday_buy_stats = self._empty_intraday_buy_stats()

    def _build_intraday_factor3_buy_action(self, today, plan):
        if self.intraday_factor_executor is None:
            return None

        code = plan["code"]
        data = self.data_by_code.get(code)
        if data is None or not self._data_has_bar_today(data, today):
            return None
        if self.getposition(data).size > 0:
            return None

        snapshot = self.intraday_factor_executor.get_snapshot(code, today)
        if snapshot is None:
            return None

        prefer_early = pd.notna(snapshot["buy_score"]) and snapshot["buy_score"] >= float(self.p.intraday_factor_buy_threshold)
        stage_order = [
            ("early", snapshot.get("exec_early_price", np.nan)),
            ("late", snapshot.get("exec_late_price", np.nan)),
        ] if prefer_early else [
            ("late", snapshot.get("exec_late_price", np.nan)),
            ("early", snapshot.get("exec_early_price", np.nan)),
        ]

        prev_close = plan.get("prev_close", np.nan)
        for stage_name, raw_price in stage_order:
            if self._can_trade_buy_price(code, prev_close, raw_price):
                action = dict(plan)
                action.update(
                    {
                        "stage": stage_name,
                        "raw_price": raw_price,
                        "signal": snapshot,
                    }
                )
                return action
        return None

    def _build_intraday_factor3_sell_action(self, today, code):
        if self.intraday_factor_executor is None:
            return None

        data = self.data_by_code.get(code)
        if data is None or not self._data_has_bar_today(data, today):
            return None

        position = self.getposition(data)
        shares = int(position.size)
        if shares <= 0:
            return None

        snapshot = self.intraday_factor_executor.get_snapshot(code, today)
        if snapshot is None:
            return None

        prefer_early = pd.notna(snapshot["sell_score"]) and snapshot["sell_score"] >= float(self.p.intraday_factor_sell_threshold)
        stage_order = [
            ("early", snapshot.get("exec_early_price", np.nan)),
            ("late", snapshot.get("exec_late_price", np.nan)),
        ] if prefer_early else [
            ("late", snapshot.get("exec_late_price", np.nan)),
            ("early", snapshot.get("exec_early_price", np.nan)),
        ]

        prev_close = self._previous_close(data)
        for stage_name, raw_price in stage_order:
            if self._can_trade_sell_price(code, prev_close, raw_price):
                return {
                    "code": code,
                    "data": data,
                    "shares": shares,
                    "stage": stage_name,
                    "raw_price": raw_price,
                    "signal": snapshot,
                }
        return None

    def _execute_intraday_factor3_day(self, today):
        today = pd.Timestamp(today)
        sell_stats = self._empty_intraday_sell_stats()
        buy_stats = self._empty_intraday_buy_stats()

        buy_plans = [plan for plan in self.pending_intraday_buy_plan if pd.Timestamp(plan["date"]) == today]
        self.pending_intraday_buy_plan = [plan for plan in self.pending_intraday_buy_plan if pd.Timestamp(plan["date"]) != today]
        sell_codes = [code for code in self._dedupe_codes(list(self.pending_sells)) if code in self.data_by_code]
        sell_stats["planned"] = len(sell_codes)
        buy_stats["planned"] = len(buy_plans)

        sell_actions = []
        for code in sell_codes:
            action = self._build_intraday_factor3_sell_action(today, code)
            if action is not None:
                sell_actions.append(action)

        buy_actions = []
        for plan in buy_plans:
            action = self._build_intraday_factor3_buy_action(today, plan)
            if action is not None:
                buy_actions.append(action)

        def execute_sell_group(actions, stage_name):
            nonlocal sell_stats
            for action in actions:
                code = action["code"]
                data = action["data"]
                if self.getposition(data).size <= 0:
                    self.pending_sells.discard(code)
                    self.pending_sell_reasons.pop(code, None)
                    continue

                sell_price = action["raw_price"] * (1.0 - base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
                if pd.isna(sell_price) or sell_price <= 0:
                    continue

                shares = int(self.getposition(data).size)
                sell_value = shares * sell_price
                cost = sell_value * (base.COMMISSION_RATE + base.STAMP_DUTY_RATE)
                self.broker.positions[data].fix(0, 0.0)
                self.broker.add_cash(sell_value - cost)

                signal = action["signal"]
                reason = self.pending_sell_reasons.get(code, "unknown")
                reason = (
                    f"{reason}|sell15m_factor3("
                    f"stage={stage_name},score={signal.get('sell_score', np.nan):.3f},"
                    f"ivol={signal.get('ivol', np.nan):.6f},delta={signal.get('delta_ivol', np.nan):.6f},"
                    f"cbmom={signal.get('cbmom', np.nan):.3f})"
                )
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

                self.pending_sells.discard(code)
                self.pending_sell_reasons.pop(code, None)
                sell_stats["executed"] += 1
                if stage_name == "early":
                    sell_stats["early"] += 1
                else:
                    sell_stats["late"] += 1

        def execute_buy_group(actions, stage_name):
            nonlocal buy_stats
            remaining = []
            for action in actions:
                data = self.data_by_code.get(action["code"])
                if data is None or self.getposition(data).size > 0:
                    continue
                remaining.append(action)
            if not remaining:
                return

            cash_available = float(self.broker.getcash())
            standard_position_value = self.last_nav / base.TARGET_POSITIONS
            allocatable_per_stock = min(cash_available / len(remaining), standard_position_value)

            for action in remaining:
                code = action["code"]
                data = self.data_by_code.get(code)
                if data is None or self.getposition(data).size > 0:
                    continue

                buy_price = action["raw_price"] * (1.0 + base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
                if pd.isna(buy_price) or buy_price <= 0:
                    continue

                raw_shares = allocatable_per_stock / buy_price if buy_price > 0 else 0.0
                trade_shares = int(raw_shares // 100) * 100
                if trade_shares < 100:
                    continue

                trade_value = trade_shares * buy_price
                total_cost = trade_value * (1.0 + base.COMMISSION_RATE)
                if float(self.broker.getcash()) < total_cost:
                    continue

                self.broker.positions[data].fix(trade_shares, buy_price)
                self.broker.add_cash(-total_cost)

                signal = action["signal"]
                reason = (
                    f"{self.variant_name}(crowding={action.get('crowding', np.nan):.3f})|buy15m_factor3("
                    f"stage={stage_name},score={signal.get('buy_score', np.nan):.3f},"
                    f"ivol={signal.get('ivol', np.nan):.6f},delta={signal.get('delta_ivol', np.nan):.6f},"
                    f"cbmom={signal.get('cbmom', np.nan):.3f})"
                )
                self.trade_log.append(
                    {
                        "date": today.strftime("%Y-%m-%d"),
                        "code": code,
                        "action": "BUY",
                        "price": round(float(buy_price), 4),
                        "shares": int(trade_shares),
                        "value": round(float(trade_value), 2),
                        "reason": reason,
                    }
                )
                buy_stats["executed"] += 1
                if stage_name == "early":
                    buy_stats["triggered"] += 1
                else:
                    buy_stats["fallback"] += 1

        early_sells = [row for row in sell_actions if row["stage"] == "early"]
        late_sells = [row for row in sell_actions if row["stage"] == "late"]
        early_buys = [row for row in buy_actions if row["stage"] == "early"]
        late_buys = [row for row in buy_actions if row["stage"] == "late"]

        execute_sell_group(early_sells, "early")
        execute_buy_group(early_buys, "early")
        execute_sell_group(late_sells, "late")
        execute_buy_group(late_buys, "late")

        self.last_intraday_sell_stats = sell_stats
        self.last_intraday_buy_stats = buy_stats

    def _manual_delist_liquidation(self, today):
        for code in list(self._holding_codes()):
            last_trade_date = self.last_data_date.get(code)
            if last_trade_date is None:
                continue
            if today <= pd.Timestamp(last_trade_date) + pd.Timedelta(days=base.DELIST_NO_DATA_DAYS):
                continue

            data = self.data_by_code.get(code)
            if data is None:
                continue

            position = self.getposition(data)
            shares = int(position.size)
            if shares <= 0:
                continue

            liq_price = self._safe_float(data.close[0]) if len(data) else np.nan
            if pd.isna(liq_price) or liq_price <= 0:
                liq_price = 0.0

            sell_value = shares * liq_price
            cost = sell_value * (base.COMMISSION_RATE + base.STAMP_DUTY_RATE)

            self.broker.positions[data].fix(0, 0.0)
            self.broker.add_cash(sell_value - cost)

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

            self.pending_sells.discard(code)
            self.pending_sell_reasons.pop(code, None)

    def _set_trigger_date(self, today):
        if hasattr(self.trigger_engine, "current_date"):
            self.trigger_engine.current_date = pd.Timestamp(today)

    def next_open(self):
        today = self._current_date()
        for order in list(self.broker.get_orders_open(safe=False)):
            self.cancel(order)

        self._manual_delist_liquidation(today)

        cash_available = float(self.broker.getcash())
        self.today_selection_stats = self.variant_engine._empty_stats(len(self.pending_buys), self.pending_buy_slots)

        if self.pending_sells and self.buy_execution_mode not in {"intraday_attention", "sell_intraday_attention", "intraday_factor3"}:
            for code in list(self.pending_sells):
                data = self.data_by_code.get(code)
                if data is None or not self._data_has_bar_today(data, today):
                    continue

                position = self.getposition(data)
                shares = int(position.size)
                if shares <= 0:
                    self.pending_sells.discard(code)
                    self.pending_sell_reasons.pop(code, None)
                    continue

                sell_open = self._safe_float(data.open[0])
                prev_close = self._previous_close(data)
                if pd.isna(sell_open) or sell_open <= 0:
                    continue

                limit_pct = base._get_limit_pct(code)
                if pd.notna(prev_close) and prev_close > 0:
                    if sell_open <= prev_close * (1.0 - limit_pct * 0.95):
                        continue

                _, _, net_value = self._estimate_sell_value(sell_open, shares)
                order = self.close(data=data, size=shares)
                if order is None:
                    continue

                order.addinfo(reason=self.pending_sell_reasons.get(code, "unknown"))
                cash_available += net_value
                self.pending_sells.discard(code)
                self.pending_sell_reasons.pop(code, None)

        if self.pending_buys and self.pending_buy_slots > 0 and cash_available > 0:
            selected_buys, buy_meta, self.today_selection_stats = self.variant_engine._select_buys(
                today,
                self._dedupe_codes(self.pending_buys),
                self.pending_buy_slots,
            )

            if selected_buys:
                selected_buys = self._dedupe_codes(selected_buys)
                standard_position_value = self.last_nav / base.TARGET_POSITIONS
                allocatable_per_stock = min(cash_available / len(selected_buys), standard_position_value)
                planned_buy_codes = set()

                for code in selected_buys:
                    data = self.data_by_code.get(code)
                    if data is None or not self._data_has_bar_today(data, today):
                        continue

                    if self.filter.is_st(today, code):
                        continue

                    if self.getposition(data).size > 0 or code in planned_buy_codes:
                        continue

                    if self.buy_execution_mode in {"intraday_ivol", "tail_1450", "intraday_attention", "buy_intraday_attention", "intraday_factor3"}:
                        planned_buy_codes.add(code)
                        continue

                    buy_open = self._safe_float(data.open[0])
                    prev_close = self._previous_close(data)
                    if pd.isna(buy_open) or buy_open <= 0:
                        continue

                    limit_pct = base._get_limit_pct(code)
                    if pd.notna(prev_close) and prev_close > 0:
                        if buy_open >= prev_close * (1.0 + limit_pct * 0.95):
                            continue
                        if buy_open <= prev_close * (1.0 - limit_pct * 0.95):
                            continue

                    est_exec_price = buy_open * (1.0 + base.SLIPPAGE_BPS_PER_SIDE / 10000.0)
                    raw_shares = allocatable_per_stock / est_exec_price
                    trade_shares = int(raw_shares // 100) * 100
                    if trade_shares < 100:
                        continue

                    _, trade_value, total_cost = self._estimate_buy_cost(buy_open, trade_shares)
                    if cash_available < total_cost:
                        continue

                    crowding = buy_meta.get(code, {}).get("crowding_pct", np.nan)
                    order = self.buy(data=data, size=trade_shares)
                    if order is None:
                        continue

                    order.addinfo(reason=f"{self.variant_name}(crowding={crowding:.3f})")
                    cash_available -= total_cost
                    planned_buy_codes.add(code)

                if self.buy_execution_mode in {"intraday_ivol", "tail_1450", "intraday_attention", "buy_intraday_attention", "intraday_factor3"} and planned_buy_codes:
                    planned_codes = [code for code in selected_buys if code in planned_buy_codes]
                    self._queue_intraday_buy_plan(today, planned_codes, buy_meta, allocatable_per_stock)

        self.pending_buys = []
        self.pending_buy_slots = 0

    def next(self):
        today = self._current_date()
        if self.buy_execution_mode == "intraday_attention":
            if self.pending_sells or self.pending_intraday_buy_plan:
                self._execute_intraday_attention_day(today)
        elif self.buy_execution_mode == "buy_intraday_attention":
            if self.pending_intraday_buy_plan:
                self._execute_intraday_attention_buys_only(today)
        elif self.buy_execution_mode == "sell_intraday_attention":
            if self.pending_sells:
                self._execute_intraday_attention_sells_only(today)
        elif self.buy_execution_mode == "intraday_factor3":
            if self.pending_sells or self.pending_intraday_buy_plan:
                self._execute_intraday_factor3_day(today)
        elif self.buy_execution_mode in {"intraday_ivol", "tail_1450"} and self.pending_intraday_buy_plan:
            self._execute_intraday_buys(today)

        nav = float(self.broker.getvalue())
        self.last_nav = nav
        self.daily_nav.append((today, nav))

        need_factor_refresh = (today in self.refresh_dates) or (not self.initialized and today >= self.first_factor_date)
        if need_factor_refresh:
            valid_codes = self.filter.filter_universe(today)
            if len(valid_codes) >= 50:
                factor_cache = self.factor_engine.compute_factors(valid_codes, today)
                if len(factor_cache) > 0:
                    self.factor_cache = factor_cache
                    if not self.initialized:
                        self.pending_buy_slots = base.TARGET_POSITIONS
                        self._set_trigger_date(today)
                        buy_candidates = self.trigger_engine.get_buy_candidates(set(), self.pending_buy_slots, self.factor_cache)
                        if buy_candidates:
                            self.pending_buys = self._dedupe_codes(buy_candidates)
                            self.initialized = True

        holding_codes = self._holding_codes()
        if self.initialized and self.factor_cache is not None and holding_codes:
            sell_codes, sell_reasons = self.trigger_engine.check_sell_triggers(holding_codes, today, self.factor_cache)
            if sell_codes:
                self.pending_sells.update(sell_codes)
                self.pending_sell_reasons.update(sell_reasons)

            current_count = len(holding_codes) - len(self.pending_sells)
            n_needed = base.TARGET_POSITIONS - current_count
            if n_needed > 0:
                current_holding_codes = set(holding_codes) - self.pending_sells
                self.pending_buy_slots = n_needed
                self._set_trigger_date(today)
                buy_candidates = self.trigger_engine.get_buy_candidates(current_holding_codes, n_needed, self.factor_cache)
                if buy_candidates:
                    self.pending_buys = self._dedupe_codes(buy_candidates)
                else:
                    self.pending_buy_slots = 0

        self.daily_status.append(
            {
                "date": today.strftime("%Y-%m-%d"),
                "nav": round(nav, 2),
                "n_holdings": len(holding_codes),
                "n_sold": len(self.pending_sells),
                "n_buy_slots": int(self.pending_buy_slots),
                "n_buy_backups": int(len(self.pending_buys)),
                "attention_candidates": int(self.today_selection_stats["candidate_count"]),
                "attention_selected": int(self.today_selection_stats["selected_count"]),
                "attention_blocked": int(self.today_selection_stats["blocked_count"]),
                "attention_avg_selected_crowding": self.today_selection_stats["avg_selected_crowding"],
                "intraday_buy_planned": int(self.last_intraday_buy_stats["planned"]),
                "intraday_buy_executed": int(self.last_intraday_buy_stats["executed"]),
                "intraday_buy_triggered": int(self.last_intraday_buy_stats["triggered"]),
                "intraday_buy_fallback": int(self.last_intraday_buy_stats["fallback"]),
                "intraday_sell_planned": int(self.last_intraday_sell_stats["planned"]),
                "intraday_sell_executed": int(self.last_intraday_sell_stats["executed"]),
                "intraday_sell_early": int(self.last_intraday_sell_stats["early"]),
                "intraday_sell_late": int(self.last_intraday_sell_stats["late"]),
            }
        )

        self.today_selection_stats = self.variant_engine._empty_stats(0, 0)
        self.last_intraday_buy_stats = self._empty_intraday_buy_stats()
        self.last_intraday_sell_stats = self._empty_intraday_sell_stats()

        if self.initialized and not holding_codes and not self.pending_buys and self.factor_cache is not None:
            self.pending_buy_slots = base.TARGET_POSITIONS
            self._set_trigger_date(today)
            buy_candidates = self.trigger_engine.get_buy_candidates(set(), self.pending_buy_slots, self.factor_cache)
            if buy_candidates:
                self.pending_buys = self._dedupe_codes(buy_candidates)
            else:
                self.pending_buy_slots = 0

    def notify_order(self, order):
        if order.status != order.Completed:
            return

        if order.data._name == CLOCK_DATA_NAME:
            return

        exec_dt = bt.num2date(order.executed.dt)
        shares = abs(int(order.executed.size))
        price = float(order.executed.price)
        value = abs(float(order.executed.size) * price)
        action = "BUY" if order.isbuy() else "SELL"

        self.trade_log.append(
            {
                "date": exec_dt.strftime("%Y-%m-%d"),
                "code": order.data._name,
                "action": action,
                "price": round(price, 4),
                "shares": shares,
                "value": round(value, 2),
                "reason": getattr(order.info, "reason", "fill"),
            }
        )

    def stop(self):
        print("\nBacktrader backtest completed.")
        print(f"  Total trades: {len(self.trade_log)}")
        n_sells = sum(1 for row in self.trade_log if row["action"] == "SELL")
        n_buys = sum(1 for row in self.trade_log if row["action"] == "BUY")
        print(f"  Sells: {n_sells}, Buys: {n_buys}")


def parse_args():
    parser = argparse.ArgumentParser(description="Backtrader port of old_main attention penalty 0.12 with capped daily buys")
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
        "--attention-tsmean-window",
        type=int,
        default=DEFAULT_ATTENTION_TSMEAN_WINDOW,
        help="Rolling ts_mean window for attention crowding penalty. Default: 1 (disabled)",
    )
    parser.add_argument(
        "--short-ivol-tsmean-window",
        type=int,
        default=DEFAULT_SHORT_IVOL_TSMEAN_WINDOW,
        help="Rolling ts_mean window for short-ivol sell trigger ratio. Default: 1 (disabled)",
    )
    parser.add_argument(
        "--buy-vol-signal",
        choices=["long", "short", "tech_composite", "ra_approx", "ra_only", "short_preprocessed", "long_short_filter", "attention"],
        default=DEFAULT_BUY_VOL_SIGNAL,
        help="Buy-side signal mode. 'long' keeps the original 252d ivol; 'short' replaces the main ivol filter/ranking with 5d short ivol; 'tech_composite' builds a monthly A-share approximation of the paper's 14-indicator technical composite via SOLS and ranks toward higher expected returns while keeping delta_ivol/cbmom guards; 'ra_approx' uses a density-based approximation of the paper's idiosyncratic return asymmetry factor on A-shares and ranks toward lower RA while keeping delta_ivol/cbmom guards; 'ra_only' lets approximate RA alone generate both buy and sell signals; 'short_preprocessed' uses 5d short ivol after daily winsor/z-score/size+industry neutralization; 'long_short_filter' keeps long ivol as the main signal and only uses short ivol as a secondary low-vol filter; 'attention' removes the main ivol filter and lets delta_ivol + cbmom feed the candidate pool while attention decides the final buys. Default: long",
    )
    parser.add_argument(
        "--universe-index",
        choices=["all", "csi2000_current", "csi300_hist_reconstructed", "csi1000_hist_reconstructed", "csi2000_hist_reconstructed"],
        default=DEFAULT_UNIVERSE_INDEX,
        help="Optional stock-pool restriction. 'csi2000_current' uses the latest available CSIndex CSI 2000 constituent snapshot as a fixed universe across the whole backtest. 'csi300_hist_reconstructed', 'csi1000_hist_reconstructed' and 'csi2000_hist_reconstructed' rebuild time-varying CSI 300 / CSI 1000 / CSI 2000 universes from official methodology plus local amount/turnover data. Default: all",
    )
    parser.add_argument(
        "--buy-short-ivol-filter-pct",
        type=float,
        default=DEFAULT_BUY_SHORT_IVOL_FILTER_PCT,
        help=f"When --buy-vol-signal=long_short_filter, keep the lowest short-ivol bucket inside the primary candidate pool. Default: {DEFAULT_BUY_SHORT_IVOL_FILTER_PCT:.2f}",
    )
    parser.add_argument(
        "--refresh-trading-step",
        type=int,
        default=DEFAULT_REFRESH_TRADING_STEP,
        help="Refresh the main factor cache every N trading days. Use 0 to keep the base calendar rule. Default: 0",
    )
    parser.add_argument(
        "--buy-execution-mode",
        choices=[
            "open",
            "intraday_ivol",
            "tail_1450",
            "intraday_attention",
            "buy_intraday_attention",
            "sell_intraday_attention",
            "intraday_factor3",
        ],
        default=DEFAULT_BUY_EXECUTION_MODE,
        help="Execution mode. 'open' keeps next-open fills; 'intraday_ivol' waits for 15m ivol timing; 'tail_1450' buys at the 14:50 tail proxy; 'intraday_attention' uses early 15m attention for both sides; 'buy_intraday_attention' only times buys; 'sell_intraday_attention' only times sells; 'intraday_factor3' uses real 15m data to build intraday ivol / delta_ivol / cbmom for both buy and sell execution while keeping daily decisions unchanged. Default: open",
    )
    parser.add_argument(
        "--buy-exec-15m-dir",
        default=DEFAULT_BUY_15M_EXEC_DIR,
        help=f"15m unadjusted directory used for intraday buy timing/execution. Default: {DEFAULT_BUY_15M_EXEC_DIR}",
    )
    parser.add_argument(
        "--buy-15m-ivol-lookback-days",
        type=int,
        default=DEFAULT_BUY_15M_IVOL_LOOKBACK_DAYS,
        help=f"Lookback trading days for intraday 15m ivol baseline. Default: {DEFAULT_BUY_15M_IVOL_LOOKBACK_DAYS}",
    )
    parser.add_argument(
        "--buy-15m-ivol-trigger-ratio",
        type=float,
        default=DEFAULT_BUY_15M_IVOL_TRIGGER_RATIO,
        help=f"Trigger buy when current intraday 15m ivol <= baseline * ratio. Default: {DEFAULT_BUY_15M_IVOL_TRIGGER_RATIO}",
    )
    parser.add_argument(
        "--buy-15m-ivol-min-bars",
        type=int,
        default=DEFAULT_BUY_15M_IVOL_MIN_BARS,
        help=f"Earliest number of 15m bars used before evaluating intraday ivol. Default: {DEFAULT_BUY_15M_IVOL_MIN_BARS}",
    )
    parser.add_argument(
        "--buy-15m-fallback-mode",
        choices=["vwap", "twap", "hybrid"],
        default=DEFAULT_BUY_15M_FALLBACK_MODE,
        help=f"Fallback execution proxy on the 1500 bar when no trigger occurs. Default: {DEFAULT_BUY_15M_FALLBACK_MODE}",
    )
    parser.add_argument(
        "--intraday-factor-lookback-days",
        type=int,
        default=DEFAULT_INTRADAY_FACTOR_LOOKBACK_DAYS,
        help=f"Lookback trading days used to build 15m ivol / delta_ivol / cbmom execution snapshots. Default: {DEFAULT_INTRADAY_FACTOR_LOOKBACK_DAYS}",
    )
    parser.add_argument(
        "--intraday-factor-early-bars",
        type=int,
        default=DEFAULT_INTRADAY_FACTOR_EARLY_BARS,
        help=f"Number of completed early 15m returns used to compute intraday ivol / delta_ivol / cbmom. Default: {DEFAULT_INTRADAY_FACTOR_EARLY_BARS}",
    )
    parser.add_argument(
        "--intraday-factor-buy-threshold",
        type=float,
        default=DEFAULT_INTRADAY_FACTOR_BUY_THRESHOLD,
        help=f"Buy early when the intraday ivol / delta_ivol / cbmom composite score >= threshold, else buy at the tail proxy. Default: {DEFAULT_INTRADAY_FACTOR_BUY_THRESHOLD}",
    )
    parser.add_argument(
        "--intraday-factor-sell-threshold",
        type=float,
        default=DEFAULT_INTRADAY_FACTOR_SELL_THRESHOLD,
        help=f"Sell early when the intraday ivol / delta_ivol / cbmom composite score >= threshold, else sell at the tail proxy. Default: {DEFAULT_INTRADAY_FACTOR_SELL_THRESHOLD}",
    )
    parser.add_argument(
        "--intraday-factor-exec-mode",
        choices=["vwap", "twap", "hybrid"],
        default=DEFAULT_INTRADAY_FACTOR_EXEC_MODE,
        help=f"Execution price proxy for intraday factor3 timing. Default: {DEFAULT_INTRADAY_FACTOR_EXEC_MODE}",
    )
    parser.add_argument(
        "--intraday-attention-cache",
        default=DEFAULT_INTRADAY_ATTENTION_CACHE,
        help=f"Path to cached early-session 15m attention timing panel. Default: {DEFAULT_INTRADAY_ATTENTION_CACHE}",
    )
    parser.add_argument(
        "--intraday-attention-early-bars",
        type=int,
        default=DEFAULT_INTRADAY_ATTENTION_EARLY_BARS,
        help=f"Number of completed same-day 15m return bars used to build early attention. Default: {DEFAULT_INTRADAY_ATTENTION_EARLY_BARS}",
    )
    parser.add_argument(
        "--intraday-attention-buy-threshold",
        type=float,
        default=DEFAULT_INTRADAY_ATTENTION_BUY_THRESHOLD,
        help=f"Buy at the 10:20 proxy when early attention crowding <= threshold, else 14:50. Default: {DEFAULT_INTRADAY_ATTENTION_BUY_THRESHOLD}",
    )
    parser.add_argument(
        "--intraday-attention-sell-threshold",
        type=float,
        default=DEFAULT_INTRADAY_ATTENTION_SELL_THRESHOLD,
        help=f"Sell at the 10:20 proxy when early attention crowding >= threshold, else 14:50. Default: {DEFAULT_INTRADAY_ATTENTION_SELL_THRESHOLD}",
    )
    parser.add_argument(
        "--intraday-attention-exec-mode",
        choices=["vwap", "twap", "hybrid"],
        default=DEFAULT_INTRADAY_ATTENTION_EXEC_MODE,
        help=f"Execution price proxy for intraday attention timing. Default: {DEFAULT_INTRADAY_ATTENTION_EXEC_MODE}",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix used for output files. Default: auto-generated from parameters.",
    )
    return parser.parse_args()


def resolve_output_prefix(prefix):
    if os.path.isabs(prefix):
        output_prefix = prefix
    else:
        output_prefix = os.path.join(BASE_DIR, prefix)

    parent = os.path.dirname(output_prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return output_prefix


def auto_output_prefix(args):
    if args.output_prefix:
        return args.output_prefix

    parts = ["backtrader_old_main_attention_penalty012"]
    if str(args.buy_vol_signal).lower() == "short":
        parts.append("buyshortivol")
    elif str(args.buy_vol_signal).lower() == "tech_composite":
        parts.append("buytechcomposite")
    elif str(args.buy_vol_signal).lower() == "ra_only":
        parts.append("raonly")
    elif str(args.buy_vol_signal).lower() == "ra_approx":
        parts.append("buyraapprox")
    elif str(args.buy_vol_signal).lower() == "short_preprocessed":
        parts.append("buyshortivolprep")
    elif str(args.buy_vol_signal).lower() == "long_short_filter":
        parts.append(f"longshortfilter{int(round(float(args.buy_short_ivol_filter_pct) * 100))}")
    elif str(args.buy_vol_signal).lower() == "attention":
        parts.append("buyattention")
    if str(args.universe_index).lower() == "csi2000_current":
        parts.append("csi2000current")
    elif str(args.universe_index).lower() == "csi300_hist_reconstructed":
        parts.append("csi300hist")
    elif str(args.universe_index).lower() == "csi1000_hist_reconstructed":
        parts.append("csi1000hist")
    elif str(args.universe_index).lower() == "csi2000_hist_reconstructed":
        parts.append("csi2000hist")
    if int(args.refresh_trading_step) > 0:
        parts.append(f"refreshstep{int(args.refresh_trading_step)}")
    if args.attention_tsmean_window > 1 or args.short_ivol_tsmean_window > 1:
        parts.append(f"atttsmean{int(args.attention_tsmean_window)}")
        parts.append(f"ivoltsmean{int(args.short_ivol_tsmean_window)}")
    if args.buy_execution_mode == "intraday_ivol":
        parts.append(
            "buy15mivol"
            f"{int(round(float(args.buy_15m_ivol_trigger_ratio) * 100)):02d}"
            f"lb{int(args.buy_15m_ivol_lookback_days)}"
            f"mb{int(args.buy_15m_ivol_min_bars)}"
        )
    elif args.buy_execution_mode == "tail_1450":
        parts.append(f"buy1450tail{str(args.buy_15m_fallback_mode)}")
    elif args.buy_execution_mode == "intraday_attention":
        parts.append(
            "buysell15matt"
            f"b{int(round(float(args.intraday_attention_buy_threshold) * 100)):02d}"
            f"s{int(round(float(args.intraday_attention_sell_threshold) * 100)):02d}"
            f"e{int(args.intraday_attention_early_bars)}"
        )
    elif args.buy_execution_mode == "buy_intraday_attention":
        parts.append(
            "buy15matt"
            f"b{int(round(float(args.intraday_attention_buy_threshold) * 100)):02d}"
            f"e{int(args.intraday_attention_early_bars)}"
        )
    elif args.buy_execution_mode == "sell_intraday_attention":
        parts.append(
            "sell15matt"
            f"s{int(round(float(args.intraday_attention_sell_threshold) * 100)):02d}"
            f"e{int(args.intraday_attention_early_bars)}"
        )
    elif args.buy_execution_mode == "intraday_factor3":
        parts.append(
            "factor15m"
            f"b{int(round(float(args.intraday_factor_buy_threshold) * 100)):02d}"
            f"s{int(round(float(args.intraday_factor_sell_threshold) * 100)):02d}"
            f"e{int(args.intraday_factor_early_bars)}"
            f"lb{int(args.intraday_factor_lookback_days)}"
        )
    parts.append(f"maxbuy{int(args.max_new_buys)}")
    parts.append("full")
    return "_".join(parts) if len(parts) > 3 else DEFAULT_OUTPUT_PREFIX


def build_calendar_dates(df, clip_to_window=True):
    dates = sorted(df["date"].unique())
    calendar = [pd.Timestamp(d) for d in dates]
    if not clip_to_window:
        return calendar
    return [pd.Timestamp(d) for d in calendar if pd.Timestamp(base.START_DATE) <= pd.Timestamp(d) <= pd.Timestamp(base.END_DATE)]


def build_calendar_feed(calendar_dates):
    calendar_df = pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 0.0,
            "openinterest": 0.0,
        },
        index=pd.DatetimeIndex(calendar_dates),
    )
    return bt.feeds.PandasData(dataname=calendar_df, timeframe=bt.TimeFrame.Days)


def build_stock_feed(raw_group):
    frame = raw_group.loc[:, ["date", "open", "close", "turnover"]].copy()
    frame["high"] = frame[["open", "close"]].max(axis=1)
    frame["low"] = frame[["open", "close"]].min(axis=1)
    frame["volume"] = frame["turnover"].fillna(0.0)
    frame["openinterest"] = 0.0
    frame = frame[["date", "open", "high", "low", "close", "volume", "openinterest"]]
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame = frame[(frame["date"] >= pd.Timestamp(base.START_DATE)) & (frame["date"] <= pd.Timestamp(base.END_DATE))]
    if frame.empty:
        return None

    frame = frame.set_index("date")
    return bt.feeds.PandasData(dataname=frame, timeframe=bt.TimeFrame.Days)


def plot_nav(nav_df, output_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(nav_df["date"], nav_df["nav"] / base.INITIAL_CAPITAL, linewidth=1.6, color="#ff7f0e")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_title(
        "Backtrader old_main + attention_penalty(0.12) + capped daily new buys\n"
        f"{base.START_DATE} ~ {base.END_DATE}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("NAV (start=1.0)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_turnover_metrics(nav_df, trade_log):
    if nav_df.empty or not trade_log:
        return {}

    trades_df = pd.DataFrame(trade_log)
    trades_df["value"] = pd.to_numeric(trades_df["value"], errors="coerce")
    trades_df = trades_df.dropna(subset=["value"])
    if trades_df.empty:
        return {}

    avg_nav = float(nav_df["nav"].mean())
    calendar_years = (nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days / 365.25
    trading_years = len(nav_df) / 252.0
    total_trade_value = float(trades_df["value"].sum())
    buy_value = float(trades_df.loc[trades_df["action"] == "BUY", "value"].sum())
    sell_value = float(trades_df.loc[trades_df["action"] == "SELL", "value"].sum())
    one_side_turnover = total_trade_value / 2.0 / avg_nav if avg_nav > 0 else np.nan
    two_side_turnover = total_trade_value / avg_nav if avg_nav > 0 else np.nan

    return {
        "avg_nav": avg_nav,
        "total_trade_value": total_trade_value,
        "buy_value": buy_value,
        "sell_value": sell_value,
        "one_side_turnover_total": one_side_turnover,
        "two_side_turnover_total": two_side_turnover,
        "sell_over_avg_nav_total": sell_value / avg_nav if avg_nav > 0 else np.nan,
        "buy_over_avg_nav_total": buy_value / avg_nav if avg_nav > 0 else np.nan,
        "annualized_one_side_turnover_calendar": one_side_turnover / calendar_years if calendar_years > 0 else np.nan,
        "annualized_two_side_turnover_calendar": two_side_turnover / calendar_years if calendar_years > 0 else np.nan,
        "annualized_sell_over_avg_nav_calendar": (sell_value / avg_nav) / calendar_years
        if avg_nav > 0 and calendar_years > 0
        else np.nan,
        "annualized_buy_over_avg_nav_calendar": (buy_value / avg_nav) / calendar_years
        if avg_nav > 0 and calendar_years > 0
        else np.nan,
        "annualized_one_side_turnover_trading": one_side_turnover / trading_years if trading_years > 0 else np.nan,
        "annualized_two_side_turnover_trading": two_side_turnover / trading_years if trading_years > 0 else np.nan,
    }


def main():
    args = parse_args()
    output_prefix = resolve_output_prefix(auto_output_prefix(args))
    universe_info = None

    base.set_backtest_window(args.start, args.end)
    base.set_target_positions(args.positions)
    base.set_slippage_bps(args.slippage_bps)

    t0 = time.time()
    print("=" * 60)
    print("  Backtrader event-driven backtest")
    print("  old_main + attention_penalty(0.12) + max_new_buys")
    print(f"  Window: {base.START_DATE} ~ {base.END_DATE}")
    print(f"  Target positions: {base.TARGET_POSITIONS}")
    print(f"  Penalty weight: {args.penalty_weight:.2f}")
    print(f"  Max new buys/day: {args.max_new_buys}")
    print(f"  Attention ts_mean window: {args.attention_tsmean_window}")
    print(f"  Short-ivol ts_mean window: {args.short_ivol_tsmean_window}")
    print(f"  Buy vol signal: {args.buy_vol_signal}")
    print(f"  Universe index: {args.universe_index}")
    if str(args.buy_vol_signal).lower() == "long_short_filter":
        print(f"  Buy short-ivol filter pct: {args.buy_short_ivol_filter_pct:.2f}")
    if int(args.refresh_trading_step) > 0:
        print(f"  Refresh trading step: {args.refresh_trading_step} trading days")
    else:
        print(f"  Refresh calendar rule: {base.FACTOR_REFRESH_FREQ}")
    print(f"  Buy execution mode: {args.buy_execution_mode}")
    if args.buy_execution_mode in {"intraday_ivol", "tail_1450"}:
        print(f"  Buy 15m dir (unadjusted): {os.path.abspath(args.buy_exec_15m_dir)}")
        print(f"  Buy fallback mode: {args.buy_15m_fallback_mode} (1500 bar proxy)")
    if args.buy_execution_mode == "intraday_ivol":
        print(f"  Buy 15m ivol lookback days: {args.buy_15m_ivol_lookback_days}")
        print(f"  Buy 15m ivol trigger ratio: {args.buy_15m_ivol_trigger_ratio:.2f}")
        print(f"  Buy 15m ivol min bars: {args.buy_15m_ivol_min_bars}")
    if args.buy_execution_mode == "intraday_factor3":
        print(f"  Intraday factor3 exec dir (unadjusted): {os.path.abspath(args.buy_exec_15m_dir)}")
        print(f"  Intraday factor3 lookback days: {args.intraday_factor_lookback_days}")
        print(f"  Intraday factor3 early bars: {args.intraday_factor_early_bars}")
        print(f"  Intraday factor3 buy threshold: {args.intraday_factor_buy_threshold:.2f}")
        print(f"  Intraday factor3 sell threshold: {args.intraday_factor_sell_threshold:.2f}")
        print(f"  Intraday factor3 exec mode: {args.intraday_factor_exec_mode}")
    if args.buy_execution_mode in {"intraday_attention", "buy_intraday_attention", "sell_intraday_attention"}:
        print(f"  Intraday attention signal dir: {os.path.abspath(args.intraday_dir)}")
        print(f"  Intraday attention exec dir (unadjusted): {os.path.abspath(args.buy_exec_15m_dir)}")
        print(f"  Intraday attention early bars: {args.intraday_attention_early_bars}")
        if args.buy_execution_mode == "intraday_attention":
            print(f"  Intraday attention buy threshold: {args.intraday_attention_buy_threshold:.2f}")
            print(f"  Intraday attention sell threshold: {args.intraday_attention_sell_threshold:.2f}")
        elif args.buy_execution_mode == "buy_intraday_attention":
            print(f"  Intraday attention buy threshold: {args.intraday_attention_buy_threshold:.2f}")
            print("  Intraday attention sells: next-open (disabled for intraday timing)")
        else:
            print("  Intraday attention buys: next-open (disabled for intraday timing)")
            print(f"  Intraday attention sell threshold: {args.intraday_attention_sell_threshold:.2f}")
        print(f"  Intraday attention exec mode: {args.intraday_attention_exec_mode}")
    print("=" * 60)

    loader = base.DataLoader()
    loader.load_stock_data()
    loader.load_ff3_factors()
    loader.compute_returns()
    base_filter = base.UniverseFilter(loader.df_hfq, loader.df_raw)
    universe_info = None

    if str(args.universe_index).lower() == "csi2000_current":
        universe_info = fetch_csindex_constituent_snapshot(symbol="932000")
        print(
            f"  Universe snapshot loaded: CSI 2000 current | "
            f"date={universe_info['snapshot_date']} | codes={len(universe_info['codes'])}"
        )
        print(f"  Universe cache: {universe_info['cache_path']}")
    elif str(args.universe_index).lower() == "csi300_hist_reconstructed":
        universe_info = build_csi300_reconstructed_panels(
            calendar_dates=build_calendar_dates(loader.df_hfq, clip_to_window=False),
            base_filter=base_filter,
            daily_dir=os.path.join(BASE_DIR, "data_stock_daily_unadj"),
            intraday_dir=args.buy_exec_15m_dir,
            start_date=args.start,
            end_date=args.end,
            effective_panel_path=DEFAULT_CSI300_RECON_EFFECTIVE_PANEL,
            monthly_panel_path=DEFAULT_CSI300_RECON_MONTHLY_PANEL,
            metric_cache_path=DEFAULT_CSI300_RECON_METRIC_CACHE,
            max_workers=min(8, max(os.cpu_count() or 4, 1)),
        )
        print(
            f"  Universe panel loaded: {universe_info['name']} reconstructed history | "
            f"launch_effective={universe_info['launch_effective_date']} | "
            f"snapshots={len(universe_info['effective_entries'])}"
        )
        print(f"  Universe effective panel: {universe_info['effective_panel_path']}")
        print(f"  Universe monthly panel: {universe_info['monthly_panel_path']}")
    elif str(args.universe_index).lower() == "csi1000_hist_reconstructed":
        universe_info = build_csi1000_reconstructed_panels(
            calendar_dates=build_calendar_dates(loader.df_hfq, clip_to_window=False),
            base_filter=base_filter,
            daily_dir=os.path.join(BASE_DIR, "data_stock_daily_unadj"),
            intraday_dir=args.buy_exec_15m_dir,
            start_date=args.start,
            end_date=args.end,
            effective_panel_path=DEFAULT_CSI1000_RECON_EFFECTIVE_PANEL,
            monthly_panel_path=DEFAULT_CSI1000_RECON_MONTHLY_PANEL,
            metric_cache_path=DEFAULT_CSI1000_RECON_METRIC_CACHE,
            max_workers=min(8, max(os.cpu_count() or 4, 1)),
        )
        print(
            f"  Universe panel loaded: {universe_info['name']} reconstructed history | "
            f"launch_effective={universe_info['launch_effective_date']} | "
            f"snapshots={len(universe_info['effective_entries'])}"
        )
        print(f"  Universe effective panel: {universe_info['effective_panel_path']}")
        print(f"  Universe monthly panel: {universe_info['monthly_panel_path']}")
    elif str(args.universe_index).lower() == "csi2000_hist_reconstructed":
        universe_info = build_csi2000_reconstructed_panels(
            calendar_dates=build_calendar_dates(loader.df_hfq, clip_to_window=False),
            base_filter=base_filter,
            daily_dir=os.path.join(BASE_DIR, "data_stock_daily_unadj"),
            intraday_dir=args.buy_exec_15m_dir,
            start_date=max(pd.Timestamp(args.start), pd.Timestamp("2023-08-11")),
            end_date=args.end,
            effective_panel_path=DEFAULT_CSI2000_RECON_EFFECTIVE_PANEL,
            monthly_panel_path=DEFAULT_CSI2000_RECON_MONTHLY_PANEL,
            metric_cache_path=DEFAULT_CSI2000_RECON_METRIC_CACHE,
            max_workers=min(8, max(os.cpu_count() or 4, 1)),
        )
        print(
            f"  Universe panel loaded: {universe_info['name']} reconstructed history | "
            f"launch_effective={universe_info['launch_effective_date']} | "
            f"snapshots={len(universe_info['effective_entries'])}"
        )
        print(f"  Universe effective panel: {universe_info['effective_panel_path']}")
        print(f"  Universe monthly panel: {universe_info['monthly_panel_path']}")

    calendar_dates = build_calendar_dates(loader.df_hfq)
    if not calendar_dates:
        raise RuntimeError("No data available inside the requested backtest window.")

    attention_panel = build_attention_panel(
        cache_path=args.attention_cache,
        start_date=base.START_DATE,
        end_date=base.END_DATE,
        intraday_dir=args.intraday_dir,
    )
    intraday_attention_panel = None
    if args.buy_execution_mode in {"intraday_attention", "buy_intraday_attention", "sell_intraday_attention"}:
        intraday_attention_panel = build_intraday_attention_timing_panel(
            cache_path=args.intraday_attention_cache,
            calendar_dates=calendar_dates,
            signal_dir=args.intraday_dir,
            exec_dir=args.buy_exec_15m_dir,
            early_bars=args.intraday_attention_early_bars,
        )

    date_series = pd.Series(calendar_dates)
    if int(args.refresh_trading_step) > 0:
        refresh_dates = set(date_series.iloc[:: int(args.refresh_trading_step)].tolist())
    else:
        refresh_dates = set(date_series.groupby(date_series.dt.to_period(base.FACTOR_REFRESH_FREQ)).max().tolist())
    first_factor_date = next(iter(sorted(refresh_dates))) if refresh_dates else calendar_dates[0]
    last_data_date = loader.df_raw.dropna(subset=["close"]).groupby("code")["date"].max().to_dict()

    cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
    cerebro.broker.setcash(base.INITIAL_CAPITAL)
    cerebro.broker.addcommissioninfo(ChinaAStockCommInfo())
    cerebro.broker.set_slippage_perc(
        perc=base.SLIPPAGE_BPS_PER_SIDE / 10000.0,
        slip_open=True,
        slip_limit=True,
        slip_match=True,
        slip_out=True,
    )

    cerebro.adddata(build_calendar_feed(calendar_dates), name=CLOCK_DATA_NAME)

    raw_groups = loader.df_raw.groupby("code", sort=True)
    data_feed_count = 0
    for code, raw_group in raw_groups:
        feed = build_stock_feed(raw_group)
        if feed is None:
            continue
        cerebro.adddata(feed, name=code)
        data_feed_count += 1
        if data_feed_count % 500 == 0:
            print(f"  Added {data_feed_count} stock feeds...")

    print(f"Total stock feeds added: {data_feed_count}")

    universe_codes = tuple() if universe_info is None else tuple(universe_info.get("codes", tuple()))
    universe_effective_entries = tuple() if universe_info is None else tuple(universe_info.get("effective_entries", tuple()))
    universe_snapshot_date = None if universe_info is None else universe_info.get("snapshot_date")
    universe_cache_path = None if universe_info is None else universe_info.get("cache_path")
    universe_effective_panel_path = None if universe_info is None else universe_info.get("effective_panel_path")
    universe_monthly_panel_path = None if universe_info is None else universe_info.get("monthly_panel_path")

    cerebro.addstrategy(
        AttentionPenaltyBacktraderStrategy,
        loader=loader,
        attention_panel=attention_panel,
        penalty_weight=args.penalty_weight,
        max_new_buys=args.max_new_buys,
        backup_multiple=args.backup_multiple,
        backup_extra=args.backup_extra,
        attention_tsmean_window=args.attention_tsmean_window,
        short_ivol_tsmean_window=args.short_ivol_tsmean_window,
        buy_vol_signal=args.buy_vol_signal,
        buy_short_ivol_filter_pct=args.buy_short_ivol_filter_pct,
        buy_execution_mode=args.buy_execution_mode,
        buy_exec_15m_dir=args.buy_exec_15m_dir,
        buy_15m_ivol_lookback_days=args.buy_15m_ivol_lookback_days,
        buy_15m_ivol_trigger_ratio=args.buy_15m_ivol_trigger_ratio,
        buy_15m_ivol_min_bars=args.buy_15m_ivol_min_bars,
        buy_15m_fallback_mode=args.buy_15m_fallback_mode,
        intraday_factor_lookback_days=args.intraday_factor_lookback_days,
        intraday_factor_early_bars=args.intraday_factor_early_bars,
        intraday_factor_buy_threshold=args.intraday_factor_buy_threshold,
        intraday_factor_sell_threshold=args.intraday_factor_sell_threshold,
        intraday_factor_exec_mode=args.intraday_factor_exec_mode,
        intraday_attention_panel=intraday_attention_panel,
        intraday_attention_early_bars=args.intraday_attention_early_bars,
        intraday_attention_buy_threshold=args.intraday_attention_buy_threshold,
        intraday_attention_sell_threshold=args.intraday_attention_sell_threshold,
        intraday_attention_exec_mode=args.intraday_attention_exec_mode,
        refresh_dates=sorted(refresh_dates),
        first_factor_date=first_factor_date,
        last_data_date=last_data_date,
        universe_index=args.universe_index,
        universe_codes=universe_codes,
        universe_effective_entries=universe_effective_entries,
        universe_snapshot_date=None if universe_snapshot_date is None else str(universe_snapshot_date),
        universe_cache_path=None if universe_cache_path is None else str(universe_cache_path),
        universe_effective_panel_path=universe_effective_panel_path,
        universe_monthly_panel_path=universe_monthly_panel_path,
    )

    strategies = cerebro.run(runonce=False, preload=True)
    strategy = strategies[0]

    nav_df = pd.DataFrame(strategy.daily_nav, columns=["date", "nav"])
    summary = compute_summary(strategy.daily_nav, strategy.trade_log)
    turnover = compute_turnover_metrics(nav_df, strategy.trade_log)

    nav_path = f"{output_prefix}_nav.csv"
    trades_path = f"{output_prefix}_trades.csv"
    status_path = f"{output_prefix}_daily_status.csv"
    plot_path = f"{output_prefix}_nav.png"
    summary_path = f"{output_prefix}_summary.json"

    if not nav_df.empty:
        nav_df.to_csv(nav_path, index=False, encoding="utf-8-sig")
        plot_nav(nav_df, plot_path)
        print(f"NAV exported: {nav_path}")
        print(f"NAV plot exported: {plot_path}")

    if strategy.trade_log:
        pd.DataFrame(strategy.trade_log).to_csv(trades_path, index=False, encoding="utf-8-sig")
        print(f"Trade log exported: {trades_path} ({len(strategy.trade_log)} rows)")

    if strategy.daily_status:
        pd.DataFrame(strategy.daily_status).to_csv(status_path, index=False, encoding="utf-8-sig")
        print(f"Daily status exported: {status_path}")

    payload = {
        "engine": "backtrader",
        "strategy": strategy.variant_name,
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
            "attention_tsmean_window": int(args.attention_tsmean_window),
            "short_ivol_tsmean_window": int(args.short_ivol_tsmean_window),
            "buy_vol_signal": str(args.buy_vol_signal),
            "universe_index": str(args.universe_index),
            "buy_short_ivol_filter_pct": float(args.buy_short_ivol_filter_pct),
            "refresh_trading_step": int(args.refresh_trading_step),
            "refresh_rule": None if int(args.refresh_trading_step) > 0 else str(base.FACTOR_REFRESH_FREQ),
            "buy_execution_mode": str(args.buy_execution_mode),
            "buy_exec_15m_dir": os.path.abspath(args.buy_exec_15m_dir),
            "buy_15m_ivol_lookback_days": int(args.buy_15m_ivol_lookback_days),
            "buy_15m_ivol_trigger_ratio": float(args.buy_15m_ivol_trigger_ratio),
            "buy_15m_ivol_min_bars": int(args.buy_15m_ivol_min_bars),
            "buy_15m_fallback_mode": str(args.buy_15m_fallback_mode),
            "intraday_factor_lookback_days": int(args.intraday_factor_lookback_days),
            "intraday_factor_early_bars": int(args.intraday_factor_early_bars),
            "intraday_factor_buy_threshold": float(args.intraday_factor_buy_threshold),
            "intraday_factor_sell_threshold": float(args.intraday_factor_sell_threshold),
            "intraday_factor_exec_mode": str(args.intraday_factor_exec_mode),
            "intraday_attention_cache": os.path.abspath(args.intraday_attention_cache),
            "intraday_attention_early_bars": int(args.intraday_attention_early_bars),
            "intraday_attention_buy_threshold": float(args.intraday_attention_buy_threshold),
            "intraday_attention_sell_threshold": float(args.intraday_attention_sell_threshold),
            "intraday_attention_exec_mode": str(args.intraday_attention_exec_mode),
            "stock_feeds": int(data_feed_count),
        },
        "attention_dates": {
            "start": attention_panel.dates.min().strftime("%Y-%m-%d"),
            "end": attention_panel.dates.max().strftime("%Y-%m-%d"),
        },
        "summary": summary,
        "turnover": turnover,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    if universe_info is not None:
        if str(universe_info.get("panel_type", "")).lower() == "reconstructed":
            payload["universe_panel"] = {
                "type": str(args.universe_index),
                "name": str(universe_info.get("name")),
                "index_code": str(universe_info.get("index_code")),
                "member_key": str(universe_info.get("member_key")),
                "launch_effective_date": None
                if universe_info.get("launch_effective_date") is None
                else str(universe_info.get("launch_effective_date")),
                "effective_snapshot_count": int(len(universe_info.get("effective_entries", tuple()))),
                "effective_panel_path": str(universe_info.get("effective_panel_path")),
                "monthly_panel_path": str(universe_info.get("monthly_panel_path")),
                "metric_cache_path": None
                if universe_info.get("metric_cache_path") is None
                else str(universe_info.get("metric_cache_path")),
                "review_events": list(universe_info.get("review_events", [])),
            }
        else:
            payload["universe_snapshot"] = {
                "symbol": str(universe_info["symbol"]),
                "snapshot_date": str(universe_info["snapshot_date"]),
                "count": int(len(universe_info["codes"])),
                "cache_path": str(universe_info["cache_path"]),
            }

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Summary exported: {summary_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
