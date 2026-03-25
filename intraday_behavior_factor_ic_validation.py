import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
INTRADAY_DIR = BASE_DIR / "data_stock_15m"
DAILY_RAW_DIR = BASE_DIR / "data_stock_daily_unadj"
DAILY_HFQ_DIR = BASE_DIR / "data_stock_daily_hfq"

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-03-18"
DEFAULT_OUTPUT_PREFIX = "intraday_behavior_factor_ic"
HORIZONS = (1, 3, 5)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate daily behavior factors derived from A-share 15m data using IC / Rank IC."
    )
    parser.add_argument("--start", default=DEFAULT_START, help=f"Start date. Default: {DEFAULT_START}")
    parser.add_argument("--end", default=DEFAULT_END, help=f"End date. Default: {DEFAULT_END}")
    parser.add_argument(
        "--intraday-dir",
        default=str(INTRADAY_DIR),
        help=f"15m raw directory. Default: {INTRADAY_DIR}",
    )
    parser.add_argument(
        "--daily-raw-dir",
        default=str(DAILY_RAW_DIR),
        help=f"Daily raw directory. Default: {DAILY_RAW_DIR}",
    )
    parser.add_argument(
        "--daily-hfq-dir",
        default=str(DAILY_HFQ_DIR),
        help=f"Daily HFQ directory. Default: {DAILY_HFQ_DIR}",
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Output prefix. Default: {DEFAULT_OUTPUT_PREFIX}",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional debug limit on number of stock files to process. Default: 0 (all files)",
    )
    return parser.parse_args()


def is_equity_code(code):
    return len(code) == 6 and code.isdigit() and not code.startswith(("39", "88"))


def summarize_series(series, label):
    s = pd.Series(series, dtype=float).dropna()
    if len(s) == 0:
        return {
            "metric": label,
            "n_dates": 0,
            "mean": np.nan,
            "std": np.nan,
            "tstat": np.nan,
            "icir": np.nan,
            "positive_ratio": np.nan,
        }

    mean = float(s.mean())
    std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    tstat = mean / (std / np.sqrt(len(s))) if len(s) > 1 and std > 0 else np.nan
    icir = mean / std if len(s) > 1 and std > 0 else np.nan
    return {
        "metric": label,
        "n_dates": int(len(s)),
        "mean": mean,
        "std": std if np.isfinite(std) else np.nan,
        "tstat": float(tstat) if np.isfinite(tstat) else np.nan,
        "icir": float(icir) if np.isfinite(icir) else np.nan,
        "positive_ratio": float((s > 0).mean()),
    }


def safe_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return np.nan, int(mask.sum())
    return float(np.corrcoef(x[mask], y[mask])[0, 1]), int(mask.sum())


def safe_rank_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return np.nan
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(xr, yr)[0, 1])


def load_master_intraday_calendar(first_file):
    df = pd.read_csv(first_file, encoding="utf-8-sig", usecols=[0])
    ts = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(np.int64).to_numpy()
    ts = np.unique(ts)
    day_int = ts // 1000000
    unique_days = np.unique(day_int)
    return ts, day_int, unique_days


def build_intraday_behavior_matrices(files, master_ts, master_day_int, master_days):
    n_rows = len(master_ts)
    n_days = len(master_days)
    n_codes = len(files)

    row_day_idx = np.searchsorted(master_days, master_day_int)
    returns_mat = np.full((n_rows, n_codes), np.nan, dtype=np.float32)
    flow_mat = np.full((n_days, n_codes), np.nan, dtype=np.float32)
    valid_bar_mat = np.zeros((n_days, n_codes), dtype=np.uint8)

    codes = []
    for col_idx, path in enumerate(tqdm(files, desc="Loading 15m behavior inputs")):
        code = Path(path).stem
        codes.append(code)

        try:
            df = pd.read_csv(path, encoding="utf-8-sig", usecols=[0, 1, 2, 3, 4, 6])
        except Exception:
            continue

        if df.shape[1] < 6:
            continue

        ts = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=np.float64)
        open_ = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=np.float64)
        high = pd.to_numeric(df.iloc[:, 2], errors="coerce").to_numpy(dtype=np.float64)
        low = pd.to_numeric(df.iloc[:, 3], errors="coerce").to_numpy(dtype=np.float64)
        close = pd.to_numeric(df.iloc[:, 4], errors="coerce").to_numpy(dtype=np.float64)
        amount = pd.to_numeric(df.iloc[:, 5], errors="coerce").to_numpy(dtype=np.float64)

        base_mask = np.isfinite(ts)
        if not base_mask.any():
            continue

        ts = ts[base_mask].astype(np.int64, copy=False)
        open_ = open_[base_mask]
        high = high[base_mask]
        low = low[base_mask]
        close = close[base_mask]
        amount = amount[base_mask]

        order = np.argsort(ts, kind="mergesort")
        ts = ts[order]
        open_ = open_[order]
        high = high[order]
        low = low[order]
        close = close[order]
        amount = amount[order]

        day_int = ts // 1000000
        prev_close = np.empty_like(close)
        prev_close[:] = np.nan
        prev_close[1:] = close[:-1]
        same_day = np.zeros(len(day_int), dtype=bool)
        same_day[1:] = day_int[1:] == day_int[:-1]

        ret = np.full(len(close), np.nan, dtype=np.float32)
        valid_ret = same_day & np.isfinite(prev_close) & (prev_close > 0) & np.isfinite(close) & (close > 0)
        ret[valid_ret] = np.log(close[valid_ret] / prev_close[valid_ret]).astype(np.float32)

        hl_range = high - low
        clv = np.zeros(len(close), dtype=np.float64)
        valid_range = np.isfinite(hl_range) & (np.abs(hl_range) > 1e-12)
        clv[valid_range] = (
            ((close[valid_range] - low[valid_range]) - (high[valid_range] - close[valid_range])) / hl_range[valid_range]
        )

        pos = np.searchsorted(master_ts, ts)
        matched = pos < len(master_ts)
        if matched.any():
            matched_idx = np.flatnonzero(matched)
            matched[matched_idx] = master_ts[pos[matched_idx]] == ts[matched_idx]
        if matched.any():
            returns_mat[pos[matched], col_idx] = ret[matched]

        daily_df = pd.DataFrame(
            {
                "day": day_int,
                "signed_amount": clv * np.nan_to_num(amount, nan=0.0),
                "amount": np.nan_to_num(amount, nan=0.0),
                "valid_ret": np.isfinite(ret).astype(np.int16),
            }
        )
        agg = daily_df.groupby("day", sort=False).agg(
            signed_amount=("signed_amount", "sum"),
            amount=("amount", "sum"),
            valid_ret=("valid_ret", "sum"),
        )
        agg_days = agg.index.to_numpy(dtype=np.int64)
        day_pos = np.searchsorted(master_days, agg_days)
        matched_days = day_pos < len(master_days)
        if matched_days.any():
            matched_day_idx = np.flatnonzero(matched_days)
            matched_days[matched_day_idx] = master_days[day_pos[matched_day_idx]] == agg_days[matched_day_idx]
        if matched_days.any():
            amt = agg["amount"].to_numpy(dtype=np.float64)[matched_days]
            signed = agg["signed_amount"].to_numpy(dtype=np.float64)[matched_days]
            flow = np.full(len(amt), np.nan, dtype=np.float32)
            valid_amt = amt > 0
            flow[valid_amt] = (signed[valid_amt] / amt[valid_amt]).astype(np.float32)
            flow_mat[day_pos[matched_days], col_idx] = flow
            valid_bar_mat[day_pos[matched_days], col_idx] = agg["valid_ret"].to_numpy(dtype=np.int64)[matched_days].clip(
                0, 255
            ).astype(np.uint8)

    top_hits = np.zeros((n_days, n_codes), dtype=np.uint8)
    low_hits = np.zeros((n_days, n_codes), dtype=np.uint8)

    for row_idx in tqdm(range(n_rows), desc="Ranking intraday returns into top/bottom 5%"):
        values = returns_mat[row_idx]
        mask = np.isfinite(values)
        n_valid = int(mask.sum())
        if n_valid < 20:
            continue

        k = max(int(np.floor(n_valid * 0.05 + 1e-12)), 1)
        valid_idx = np.flatnonzero(mask)
        valid_vals = values[mask]

        top_local = np.argpartition(valid_vals, n_valid - k)[-k:]
        low_local = np.argpartition(valid_vals, k - 1)[:k]
        day_idx = row_day_idx[row_idx]
        top_hits[day_idx, valid_idx[top_local]] += 1
        low_hits[day_idx, valid_idx[low_local]] += 1

    attention_up = np.full((n_days, n_codes), np.nan, dtype=np.float32)
    attention_down = np.full((n_days, n_codes), np.nan, dtype=np.float32)
    valid_mask = valid_bar_mat > 0
    attention_up[valid_mask] = (top_hits[valid_mask] / valid_bar_mat[valid_mask]).astype(np.float32)
    attention_down[valid_mask] = (low_hits[valid_mask] / valid_bar_mat[valid_mask]).astype(np.float32)

    return {
        "codes": np.array(codes, dtype="U6"),
        "flow": flow_mat,
        "attention_up": attention_up,
        "attention_down": attention_down,
        "valid_bars": valid_bar_mat,
    }


def build_overnight_matrix(codes, factor_days, daily_raw_dir):
    n_days = len(factor_days)
    n_codes = len(codes)
    day_int = factor_days.strftime("%Y%m%d").astype(np.int64)
    overnight = np.full((n_days, n_codes), np.nan, dtype=np.float32)

    code_to_idx = {code: idx for idx, code in enumerate(codes)}
    for path in tqdm(glob.glob(os.path.join(daily_raw_dir, "*.csv")), desc="Loading daily overnight factor"):
        code = Path(path).stem
        if code not in code_to_idx:
            continue

        try:
            df = pd.read_csv(path, encoding="utf-8-sig", usecols=[0, 1, 2])
        except Exception:
            continue

        if df.shape[1] < 3:
            continue

        df = df.iloc[:, :3].copy()
        df.columns = ["date", "close", "open"]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        if df.empty:
            continue

        prev_close = df["close"].shift(1)
        ovnt = np.where(
            np.isfinite(df["open"]) & np.isfinite(prev_close) & (prev_close > 0),
            df["open"] / prev_close - 1.0,
            np.nan,
        )
        cur_days = df["date"].dt.strftime("%Y%m%d").astype(np.int64).to_numpy()
        pos = np.searchsorted(day_int, cur_days)
        matched = pos < len(day_int)
        if matched.any():
            matched_idx = np.flatnonzero(matched)
            matched[matched_idx] = day_int[pos[matched_idx]] == cur_days[matched_idx]
        if matched.any():
            overnight[pos[matched], code_to_idx[code]] = np.asarray(ovnt, dtype=np.float32)[matched]

    return overnight


def build_hfq_close_matrix(codes, daily_hfq_dir):
    sample_path = os.path.join(daily_hfq_dir, "000001.csv")
    if not os.path.exists(sample_path):
        sample_files = sorted(glob.glob(os.path.join(daily_hfq_dir, "*.csv")))
        if not sample_files:
            raise FileNotFoundError(f"No HFQ daily files found under {daily_hfq_dir}")
        sample_path = sample_files[0]

    sample = pd.read_csv(sample_path, encoding="utf-8-sig", usecols=[0])
    master_dates = pd.DatetimeIndex(
        pd.to_datetime(sample.iloc[:, 0], errors="coerce").dropna().sort_values().drop_duplicates().to_numpy()
    )
    master_days = master_dates.strftime("%Y%m%d").astype(np.int64).to_numpy()

    n_days = len(master_dates)
    n_codes = len(codes)
    close_mat = np.full((n_days, n_codes), np.nan, dtype=np.float32)
    code_to_idx = {code: idx for idx, code in enumerate(codes)}

    for path in tqdm(glob.glob(os.path.join(daily_hfq_dir, "*.csv")), desc="Loading daily HFQ closes"):
        code = Path(path).stem
        if code not in code_to_idx:
            continue

        try:
            df = pd.read_csv(path, encoding="utf-8-sig", usecols=[0, 1])
        except Exception:
            continue

        if df.shape[1] < 2:
            continue

        dt = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        close = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        mask = dt.notna() & close.notna()
        if not mask.any():
            continue

        cur_days = dt[mask].dt.strftime("%Y%m%d").astype(np.int64).to_numpy()
        cur_close = close[mask].to_numpy(dtype=np.float32)
        pos = np.searchsorted(master_days, cur_days)
        matched = pos < len(master_days)
        if matched.any():
            matched_idx = np.flatnonzero(matched)
            matched[matched_idx] = master_days[pos[matched_idx]] == cur_days[matched_idx]
        if matched.any():
            close_mat[pos[matched], code_to_idx[code]] = cur_close[matched]

    return master_dates, close_mat


def compute_forward_returns(hfq_dates, hfq_close_mat, factor_days):
    hfq_day_int = hfq_dates.strftime("%Y%m%d").astype(np.int64).to_numpy()
    factor_day_int = factor_days.strftime("%Y%m%d").astype(np.int64).to_numpy()
    factor_pos = np.searchsorted(hfq_day_int, factor_day_int)
    matched = (factor_pos < len(hfq_day_int)) & (hfq_day_int[factor_pos] == factor_day_int)
    if not matched.all():
        raise RuntimeError("Some factor dates are missing in HFQ close calendar")

    base_close = hfq_close_mat[factor_pos, :]
    forward = {}
    for horizon in HORIZONS:
        out = np.full_like(base_close, np.nan, dtype=np.float32)
        valid_pos = factor_pos + horizon < len(hfq_dates)
        if valid_pos.any():
            future_close = hfq_close_mat[factor_pos[valid_pos] + horizon, :]
            base_sub = base_close[valid_pos, :]
            good = np.isfinite(base_sub) & np.isfinite(future_close) & (base_sub > 0)
            ret = np.full_like(base_sub, np.nan, dtype=np.float32)
            ret[good] = (future_close[good] / base_sub[good] - 1.0).astype(np.float32)
            out[valid_pos, :] = ret
        forward[horizon] = out
    return forward


def compute_ic_records(factor_days, factor_mats, forward_returns):
    rows = []
    for horizon, ret_mat in forward_returns.items():
        for factor_name, factor_mat in factor_mats.items():
            for day_idx, date in enumerate(factor_days):
                ic, n_obs = safe_corr(factor_mat[day_idx], ret_mat[day_idx])
                rank_ic = safe_rank_corr(factor_mat[day_idx], ret_mat[day_idx])
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "factor": factor_name,
                        "horizon": horizon,
                        "n_obs": int(n_obs),
                        "ic": ic,
                        "rank_ic": rank_ic,
                    }
                )
    return pd.DataFrame(rows)


def summarize_ic_df(ic_df):
    rows = []
    for factor in sorted(ic_df["factor"].unique()):
        for horizon in sorted(ic_df["horizon"].unique()):
            subset = ic_df[(ic_df["factor"] == factor) & (ic_df["horizon"] == horizon)].copy()
            for metric_col, metric_name in (("ic", "Pearson IC"), ("rank_ic", "Rank IC")):
                summary = summarize_series(subset[metric_col], f"{factor} | t+{horizon} | {metric_name}")
                summary["factor"] = factor
                summary["horizon"] = int(horizon)
                summary["metric_type"] = metric_name
                summary["avg_n_obs"] = float(subset["n_obs"].mean()) if len(subset) > 0 else np.nan
                rows.append(summary)
    cols = ["factor", "horizon", "metric_type", "n_dates", "avg_n_obs", "mean", "std", "tstat", "icir", "positive_ratio", "metric"]
    return pd.DataFrame(rows)[cols]


def main():
    args = parse_args()
    t0 = time.time()

    intraday_files = sorted(
        path for path in glob.glob(os.path.join(args.intraday_dir, "*.csv")) if is_equity_code(Path(path).stem)
    )
    if not intraday_files:
        raise FileNotFoundError(f"No 15m files found under {args.intraday_dir}")
    if args.max_files > 0:
        intraday_files = intraday_files[: args.max_files]

    master_ts, master_day_int, master_days = load_master_intraday_calendar(intraday_files[0])
    factor_days = pd.to_datetime(master_days.astype(str), format="%Y%m%d")
    day_mask = (factor_days >= pd.Timestamp(args.start)) & (factor_days <= pd.Timestamp(args.end))
    selected_days = factor_days[day_mask]
    selected_day_int = master_days[day_mask.to_numpy() if hasattr(day_mask, "to_numpy") else day_mask]

    if len(selected_days) == 0:
        raise RuntimeError(f"No factor dates inside {args.start} -> {args.end}")

    print("=" * 60)
    print("  Intraday behavior factor IC validation")
    print(f"  Window: {selected_days[0].strftime('%Y-%m-%d')} ~ {selected_days[-1].strftime('%Y-%m-%d')}")
    print(f"  15m files: {len(intraday_files)}")
    print("=" * 60)

    intraday = build_intraday_behavior_matrices(intraday_files, master_ts, master_day_int, master_days)
    codes = intraday["codes"]

    overnight_mat_full = build_overnight_matrix(codes, factor_days, args.daily_raw_dir)
    hfq_dates, hfq_close_mat = build_hfq_close_matrix(codes, args.daily_hfq_dir)
    forward_returns_full = compute_forward_returns(hfq_dates, hfq_close_mat, factor_days)

    day_sel = np.searchsorted(master_days, selected_day_int)
    factor_mats = {
        "overnight_sentiment": overnight_mat_full[day_sel, :],
        "intraday_attention_up": intraday["attention_up"][day_sel, :],
        "intraday_attention_down": intraday["attention_down"][day_sel, :],
        "intraday_flow": intraday["flow"][day_sel, :],
    }
    forward_returns = {h: mat[day_sel, :] for h, mat in forward_returns_full.items()}

    ic_df = compute_ic_records(selected_days, factor_mats, forward_returns)
    summary_df = summarize_ic_df(ic_df)

    prefix = f"{args.output_prefix}_{selected_days[0].strftime('%Y%m%d')}_{selected_days[-1].strftime('%Y%m%d')}"
    daily_path = BASE_DIR / f"{prefix}_daily.csv"
    summary_path = BASE_DIR / f"{prefix}_summary.csv"
    meta_path = BASE_DIR / f"{prefix}_meta.json"

    ic_df.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    meta = {
        "start": selected_days[0].strftime("%Y-%m-%d"),
        "end": selected_days[-1].strftime("%Y-%m-%d"),
        "n_dates": int(len(selected_days)),
        "n_codes": int(len(codes)),
        "intraday_dir": os.path.abspath(args.intraday_dir),
        "daily_raw_dir": os.path.abspath(args.daily_raw_dir),
        "daily_hfq_dir": os.path.abspath(args.daily_hfq_dir),
        "max_files": int(args.max_files),
        "daily_output": str(daily_path),
        "summary_output": str(summary_path),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print("\nTop-line summary:")
    print(summary_df[["factor", "horizon", "metric_type", "n_dates", "avg_n_obs", "mean", "icir"]].to_string(index=False))
    print(f"\nDaily IC exported: {daily_path}")
    print(f"Summary exported:  {summary_path}")
    print(f"Meta exported:     {meta_path}")
    print(f"Elapsed: {meta['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
