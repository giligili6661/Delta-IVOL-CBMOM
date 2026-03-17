from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None


ROOT = Path(__file__).resolve().parent
SRC_15M_DIR = ROOT / "data_stock_15m"
QFQ_DAILY_DIR = ROOT / "data_stock_daily"
UNADJ_DAILY_DIR = ROOT / "data_stock_daily_unadj"
OUT_15M_DIR = ROOT / "data_stock_15m_unadj"

VALID_PREFIXES = (
    "000",
    "001",
    "002",
    "003",
    "300",
    "301",
    "600",
    "601",
    "603",
    "605",
    "688",
    "689",
)


def is_valid_stock(code: str) -> bool:
    return len(code) == 6 and code.isdigit() and code.startswith(VALID_PREFIXES)


def code_to_symbol(code: str) -> str:
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def list_valid_codes() -> list[str]:
    return sorted(
        path.stem
        for path in SRC_15M_DIR.glob("*.csv")
        if is_valid_stock(path.stem)
    )


def read_daily_close(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=[0, 1])
    except Exception:
        return None
    if df.empty or df.shape[1] < 2:
        return None
    df = df.iloc[:, :2].copy()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    if df.empty:
        return None
    return df.reset_index(drop=True)


def save_unadj_daily(df: pd.DataFrame, path: Path) -> None:
    out = df[["date", "close", "open"]].copy()
    out["turnover"] = 0.0
    out.columns = ["日期", "收盘", "开盘", "换手率"]
    out["日期"] = pd.to_datetime(out["日期"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, encoding="utf-8")


def fetch_unadj_daily_tx(code: str) -> pd.DataFrame | None:
    if ak is None:
        return None
    try:
        df = ak.stock_zh_a_hist_tx(
            symbol=code_to_symbol(code),
            start_date="1990-01-01",
            end_date=dt.date.today().strftime("%Y-%m-%d"),
            adjust="",
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    expected = {"date", "open", "close"}
    if not expected.issubset(df.columns):
        return None
    keep = df[["date", "close", "open"]].copy()
    keep["date"] = pd.to_datetime(keep["date"], errors="coerce")
    keep["close"] = pd.to_numeric(keep["close"], errors="coerce")
    keep["open"] = pd.to_numeric(keep["open"], errors="coerce")
    keep = keep.dropna(subset=["date", "close", "open"])
    if keep.empty:
        return None
    return keep.sort_values("date").reset_index(drop=True)


def ensure_unadj_daily(code: str) -> str:
    path = UNADJ_DAILY_DIR / f"{code}.csv"
    if path.exists():
        return "local"
    fetched = fetch_unadj_daily_tx(code)
    if fetched is None:
        return "fetch_failed"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_unadj_daily(fetched, path)
    return "fetched"


def build_unadj_factor(code: str) -> pd.Series | None:
    qfq = read_daily_close(QFQ_DAILY_DIR / f"{code}.csv")
    unadj = read_daily_close(UNADJ_DAILY_DIR / f"{code}.csv")
    if qfq is None or unadj is None:
        return None
    merged = qfq.merge(unadj, on="date", how="inner", suffixes=("_qfq", "_unadj"))
    merged = merged[merged["close_qfq"] > 0]
    if merged.empty:
        return None
    merged["factor"] = merged["close_unadj"] / merged["close_qfq"]
    merged["date_key"] = merged["date"].dt.strftime("%Y%m%d")
    return merged.set_index("date_key")["factor"]


def convert_one(code: str) -> dict[str, str]:
    daily_status = ensure_unadj_daily(code)
    factor = build_unadj_factor(code)
    if factor is None:
        return {
            "code": code,
            "unadj_daily": daily_status,
            "status": "skip",
            "note": "no_factor",
        }

    src_path = SRC_15M_DIR / f"{code}.csv"
    out_path = OUT_15M_DIR / f"{code}.csv"
    try:
        df = pd.read_csv(src_path)
    except Exception as exc:
        return {
            "code": code,
            "unadj_daily": daily_status,
            "status": "failed",
            "note": f"read_15m:{exc}",
        }

    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return {
            "code": code,
            "unadj_daily": daily_status,
            "status": "failed",
            "note": "bad_15m_schema",
        }

    df["_date"] = df["time"].astype(str).str[:8]
    df["_factor"] = df["_date"].map(factor)
    df["_factor"] = df["_factor"].ffill().bfill()
    if df["_factor"].isna().all():
        return {
            "code": code,
            "unadj_daily": daily_status,
            "status": "skip",
            "note": "factor_all_nan",
        }

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") * df["_factor"]
        df[col] = df[col].round(4)

    OUT_15M_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = [col for col in df.columns if not col.startswith("_")]
    df[out_cols].to_csv(out_path, index=False, encoding="utf-8")
    return {
        "code": code,
        "unadj_daily": daily_status,
        "status": "ok",
        "note": "rebuilt",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 15m unadjusted prices from qfq-like 15m data.")
    parser.add_argument("--codes", nargs="*", help="Optional stock codes to process.")
    parser.add_argument("--workers", type=int, default=8, help="Thread workers.")
    args = parser.parse_args()

    all_codes = list_valid_codes()
    target_codes = [code for code in (args.codes or all_codes) if is_valid_stock(code)]

    print(f"valid 15m stock files: {len(all_codes)}")
    print(f"target codes: {len(target_codes)}")
    OUT_15M_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_one, code): code for code in target_codes}
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{idx}/{len(target_codes)}] {row['code']} "
                f"unadj_daily={row['unadj_daily']} -> {row['status']}:{row['note']}"
            )

    summary_path = ROOT / f"build_15m_unadj_summary_{dt.date.today():%Y%m%d}.csv"
    pd.DataFrame(rows).sort_values("code").to_csv(summary_path, index=False, encoding="utf-8")
    print(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
