import os
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


INDEX_SPECS = {
    "csi300": {
        "name": "CSI 300",
        "index_code": "000300",
        "launch_date": None,
        "member_key": "hs300",
    },
    "csi1000": {
        "name": "CSI 1000",
        "index_code": "000852",
        "launch_date": None,
        "member_key": "csi1000",
    },
    "csi2000": {
        "name": "CSI 2000",
        "index_code": "932000",
        "launch_date": pd.Timestamp("2023-08-11"),
        "member_key": "csi2000",
    },
}


def normalize_code(code):
    return str(code).zfill(6)


def previous_trading_date(calendar_dates, date):
    target = pd.Timestamp(date).normalize()
    dates = pd.DatetimeIndex(calendar_dates)
    pos = dates.searchsorted(target, side="right") - 1
    if pos < 0:
        return None
    return pd.Timestamp(dates[pos]).normalize()


def next_trading_date(calendar_dates, date):
    target = pd.Timestamp(date).normalize()
    dates = pd.DatetimeIndex(calendar_dates)
    pos = dates.searchsorted(target, side="left")
    if pos >= len(dates):
        return None
    return pd.Timestamp(dates[pos]).normalize()


def trading_month_ends(calendar_dates, start_date, end_date):
    dates = pd.DatetimeIndex(calendar_dates)
    mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    if not mask.any():
        return []
    dates = dates[mask]
    series = pd.Series(dates)
    return [pd.Timestamp(x).normalize() for x in series.groupby(series.dt.to_period("M")).max().tolist()]


def second_friday(year, month):
    first = pd.Timestamp(year=year, month=month, day=1)
    weekday_offset = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=weekday_offset + 7)


def build_review_events(calendar_dates, start_date, end_date, launch_date=None):
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    event_candidates = []

    if launch_date is not None:
        launch_eff = next_trading_date(calendar_dates, launch_date)
        if launch_eff is not None:
            event_candidates.append(
                {
                    "effective_date": launch_eff,
                    "ref_date": previous_trading_date(calendar_dates, launch_eff),
                    "label": "launch",
                }
            )

    for year in range(start_ts.year - 1, end_ts.year + 1):
        for month, ref_month, ref_day in ((6, 4, 30), (12, 10, 31)):
            review_friday = second_friday(year, month)
            eff_date = next_trading_date(calendar_dates, review_friday + pd.Timedelta(days=1))
            ref_date = previous_trading_date(calendar_dates, pd.Timestamp(year=year, month=ref_month, day=ref_day))
            if eff_date is None or ref_date is None:
                continue
            event_candidates.append(
                {
                    "effective_date": eff_date,
                    "ref_date": ref_date,
                    "label": f"{year}{month:02d}",
                }
            )

    dedup = {}
    for event in event_candidates:
        eff_date = pd.Timestamp(event["effective_date"]).normalize()
        current = dedup.get(eff_date)
        if current is None or str(event["label"]) == "launch":
            dedup[eff_date] = {
                "effective_date": eff_date,
                "ref_date": pd.Timestamp(event["ref_date"]).normalize(),
                "label": str(event["label"]),
            }

    events_all = [dedup[key] for key in sorted(dedup)]
    if not events_all:
        return []

    seed_event = None
    future_events = []
    for event in events_all:
        eff_date = pd.Timestamp(event["effective_date"]).normalize()
        if eff_date <= start_ts:
            seed_event = event
        elif eff_date <= end_ts:
            future_events.append(event)

    if seed_event is not None:
        events = [seed_event]
        events.extend(future_events)
        return events
    return future_events


def _top_n_codes(series, n):
    if n <= 0 or series.empty:
        return []
    return list(series.sort_values(ascending=False).head(int(n)).index)


def _eligible_by_amount(df_metrics, percentile):
    if df_metrics.empty:
        return set()
    keep_n = max(int(np.floor(len(df_metrics) * float(percentile))), 1)
    return set(df_metrics.sort_values("avg_amount", ascending=False).head(keep_n).index)


def _select_with_buffer(
    df_metrics,
    prev_members,
    target_count,
    liquidity_pct,
    new_rank_cutoff,
    old_rank_cutoff,
    exclude_codes=None,
    exclude_top_n_by_size=0,
    old_liquidity_pct=None,
):
    exclude_codes = set(exclude_codes or set())
    target_count = int(target_count)
    if df_metrics.empty or target_count <= 0:
        return []

    size_sorted_all = df_metrics.sort_values("avg_size_proxy", ascending=False)
    if exclude_top_n_by_size > 0:
        exclude_codes |= set(size_sorted_all.head(int(exclude_top_n_by_size)).index)

    candidate = df_metrics.loc[~df_metrics.index.isin(exclude_codes)].copy()
    if candidate.empty:
        return []

    liquidity_set = _eligible_by_amount(candidate, liquidity_pct)
    investable = candidate.loc[candidate.index.isin(liquidity_set)].copy()
    if investable.empty:
        return []

    investable = investable.sort_values("avg_size_proxy", ascending=False)
    size_rank = {code: rank for rank, code in enumerate(investable.index, start=1)}
    prev_members = set(prev_members or set()) & set(investable.index)

    if not prev_members:
        return list(investable.head(target_count).index)

    old_liquidity_source = candidate
    if old_liquidity_pct is not None:
        old_eligible_liq = _eligible_by_amount(old_liquidity_source, old_liquidity_pct)
        old_candidate = set(prev_members) & old_eligible_liq
    else:
        old_candidate = set(prev_members)

    guaranteed_new = {
        code for code in investable.index if code not in prev_members and size_rank.get(code, 10**9) <= int(new_rank_cutoff)
    }
    guaranteed_old = {
        code for code in old_candidate if size_rank.get(code, 10**9) <= int(old_rank_cutoff)
    }

    priority = []
    priority.extend(sorted(guaranteed_new, key=lambda code: size_rank[code]))
    priority.extend(sorted(guaranteed_old, key=lambda code: size_rank[code]))
    priority.extend(
        sorted((set(prev_members) - guaranteed_old) & set(investable.index), key=lambda code: size_rank[code])
    )
    priority.extend(
        sorted((set(investable.index) - set(prev_members) - guaranteed_new), key=lambda code: size_rank[code])
    )

    selected = []
    seen = set()
    for code in priority:
        if code in seen:
            continue
        selected.append(code)
        seen.add(code)
        if len(selected) >= target_count:
            break
    return selected


def reconstruct_csi_size_memberships(metrics_df, sample_space_map, review_events):
    metrics_by_ref = {
        pd.Timestamp(ref_date).normalize(): group.set_index("code")[["avg_amount", "avg_size_proxy"]].copy()
        for ref_date, group in metrics_df.groupby("ref_date", sort=True)
    }

    events = []
    hs300_prev = set()
    csi500_prev = set()
    csi1000_prev = set()
    csi2000_prev = set()

    for event in review_events:
        ref_date = pd.Timestamp(event["ref_date"]).normalize()
        sample_space = set(sample_space_map.get(ref_date, []))
        metrics = metrics_by_ref.get(ref_date)
        if metrics is None:
            raise RuntimeError(f"Missing metric panel for ref_date={ref_date.date()}")

        metrics = metrics.loc[metrics.index.isin(sample_space)].copy()
        if metrics.empty:
            raise RuntimeError(f"No investable metrics available for ref_date={ref_date.date()}")

        size_top300_all = set(_top_n_codes(metrics["avg_size_proxy"], 300))
        size_top1500_all = set(_top_n_codes(metrics["avg_size_proxy"], 1500))

        hs300 = _select_with_buffer(
            df_metrics=metrics,
            prev_members=hs300_prev,
            target_count=300,
            liquidity_pct=0.50,
            new_rank_cutoff=240,
            old_rank_cutoff=360,
            old_liquidity_pct=0.60,
        )
        hs300_set = set(hs300)

        csi500 = _select_with_buffer(
            df_metrics=metrics,
            prev_members=csi500_prev,
            target_count=500,
            liquidity_pct=0.80,
            new_rank_cutoff=400,
            old_rank_cutoff=600,
            exclude_codes=hs300_set,
            exclude_top_n_by_size=300,
            old_liquidity_pct=0.90,
        )
        csi500_set = set(csi500)

        csi800_set = hs300_set | csi500_set

        csi1000 = _select_with_buffer(
            df_metrics=metrics,
            prev_members=csi1000_prev,
            target_count=1000,
            liquidity_pct=0.80,
            new_rank_cutoff=800,
            old_rank_cutoff=1200,
            exclude_codes=csi800_set | size_top300_all,
            exclude_top_n_by_size=0,
            old_liquidity_pct=0.90,
        )
        csi1000_set = set(csi1000)

        liquidity_eligible = metrics.loc[metrics.index.isin(_eligible_by_amount(metrics, 0.90))]
        csi2000 = _select_with_buffer(
            df_metrics=liquidity_eligible,
            prev_members=csi2000_prev,
            target_count=2000,
            liquidity_pct=1.00,
            new_rank_cutoff=1600,
            old_rank_cutoff=2400,
            exclude_codes=csi800_set | csi1000_set | size_top1500_all,
            exclude_top_n_by_size=0,
            old_liquidity_pct=None,
        )
        csi2000_set = set(csi2000)

        events.append(
            {
                "effective_date": pd.Timestamp(event["effective_date"]).normalize(),
                "ref_date": ref_date,
                "label": event["label"],
                "hs300": hs300,
                "csi500": csi500,
                "csi1000": csi1000,
                "csi2000": csi2000,
            }
        )

        hs300_prev = hs300_set
        csi500_prev = csi500_set
        csi1000_prev = csi1000_set
        csi2000_prev = csi2000_set

    return events


def build_effective_and_monthly_panels(event_memberships, calendar_dates, start_date, end_date, member_key):
    effective_rows = []
    for event in event_memberships:
        eff_date = pd.Timestamp(event["effective_date"]).normalize()
        for code in sorted(event[member_key]):
            effective_rows.append(
                {
                    "effective_date": eff_date.strftime("%Y-%m-%d"),
                    "ref_date": pd.Timestamp(event["ref_date"]).strftime("%Y-%m-%d"),
                    "event_label": event["label"],
                    "code": normalize_code(code),
                }
            )

    monthly_rows = []
    event_dates = [pd.Timestamp(item["effective_date"]).normalize() for item in event_memberships]
    event_code_sets = [set(item[member_key]) for item in event_memberships]
    snapshot_dates = []
    if event_dates:
        snapshot_dates.append(event_dates[0])
    snapshot_dates.extend(trading_month_ends(calendar_dates, start_date, end_date))
    snapshot_dates = sorted(set(pd.Timestamp(x).normalize() for x in snapshot_dates))

    for snap_date in snapshot_dates:
        pos = bisect_right(event_dates, snap_date) - 1
        if pos < 0:
            continue
        eff_date = event_dates[pos]
        for code in sorted(event_code_sets[pos]):
            monthly_rows.append(
                {
                    "snapshot_date": snap_date.strftime("%Y-%m-%d"),
                    "effective_date": eff_date.strftime("%Y-%m-%d"),
                    "code": normalize_code(code),
                }
            )

    return pd.DataFrame(effective_rows), pd.DataFrame(monthly_rows)


def _compute_ref_metrics_for_code(code, daily_dir, intraday_dir, ref_dates, rolling_window):
    daily_path = os.path.join(daily_dir, f"{code}.csv")
    intraday_path = os.path.join(intraday_dir, f"{code}.csv")
    if not os.path.exists(daily_path) or not os.path.exists(intraday_path):
        return []

    try:
        df_daily = pd.read_csv(daily_path, usecols=["日期", "收盘", "换手率"])
        df_daily.rename(columns={"日期": "date", "收盘": "close", "换手率": "turnover"}, inplace=True)
        df_daily["date"] = pd.to_datetime(df_daily["date"], errors="coerce")
        df_daily["turnover"] = pd.to_numeric(df_daily["turnover"], errors="coerce")
        df_daily.dropna(subset=["date", "turnover"], inplace=True)
        if df_daily.empty:
            return []

        df_intraday = pd.read_csv(intraday_path, usecols=["time", "amount"])
        df_intraday["date"] = pd.to_datetime(df_intraday["time"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
        df_intraday["amount"] = pd.to_numeric(df_intraday["amount"], errors="coerce")
        df_intraday = df_intraday.dropna(subset=["date", "amount"]).groupby("date", as_index=False)["amount"].sum()
        if df_intraday.empty:
            return []

        df = df_daily.merge(df_intraday, on="date", how="left")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        turnover_frac = df["turnover"] / 100.0
        df["size_proxy"] = np.where(
            (turnover_frac > 0) & pd.notna(df["amount"]) & (df["amount"] > 0),
            df["amount"] / turnover_frac,
            np.nan,
        )
        df = df.sort_values("date").reset_index(drop=True)
        if df.empty:
            return []

        records = []
        ref_dates = [pd.Timestamp(x).normalize() for x in ref_dates]
        for ref_date in ref_dates:
            window = df.loc[df["date"] <= ref_date, ["date", "amount", "size_proxy"]].tail(int(rolling_window))
            if len(window) < max(120, int(rolling_window * 0.5)):
                continue
            avg_amount = pd.to_numeric(window["amount"], errors="coerce").mean()
            avg_size = pd.to_numeric(window["size_proxy"], errors="coerce").mean()
            if not np.isfinite(avg_amount) or not np.isfinite(avg_size) or avg_amount <= 0 or avg_size <= 0:
                continue
            records.append(
                {
                    "ref_date": ref_date.strftime("%Y-%m-%d"),
                    "code": normalize_code(code),
                    "avg_amount": float(avg_amount),
                    "avg_size_proxy": float(avg_size),
                }
            )
        return records
    except Exception:
        return []


def build_metric_panel_for_ref_dates(
    codes,
    daily_dir,
    intraday_dir,
    ref_dates,
    rolling_window=252,
    max_workers=8,
):
    ref_dates = [pd.Timestamp(x).normalize() for x in ref_dates]
    tasks = [normalize_code(code) for code in sorted(set(codes))]
    records = []
    max_workers = max(int(max_workers), 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _compute_ref_metrics_for_code,
                code,
                daily_dir,
                intraday_dir,
                ref_dates,
                rolling_window,
            ): code
            for code in tasks
        }
        for future in as_completed(future_map):
            code_records = future.result()
            if code_records:
                records.extend(code_records)

    if not records:
        raise RuntimeError("No reconstructed size metrics were generated.")
    return pd.DataFrame(records)


def load_effective_panel_entries(panel_path):
    df = pd.read_csv(panel_path)
    if df.empty:
        return []
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["code"] = df["code"].astype(str).str.zfill(6)
    entries = []
    for eff_date, group in df.groupby("effective_date", sort=True):
        entries.append((pd.Timestamp(eff_date).strftime("%Y-%m-%d"), tuple(sorted(group["code"].unique()))))
    return entries


def _load_monthly_snapshot_dates(panel_path):
    df = pd.read_csv(panel_path, usecols=["snapshot_date"])
    if df.empty:
        return set()
    return set(pd.to_datetime(df["snapshot_date"], errors="coerce").dropna().dt.normalize().tolist())


def _normalize_metrics_df(metrics_df):
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(columns=["ref_date", "code", "avg_amount", "avg_size_proxy"])
    out = metrics_df.copy()
    out["ref_date"] = pd.to_datetime(out["ref_date"], errors="coerce")
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["avg_amount"] = pd.to_numeric(out["avg_amount"], errors="coerce")
    out["avg_size_proxy"] = pd.to_numeric(out["avg_size_proxy"], errors="coerce")
    out = out.dropna(subset=["ref_date", "code", "avg_amount", "avg_size_proxy"])
    return out


def _serialize_review_events(review_events):
    return [
        {
            "effective_date": pd.Timestamp(item["effective_date"]).strftime("%Y-%m-%d"),
            "ref_date": pd.Timestamp(item["ref_date"]).strftime("%Y-%m-%d"),
            "label": str(item["label"]),
        }
        for item in review_events
    ]


def _build_csi_size_reconstructed_panels(
    target_index,
    calendar_dates,
    base_filter,
    daily_dir,
    intraday_dir,
    start_date,
    end_date,
    effective_panel_path,
    monthly_panel_path,
    metric_cache_path=None,
    force_rebuild=False,
    max_workers=8,
):
    index_key = str(target_index).lower()
    if index_key not in INDEX_SPECS:
        raise ValueError(f"Unsupported target_index: {target_index}")
    spec = INDEX_SPECS[index_key]

    effective_panel_path = os.path.abspath(effective_panel_path)
    monthly_panel_path = os.path.abspath(monthly_panel_path)
    metric_cache_path = None if metric_cache_path is None else os.path.abspath(metric_cache_path)

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    review_events = build_review_events(
        calendar_dates=calendar_dates,
        start_date=start_ts,
        end_date=end_ts,
        launch_date=spec["launch_date"],
    )
    if not review_events:
        raise RuntimeError(f"No {spec['name']} review events could be constructed for the requested window.")

    required_effective_dates = {pd.Timestamp(event["effective_date"]).normalize() for event in review_events}
    required_monthly_dates = set(trading_month_ends(calendar_dates, start_ts, end_ts))
    required_monthly_dates |= {
        pd.Timestamp(event["effective_date"]).normalize() for event in review_events[:1]
    }

    if (
        not force_rebuild
        and os.path.exists(effective_panel_path)
        and os.path.exists(monthly_panel_path)
    ):
        effective_entries = load_effective_panel_entries(effective_panel_path)
        cached_effective_dates = {pd.Timestamp(item[0]).normalize() for item in effective_entries}
        cached_monthly_dates = _load_monthly_snapshot_dates(monthly_panel_path)
        if required_effective_dates.issubset(cached_effective_dates) and required_monthly_dates.issubset(
            cached_monthly_dates
        ):
            launch_eff = effective_entries[0][0] if effective_entries else None
            return {
                "panel_type": "reconstructed",
                "name": spec["name"],
                "index_code": spec["index_code"],
                "member_key": spec["member_key"],
                "effective_panel_path": effective_panel_path,
                "monthly_panel_path": monthly_panel_path,
                "metric_cache_path": metric_cache_path,
                "effective_entries": effective_entries,
                "launch_effective_date": launch_eff,
                "review_events": _serialize_review_events(review_events),
            }

    os.makedirs(os.path.dirname(effective_panel_path), exist_ok=True)
    os.makedirs(os.path.dirname(monthly_panel_path), exist_ok=True)
    if metric_cache_path:
        os.makedirs(os.path.dirname(metric_cache_path), exist_ok=True)

    ref_dates = sorted(set(pd.Timestamp(event["ref_date"]).normalize() for event in review_events))
    sample_space_map = {ref_date: set(base_filter.filter_universe(ref_date)) for ref_date in ref_dates}
    all_codes = sorted(set().union(*sample_space_map.values()))

    if metric_cache_path and os.path.exists(metric_cache_path) and not force_rebuild:
        metrics_df = _normalize_metrics_df(pd.read_csv(metric_cache_path))
        cached_ref_dates = set(metrics_df["ref_date"].dt.normalize().tolist())
        missing_ref_dates = [ref_date for ref_date in ref_dates if ref_date not in cached_ref_dates]
        if missing_ref_dates:
            extra_metrics = build_metric_panel_for_ref_dates(
                codes=all_codes,
                daily_dir=daily_dir,
                intraday_dir=intraday_dir,
                ref_dates=missing_ref_dates,
                max_workers=max_workers,
            )
            metrics_df = pd.concat([metrics_df, _normalize_metrics_df(extra_metrics)], ignore_index=True)
            metrics_df = metrics_df.drop_duplicates(subset=["ref_date", "code"], keep="last")
            metrics_df.to_csv(metric_cache_path, index=False, encoding="utf-8-sig")
    else:
        metrics_df = build_metric_panel_for_ref_dates(
            codes=all_codes,
            daily_dir=daily_dir,
            intraday_dir=intraday_dir,
            ref_dates=ref_dates,
            max_workers=max_workers,
        )
        if metric_cache_path:
            metrics_df.to_csv(metric_cache_path, index=False, encoding="utf-8-sig")

    metrics_df = _normalize_metrics_df(metrics_df)

    event_memberships = reconstruct_csi_size_memberships(metrics_df, sample_space_map, review_events)
    effective_df, monthly_df = build_effective_and_monthly_panels(
        event_memberships=event_memberships,
        calendar_dates=calendar_dates,
        start_date=start_ts,
        end_date=end_ts,
        member_key=spec["member_key"],
    )
    effective_df.to_csv(effective_panel_path, index=False, encoding="utf-8-sig")
    monthly_df.to_csv(monthly_panel_path, index=False, encoding="utf-8-sig")

    effective_entries = load_effective_panel_entries(effective_panel_path)
    launch_eff = effective_entries[0][0] if effective_entries else None
    return {
        "panel_type": "reconstructed",
        "name": spec["name"],
        "index_code": spec["index_code"],
        "member_key": spec["member_key"],
        "effective_panel_path": effective_panel_path,
        "monthly_panel_path": monthly_panel_path,
        "metric_cache_path": metric_cache_path,
        "effective_entries": effective_entries,
        "launch_effective_date": launch_eff,
        "review_events": _serialize_review_events(event_memberships),
    }


def build_csi1000_reconstructed_panels(**kwargs):
    return _build_csi_size_reconstructed_panels(target_index="csi1000", **kwargs)


def build_csi2000_reconstructed_panels(**kwargs):
    return _build_csi_size_reconstructed_panels(target_index="csi2000", **kwargs)


def build_csi300_reconstructed_panels(**kwargs):
    return _build_csi_size_reconstructed_panels(target_index="csi300", **kwargs)
