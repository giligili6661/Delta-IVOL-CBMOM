"""
中国A股 Fama-French 三因子 (FF3) 数据更新脚本
=============================================================
通过 akshare 下载指数日线数据, 构建 FF3 因子:
  - MKT: 市场风险溢价 = 中证全指日收益率 - 无风险利率
  - SMB: 市值因子 = 中证1000(小盘) - 沪深300(大盘)
  - HML: 账面市值比因子 = 中证红利(价值) - 创业板指(成长)

输出: ff3_factors_cn.csv  [date, MKT, SMB, HML]

用法: python update_ff3.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "ff3_factors_cn.csv")

START_DATE = "20150101"
END_DATE = "20260305"
RISK_FREE_RATE = 0.02  # 年化无风险利率


def _clear_proxy():
    """临时清除代理环境变量, 返回原始值以便恢复"""
    proxy_keys = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
                  'ALL_PROXY', 'all_proxy']
    saved = {}
    for k in proxy_keys:
        if k in os.environ:
            saved[k] = os.environ.pop(k)
    return saved


def _restore_proxy(saved):
    """恢复代理环境变量"""
    for k, v in saved.items():
        os.environ[k] = v


def _parse_index_df(df):
    """统一处理指数DataFrame的列名和收益率计算"""
    if df is None or len(df) == 0:
        return None

    # 中文列名映射
    col_map = {
        '日期': 'date', '收盘': 'close', '涨跌幅': 'pct_change',
        'date': 'date', 'close': 'close',
    }

    if '日期' in df.columns:
        df = df.rename(columns=col_map)
    elif 'date' not in df.columns:
        # 按位置猜测
        if len(df.columns) >= 7:
            df.columns = ['date', 'open', 'close', 'high', 'low',
                          'volume', 'amount'] + list(df.columns[7:])

    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    # 计算日收益率
    if 'pct_change' in df.columns:
        df['ret'] = pd.to_numeric(df['pct_change'], errors='coerce') / 100.0
    else:
        df['ret'] = df['close'].pct_change()

    df = df[['date', 'close', 'ret']].dropna(subset=['ret'])
    return df


def download_index(symbol, name, start_date=START_DATE, end_date=END_DATE):
    """
    下载单个指数的日线数据, 返回 DataFrame [date, close, ret]
    优先东方财富源, 失败则回退新浪源, 自动绕过代理
    """
    import akshare as ak

    print(f"  下载 {name} ({symbol})...", end=" ", flush=True)

    # 临时清除代理
    saved_proxy = _clear_proxy()

    df = None
    # 新浪源需要交易所前缀: sh/sz
    sina_symbol = f"sh{symbol}" if symbol.startswith(('0', '5', '9')) else f"sz{symbol}"

    # ---- 方法1 (优先): 新浪源 (stock_zh_index_daily) ----
    try:
        raw = ak.stock_zh_index_daily(symbol=sina_symbol)
        if raw is not None and len(raw) > 0:
            raw = raw.reset_index() if 'date' not in raw.columns else raw
            if 'date' in raw.columns:
                raw['date'] = pd.to_datetime(raw['date'])
                sd = pd.to_datetime(start_date)
                ed = pd.to_datetime(end_date)
                raw = raw[(raw['date'] >= sd) & (raw['date'] <= ed)]
            df = _parse_index_df(raw)
    except Exception as e1:
        print(f"\n    新浪源失败: {type(e1).__name__}", end=" ", flush=True)

    # ---- 方法2: 东方财富源 (index_zh_a_hist) ----
    if df is None or len(df) == 0:
        try:
            raw = ak.index_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start_date, end_date=end_date
            )
            df = _parse_index_df(raw)
        except Exception as e2:
            print(f"\n    东方财富源也失败: {type(e2).__name__}", end=" ", flush=True)

    # 恢复代理
    _restore_proxy(saved_proxy)

    if df is not None and len(df) > 0:
        print(f"{len(df)} 个交易日")
        return df
    else:
        print("所有源均失败")
        return None


def build_ff3():
    """
    构建 FF3 三因子 (使用新浪源覆盖最全的指数):
      MKT = 沪深300日收益率 - rf
      SMB = 中证1000(小盘) - 沪深300(大盘)
      HML = 上证180(大盘价值) - 创业板指(成长)
    """
    print("\n" + "="*60)
    print("  下载指数数据 (新浪源优先)")
    print("="*60)

    # ---- 1. 市场基准: 沪深300 (000300) ----
    # 新浪对沪深300覆盖最全, 优先使用
    mkt_df = download_index("000300", "沪深300 (MKT基准)")

    if mkt_df is None:
        print("错误: 无法获取市场基准指数!")
        sys.exit(1)

    # ---- 2. 小盘指数: 中证1000 (000852) ----
    small_df = download_index("000852", "中证1000 (小盘)")

    if small_df is None:
        print("  -> 回退: 使用中证500")
        small_df = download_index("000905", "中证500 (小盘)")

    # ---- 3. 大盘指数: 沪深300 (000300) ----
    # 复用 mkt_df
    big_df = mkt_df
    print(f"  沪深300 (大盘): 复用MKT基准数据")

    # ---- 4. 价值指数: 上证180 (000010) ----
    # 上证180在新浪覆盖完整, 以大盘蓝筹代表价值风格
    value_df = download_index("000010", "上证180 (价值)")

    if value_df is None:
        print("  -> 回退: 使用上证50")
        value_df = download_index("000016", "上证50 (价值)")

    # ---- 5. 成长指数: 创业板指 (399006) ----
    growth_df = download_index("399006", "创业板指 (成长)")

    # ========== 合并构建因子 ==========
    print("\n" + "="*60)
    print("  构建FF3因子")
    print("="*60)

    # 以市场基准的日期为准
    result = mkt_df[['date']].copy()

    # MKT: 市场超额收益 = 市场日收益率 - 日无风险
    daily_rf = RISK_FREE_RATE / 252
    result['MKT'] = mkt_df['ret'].values - daily_rf

    # SMB: 小盘 - 大盘
    if small_df is not None and big_df is not None:
        small_aligned = pd.merge(result[['date']], small_df[['date', 'ret']],
                                  on='date', how='left')
        big_aligned = pd.merge(result[['date']], big_df[['date', 'ret']],
                                on='date', how='left')
        result['SMB'] = small_aligned['ret'].values - big_aligned['ret'].values
    else:
        print("  警告: SMB因子缺少数据, 使用0填充")
        result['SMB'] = 0.0

    # HML: 价值 - 成长
    if value_df is not None and growth_df is not None:
        value_aligned = pd.merge(result[['date']], value_df[['date', 'ret']],
                                  on='date', how='left')
        growth_aligned = pd.merge(result[['date']], growth_df[['date', 'ret']],
                                   on='date', how='left')
        result['HML'] = value_aligned['ret'].values - growth_aligned['ret'].values
    else:
        print("  警告: HML因子缺少数据, 使用0填充")
        result['HML'] = 0.0

    # 清理 NaN
    result = result.dropna(subset=['MKT']).reset_index(drop=True)
    result['SMB'] = result['SMB'].fillna(0)
    result['HML'] = result['HML'].fillna(0)

    return result


def print_summary(df):
    """打印因子摘要统计"""
    print(f"\n{'─'*60}")
    print(f"  FF3 因子摘要")
    print(f"{'─'*60}")
    print(f"  日期范围: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ "
          f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  交易日数: {len(df)}")
    print()

    for col in ['MKT', 'SMB', 'HML']:
        vals = df[col].values
        ann_mean = np.mean(vals) * 252
        ann_std = np.std(vals) * np.sqrt(252)
        print(f"  {col}:")
        print(f"    日均值:   {np.mean(vals):>10.6f}  ({ann_mean:>+7.2%} 年化)")
        print(f"    日标准差: {np.std(vals):>10.6f}  ({ann_std:>7.2%} 年化)")
        print(f"    最小值:   {np.min(vals):>10.6f}")
        print(f"    最大值:   {np.max(vals):>10.6f}")
        print()

    # 相关性矩阵
    corr = df[['MKT', 'SMB', 'HML']].corr()
    print(f"  因子相关性矩阵:")
    print(f"         MKT      SMB      HML")
    for idx in ['MKT', 'SMB', 'HML']:
        print(f"  {idx}  {corr.loc[idx, 'MKT']:>7.3f}  "
              f"{corr.loc[idx, 'SMB']:>7.3f}  "
              f"{corr.loc[idx, 'HML']:>7.3f}")


def main():
    t0 = datetime.now()
    print("="*60)
    print("  中国A股 Fama-French 三因子 (FF3) 更新")
    print(f"  区间: {START_DATE[:4]}-{START_DATE[4:6]}-{START_DATE[6:]} ~ "
          f"{END_DATE[:4]}-{END_DATE[4:6]}-{END_DATE[6:]}")
    print(f"  运行时间: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 构建因子
    ff3 = build_ff3()

    # 打印摘要
    print_summary(ff3)

    # 保存
    ff3.to_csv(OUTPUT_PATH, index=False)
    print(f"\n{'='*60}")
    print(f"  FF3因子已保存: {OUTPUT_PATH}")
    print(f"  共 {len(ff3)} 个交易日")
    print(f"  文件大小: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

    t1 = datetime.now()
    print(f"  总运行时间: {(t1-t0).total_seconds():.1f} 秒")
    print("="*60)


if __name__ == "__main__":
    main()
