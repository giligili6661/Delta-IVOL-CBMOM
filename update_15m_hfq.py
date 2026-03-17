import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import akshare as ak
import pandas as pd
import numpy as np
import datetime
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
DATA_DIR       = "data_stock_15m_hfq"       # 15分钟后复权数据输出目录
DAILY_HFQ_DIR  = "data_stock_daily_hfq"     # 日线后复权数据目录 (已下好)
DAILY_RAW_DIR  = "data_stock_daily"         # 日线不复权数据目录 (已下好)
MAX_RETRY      = 3
REQUEST_INTERVAL = 0.3
WORKERS        = 4
# ===========================================

VALID_PREFIXES = ('000', '001', '002', '003', '300', '301', '600', '601', '603', '605', '688', '689')

def is_valid_stock(code):
    if len(code) != 6 or not code.isdigit():
        return False
    return code.startswith(VALID_PREFIXES)

def code_to_sina_symbol(code):
    if code.startswith('6'):
        return f"sh{code}"
    else:
        return f"sz{code}"

def get_all_stock_codes():
    """从日线数据目录获取股票列表"""
    codes = []
    skipped = 0
    if not os.path.exists(DAILY_RAW_DIR):
        print(f"❌ 日线数据目录 {DAILY_RAW_DIR} 不存在")
        return codes
    for f in os.listdir(DAILY_RAW_DIR):
        if not f.endswith('.csv'):
            continue
        code = f.replace('.csv', '')
        if is_valid_stock(code):
            codes.append(code)
        else:
            skipped += 1
    print(f"   📂 有效个股: {len(codes)} 只, 跳过非个股: {skipped} 只")
    return sorted(codes)

def get_local_last_datetime(file_path):
    """读取本地15m CSV最后一条记录的时间"""
    try:
        df = pd.read_csv(file_path, usecols=['day'])
        if len(df) == 0:
            return None
        return pd.to_datetime(df['day']).max()
    except:
        return None

def load_hfq_factor(code):
    """
    从本地日线数据计算每日后复权因子.
    因子 = 日线后复权收盘价 / 日线不复权收盘价
    返回 Series(index=date, values=factor)
    """
    hfq_path = os.path.join(DAILY_HFQ_DIR, f"{code}.csv")
    raw_path = os.path.join(DAILY_RAW_DIR, f"{code}.csv")

    if not os.path.exists(hfq_path) or not os.path.exists(raw_path):
        return None

    try:
        df_hfq = pd.read_csv(hfq_path, usecols=['日期', '收盘'])
        df_raw = pd.read_csv(raw_path, usecols=['日期', '收盘'])

        df_hfq['日期'] = pd.to_datetime(df_hfq['日期'])
        df_raw['日期'] = pd.to_datetime(df_raw['日期'])

        df_hfq.set_index('日期', inplace=True)
        df_raw.set_index('日期', inplace=True)

        df_hfq.columns = ['close_hfq']
        df_raw.columns = ['close_raw']

        merged = pd.merge(df_hfq, df_raw, left_index=True, right_index=True, how='inner')
        merged = merged[merged['close_raw'] > 0]
        merged['factor'] = merged['close_hfq'] / merged['close_raw']

        return merged['factor']
    except:
        return None

def download_raw_15m(code):
    """下载不复权的15分钟数据 (绕过 akshare 的 adjust bug, 直接请求新浪 API)"""
    sina_symbol = code_to_sina_symbol(code)
    for attempt in range(MAX_RETRY):
        try:
            url = "https://quotes.sina.cn/cn/api/jsonp_v2.php/=/CN_MarketDataService.getKLineData"
            params = {
                "symbol": sina_symbol,
                "scale": "15",
                "ma": "no",
                "datalen": "1970",
            }
            r = requests.get(url, params=params, timeout=15)
            data_text = r.text

            # 解析 JSON (新浪返回格式: =([ ... ]);)
            try:
                data_json = json.loads(data_text.split("=(")[1].split(");")[0])
            except (IndexError, json.JSONDecodeError):
                # 备用 URL
                url2 = f"https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_{sina_symbol}_15_1658852984203=/CN_MarketDataService.getKLineData"
                r = requests.get(url2, params=params, timeout=15)
                data_text = r.text
                try:
                    data_json = json.loads(data_text.split("=(")[1].split(");")[0])
                except (IndexError, json.JSONDecodeError):
                    return code, None, "parse_error"

            if not data_json:
                return code, None, "empty"

            df = pd.DataFrame(data_json).iloc[:, :6]
            df.columns = ['day', 'open', 'high', 'low', 'close', 'volume']
            # 转为数值
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['close'])
            if df.empty:
                return code, None, "empty"
            return code, df, "ok"

        except (ConnectionError, ConnectionResetError, TimeoutError, OSError) as e:
            if attempt < MAX_RETRY - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return code, None, f"network: {str(e)[:80]}"
        except Exception as e:
            err_str = str(e)
            if '不存在' in err_str or '404' in err_str:
                return code, None, "empty"
            if attempt < MAX_RETRY - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return code, None, err_str
    return code, None, "max_retry"

def apply_hfq(df_15m, factor_series):
    """
    将后复权因子应用到15分钟不复权数据上.
    每根15分钟K线对应其所在交易日的后复权因子.
    """
    df = df_15m.copy()
    df['_dt'] = pd.to_datetime(df['day'])
    df['_date'] = df['_dt'].dt.normalize()  # 只取日期部分

    # 合并因子
    factor_df = factor_series.reset_index()
    factor_df.columns = ['_date', '_factor']
    factor_df['_date'] = pd.to_datetime(factor_df['_date']).dt.normalize()

    df = pd.merge(df, factor_df, on='_date', how='left')

    # 对没有因子的日期 (可能因为日线数据不全), 用前值填充
    df['_factor'] = df['_factor'].ffill()
    df['_factor'] = df['_factor'].bfill()

    if df['_factor'].isna().all():
        return None

    # 应用后复权因子
    for col in ['open', 'high', 'low', 'close']:
        df[col] = (df[col] * df['_factor']).round(2)

    # 清理辅助列
    df = df[['day', 'open', 'high', 'low', 'close', 'volume']]
    return df

def process_stock(code):
    """处理单只股票: 下载不复权15m数据 → 应用后复权因子 → 保存"""
    file_path = os.path.join(DATA_DIR, f"{code}.csv")

    # 1. 加载后复权因子
    factor_series = load_hfq_factor(code)
    if factor_series is None or factor_series.empty:
        return code, "skipped"  # 没有日线数据, 跳过

    # 2. 下载不复权15m数据
    code_ret, df_raw, status = download_raw_15m(code)
    if status != "ok" or df_raw is None or df_raw.empty:
        if status == "empty":
            return code, "skipped"
        return code, f"failed: {status}"

    # 3. 应用后复权因子
    df_hfq = apply_hfq(df_raw, factor_series)
    if df_hfq is None or df_hfq.empty:
        return code, "skipped"

    # 4. 增量写入
    if os.path.exists(file_path):
        last_dt = get_local_last_datetime(file_path)
        if last_dt is not None:
            df_new_dt = pd.to_datetime(df_hfq['day'])
            df_hfq = df_hfq[df_new_dt > last_dt]
            if df_hfq.empty:
                return code, "skipped"
        try:
            df_existing = pd.read_csv(file_path)
            df_combined = pd.concat([df_existing, df_hfq], ignore_index=True)
            df_combined.drop_duplicates(subset=['day'], keep='last', inplace=True)
            df_combined.to_csv(file_path, index=False, encoding='utf-8')
            return code, "updated"
        except Exception as e:
            return code, f"failed: {e}"
    else:
        df_hfq.to_csv(file_path, index=False, encoding='utf-8')
        return code, "downloaded"

def main():
    print(f"🚀 15分钟后复权数据更新引擎 | 不复权15m × 日线复权因子")
    print(f"   📅 当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   📂 日线后复权: {DAILY_HFQ_DIR}/ | 日线不复权: {DAILY_RAW_DIR}/")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 检查日线数据是否存在
    if not os.path.exists(DAILY_HFQ_DIR):
        print(f"❌ 后复权日线目录 {DAILY_HFQ_DIR} 不存在, 请先运行 update_data_hfq.py")
        return

    all_codes = get_all_stock_codes()
    print(f"📊 目标股票数: {len(all_codes)} 只")

    if not all_codes:
        print("❌ 股票列表为空")
        return

    stats = {"downloaded": 0, "updated": 0, "skipped": 0, "failed": 0}
    failed_codes = []
    lock = threading.Lock()

    def worker(code):
        try:
            ret_code, result = process_stock(code)
            if result not in ("skipped",):
                time.sleep(REQUEST_INTERVAL)
            return ret_code, result
        except Exception as e:
            return code, f"failed: {e}"

    pbar = tqdm(total=len(all_codes), desc="下载进度")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(worker, code): code for code in all_codes}
        for future in as_completed(futures):
            ret_code, result = future.result()
            with lock:
                if result == "downloaded":
                    stats["downloaded"] += 1
                elif result == "updated":
                    stats["updated"] += 1
                elif result == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1
                    failed_codes.append(ret_code)
                pbar.update(1)
                pbar.set_postfix(ok=stats["downloaded"]+stats["updated"], skip=stats["skipped"], fail=stats["failed"])
    pbar.close()

    print("\n" + "=" * 50)
    print("🎉 15分钟后复权数据更新完成!")
    print(f"   📥 新下载: {stats['downloaded']} 只")
    print(f"   🔄 增量更新: {stats['updated']} 只")
    print(f"   ✅ 已是最新/跳过: {stats['skipped']} 只")
    print(f"   ❌ 失败: {stats['failed']} 只")
    if failed_codes:
        print(f"   ⚠️ 失败列表 (前20): {failed_codes[:20]}")
    print("=" * 50)

if __name__ == "__main__":
    main()
