import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import akshare as ak
import pandas as pd
import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
DATA_DIR = "data_stock_daily_unadj"  # 不复权数据目录
START_DATE = "20100101"              # 全量下载起始日期
MAX_RETRY = 3                        # 单只股票最大重试次数
REQUEST_INTERVAL = 0.3               # 每次请求间隔(秒), 避免被封IP
WORKERS = 4                          # 并发进程数
# ===========================================

# A股个股前缀
VALID_PREFIXES = ('000', '001', '002', '003', '300', '301', '600', '601', '603', '605', '688', '689')


def is_valid_stock(code):
    """判断是否为有效的A股个股代码 (排除指数、B股等)"""
    if len(code) != 6 or not code.isdigit():
        return False
    return code.startswith(VALID_PREFIXES)


def code_to_sina_symbol(code):
    """将纯数字代码转换为新浪格式: 600xxx -> sh600xxx, 000xxx -> sz000xxx"""
    if code.startswith('6'):
        return f"sh{code}"
    else:
        return f"sz{code}"


def get_all_stock_codes():
    """从已有数据目录获取A股个股代码列表"""
    source_dirs = ["data_stock_daily", "data_stock_daily_hfq", DATA_DIR]
    all_codes = set()

    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            continue
        for f in os.listdir(source_dir):
            if not f.endswith('.csv'):
                continue
            code = f.replace('.csv', '')
            if is_valid_stock(code):
                all_codes.add(code)

    if not all_codes:
        print("❌ 未找到任何数据目录, 尝试从 akshare 获取全 A 股列表...")
        try:
            df_list = ak.stock_zh_a_spot_em()
            for _, row in df_list.iterrows():
                code = str(row["代码"]).zfill(6)
                if is_valid_stock(code):
                    all_codes.add(code)
        except Exception as e:
            print(f"❌ 获取股票列表失败: {e}")

    print(f"   📂 有效个股: {len(all_codes)} 只")
    return sorted(all_codes)


def get_local_last_date(file_path):
    """读取本地CSV最后日期"""
    try:
        df = pd.read_csv(file_path, usecols=['日期'])
        if len(df) == 0:
            return None
        return pd.to_datetime(df['日期']).max().strftime("%Y%m%d")
    except Exception:
        return None


def download_one_stock(code, start_date, end_date):
    """下载单只股票的不复权日线数据 (新浪 stock_zh_a_daily, adjust='')"""
    sina_symbol = code_to_sina_symbol(code)
    for attempt in range(MAX_RETRY):
        try:
            df = ak.stock_zh_a_daily(
                symbol=sina_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=""          # 不复权
            )
            if df is None or df.empty:
                return code, None, "empty"

            # 新浪返回列: date, open, high, low, close, volume, amount, outstanding_share, turnover
            col_map = {'date': '日期', 'close': '收盘', 'open': '开盘', 'turnover': '换手率'}
            for eng_col in col_map:
                if eng_col not in df.columns:
                    if eng_col == 'turnover':
                        df['turnover'] = 0.0
                    else:
                        return code, None, f"missing_col:{eng_col}"

            df_out = df[list(col_map.keys())].copy()
            df_out.rename(columns=col_map, inplace=True)
            df_out['日期'] = pd.to_datetime(df_out['日期']).dt.strftime('%Y-%m-%d')
            # 换手率: 新浪返回小数形式(0.0123), 转为百分比
            df_out['换手率'] = (df_out['换手率'] * 100).round(4)
            return code, df_out, "ok"
        except KeyError:
            return code, None, "empty"
        except (ConnectionError, ConnectionResetError, TimeoutError, OSError) as e:
            if attempt < MAX_RETRY - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return code, None, f"network: {str(e)[:80]}"
        except Exception as e:
            err_str = str(e)
            if '不存在' in err_str or '无数据' in err_str or '404' in err_str:
                return code, None, "empty"
            if attempt < MAX_RETRY - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return code, None, err_str
    return code, None, "max_retry"


def process_stock(args):
    """处理单只股票: 判断增量/全量, 下载, 保存 (进程池worker入口)"""
    code, end_date_str = args
    file_path = os.path.join(DATA_DIR, f"{code}.csv")

    if os.path.exists(file_path):
        last_date = get_local_last_date(file_path)
        if last_date is None:
            start_date = START_DATE
            mode = "full"
        elif last_date >= end_date_str:
            return code, "skipped"
        else:
            next_day = (pd.to_datetime(last_date) + datetime.timedelta(days=1)).strftime("%Y%m%d")
            start_date = next_day
            mode = "append"
    else:
        start_date = START_DATE
        mode = "full"

    code_ret, df_new, status = download_one_stock(code, start_date, end_date_str)

    if status != "ok" or df_new is None or df_new.empty:
        if status == "empty":
            return code, "skipped"
        return code, f"failed: {status}"

    if mode == "append":
        try:
            df_existing = pd.read_csv(file_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['日期'], keep='last', inplace=True)
            df_combined.to_csv(file_path, index=False, encoding='utf-8')
            return code, "updated"
        except Exception as e:
            return code, f"failed: {e}"
    else:
        df_new.to_csv(file_path, index=False, encoding='utf-8')
        return code, "downloaded"


def main():
    today_str = datetime.date.today().strftime("%Y%m%d")
    print(f"🚀 不复权数据更新引擎 (新浪 stock_zh_a_daily) | adjust=''")
    print(f"   📅 起始日期: {START_DATE[:4]}-{START_DATE[4:6]}-{START_DATE[6:]}")
    print(f"   📅 目标日期: {today_str}")
    print(f"   🧵 并发进程: {WORKERS}")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    all_codes = get_all_stock_codes()
    print(f"📊 目标股票数: {len(all_codes)} 只")

    if not all_codes:
        print("❌ 股票列表为空")
        return

    stats = {"downloaded": 0, "updated": 0, "skipped": 0, "failed": 0}
    failed_codes = []

    task_args = [(code, today_str) for code in all_codes]

    pbar = tqdm(total=len(all_codes), desc="下载进度")
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_stock, arg): arg[0] for arg in task_args}
        for future in as_completed(futures):
            try:
                ret_code, result = future.result()
            except Exception as e:
                ret_code = futures[future]
                result = f"failed: {e}"

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
    print("🎉 不复权数据更新完成!")
    print(f"   📥 新下载: {stats['downloaded']} 只")
    print(f"   🔄 增量更新: {stats['updated']} 只")
    print(f"   ✅ 已是最新: {stats['skipped']} 只")
    print(f"   ❌ 失败: {stats['failed']} 只")
    if failed_codes:
        print(f"   ⚠️ 失败列表 (前20): {failed_codes[:20]}")
    print("=" * 50)


if __name__ == "__main__":
    main()
