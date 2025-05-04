import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from numba import jit
from tqdm import tqdm


# 配置类，定义数据路径、特征计算窗口和分块重叠大小
class Config:
    RAW_DATA_PATH = 'BTCUSDT-aggTrade-2021-01-10.parquet'  # 原始数据文件路径
    FEATURE_SAVE_PATH = 'processed_features/features_20210110.parquet'  # 处理后特征数据存储路径
    FEATURE_WINDOWS = [50, 100, 200]  # 计算动量和订单流特征的窗口大小
    OUTPUT_DIR = 'processed_features'  # 处理后数据存储目录
    OVERLAP_SIZE = max(FEATURE_WINDOWS) * 2  # 处理数据时的重叠窗口大小，确保分块计算一致性


@jit(nopython=True)
def _rolling_linreg(x, y, window):
    """ 计算滚动窗口内的线性回归斜率
    用于量价协同特征计算，以衡量价格变化与交易量变化的相关性 """
    n = len(x)
    slope = np.full(n, np.nan)

    for i in range(window, n):
        x_window = x[i - window:i]
        y_window = y[i - window:i]
        x_mean = np.mean(x_window)
        y_mean = np.mean(y_window)

        numerator = np.sum((x_window - x_mean) * (y_window - y_mean))
        denominator = np.sum((x_window - x_mean) ** 2)

        slope[i] = numerator / denominator if denominator != 0 else 0
    return slope


def calculate_features(df_chunk, last_chunk=None):
    """ 计算交易数据的特征，包括动量因子、订单流因子和量价协同因子
    采用分块计算方式，确保数据连续性并避免未来函数问题 """
    # 处理数据分块之间的重叠，确保连续计算
    if last_chunk is not None:
        df = pd.concat([last_chunk, df_chunk], ignore_index=True)
    else:
        df = df_chunk.copy()

    df = df.sort_values('transact_time').reset_index(drop=True)  # 确保按交易时间排序

    # ===== 计算基础特征 =====
    df['side'] = np.where(~df['is_buyer_maker'], 1, -1)  # 买方交易记为1，卖方交易记为-1
    df['return'] = df['price'].pct_change()  # 计算价格的对数收益率
    df['signed_volume'] = df['quantity'] * df['side']  # 计算带方向的成交量

    # ===== 计算动量因子 =====
    for w in Config.FEATURE_WINDOWS:
        df[f'return_{w}'] = -df['price'].pct_change(w)  # 价格动量（负号表示回归分析常用定义）

        # 计算历史VWAP（成交量加权平均价格），避免未来数据泄露
        price_vol = df['price'] * df['quantity']
        sum_price_vol = price_vol.rolling(w, min_periods=1).sum()
        sum_vol = df['quantity'].rolling(w, min_periods=1).sum()
        vwap = (sum_price_vol / sum_vol).shift(1)  # 关键修正：使用历史VWAP，避免未来函数
        df[f'vwap_mom_{w}'] = -(df['price'] / vwap - 1)  # 计算VWAP动量

    # ===== 计算订单流因子 =====
    for w in Config.FEATURE_WINDOWS:
        sum_signed_vol = df['signed_volume'].rolling(w, min_periods=1).sum()
        sum_vol = df['quantity'].rolling(w, min_periods=1).sum()
        df[f'ofi_{w}'] = sum_signed_vol / sum_vol  # 计算订单流不平衡指标（OFI）
        df[f'ofi_accel_{w}'] = df[f'ofi_{w}'].diff(w // 2)  # 计算订单流加速度

    # ===== 计算量价协同特征 =====
    price_diff = df['price'].diff().values
    volume_diff = df['quantity'].diff().values
    df['pv_cos_sim'] = _rolling_linreg(
        np.arange(len(price_diff)),
        price_diff * volume_diff,
        window=100  # 使用100笔交易窗口计算线性回归斜率
    )

    # 处理数据块的重叠部分，确保连续计算
    if last_chunk is not None:
        valid_data = df.iloc[len(last_chunk):-Config.OVERLAP_SIZE]  # 移除重叠部分
        new_last = df.iloc[-Config.OVERLAP_SIZE:]  # 保留最后的重叠部分供下一块计算
    else:
        valid_data = df.iloc[:-Config.OVERLAP_SIZE]
        new_last = df.iloc[-Config.OVERLAP_SIZE:]

    return valid_data.dropna(), new_last


def process_and_save_features():
    """ 读取原始交易数据，按分块计算特征，并保存最终结果 """
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在

    # 读取Parquet文件并初始化进度条
    parquet_file = pq.ParquetFile(Config.RAW_DATA_PATH)
    chunks = []  # 用于存储处理后的数据块
    last_chunk = None  # 存储上一块数据的尾部，确保计算连续性
    progress = tqdm(total=parquet_file.num_row_groups, desc='Processing Data')

    # 按块读取和处理数据
    for i in range(parquet_file.num_row_groups):
        df_chunk = parquet_file.read_row_group(i).to_pandas()  # 读取当前数据块
        processed, last_chunk = calculate_features(df_chunk, last_chunk)  # 计算特征
        chunks.append(processed)
        progress.update(1)

    progress.close()

    # 合并所有数据块
    full_df = pd.concat(chunks, ignore_index=True)

    # 过滤前期无效数据（避免初始窗口影响）
    max_window = max(Config.FEATURE_WINDOWS)
    full_df = full_df.iloc[max_window * 2:]

    # 保存计算后的特征数据
    full_df.to_parquet(Config.FEATURE_SAVE_PATH)
    print(f"特征数据已存储至：{os.path.abspath(Config.FEATURE_SAVE_PATH)}")
    print(f"最终数据量：{len(full_df)} 条，时间范围：{full_df['transact_time'].min()} 至 {full_df['transact_time'].max()}")


# 主程序入口，执行特征计算和保存
if __name__ == "__main__":
    process_and_save_features()
