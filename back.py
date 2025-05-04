# strategy_backtester.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class BacktestConfig:
    # 输入输出配置
    FACTOR_PATH = 'processed_features/features_selected_2021-01-05.parquet'
    RAW_DATA_PATH = 'BTCUSDT-aggTrade-2021-01-05.parquet'
    OUTPUT_DIR = 'backtest_results'

    # 策略参数（调整后）
    ENTRY_THRESHOLD = 1.8  # 降低做多阈值
    EXIT_THRESHOLD = -2.0  # 降低做空阈值
    FEE_RATE = 0.00005  # 单边交易费率
    INITIAL_CAPITAL = 100000  # 初始资金(USD)


def load_and_merge_data():
    """加载并合并数据（修复时间类型问题）"""
    try:
        print("\n[数据加载] 正在加载数据...")

        # 加载并转换原始数据时间
        raw_df = pd.read_parquet(BacktestConfig.RAW_DATA_PATH)
        raw_df = raw_df[['transact_time', 'price', 'quantity']]
        raw_df['transact_time'] = pd.to_datetime(raw_df['transact_time'], unit='ms')
        print(f"原始数据时间范围：{raw_df['transact_time'].min()} - {raw_df['transact_time'].max()}")

        # 加载并转换因子数据时间
        factor_df = pd.read_parquet(BacktestConfig.FACTOR_PATH)
        factor_df['transact_time'] = pd.to_datetime(factor_df['transact_time'], unit='ms')
        print(f"因子数据时间范围：{factor_df['transact_time'].min()} - {factor_df['transact_time'].max()}")

        # 验证时间对齐
        time_overlap = (factor_df['transact_time'].min() <= raw_df['transact_time'].max()) & \
                       (factor_df['transact_time'].max() >= raw_df['transact_time'].min())
        if not time_overlap:
            raise ValueError("原始数据与因子数据时间范围无重叠")

        # 合并数据
        merged_df = pd.merge_asof(
            raw_df.sort_values('transact_time'),
            factor_df.sort_values('transact_time'),
            on='transact_time',
            direction='nearest',
            tolerance=pd.Timedelta('1min')
        )

        # 数据质量检查
        before = len(raw_df)
        merged_df = merged_df.dropna(subset=['vwap_mom_50', 'vwap_mom_200', 'ofi_50', 'ofi_accel_100'])
        print(f"合并后有效数据：{len(merged_df)}条（丢失{before - len(merged_df)}条）")

        return merged_df

    except Exception as e:
        raise RuntimeError(f"数据加载失败: {str(e)}")


def generate_signals(df):
    """生成交易信号（增强验证）"""
    try:
        # 因子有效性验证
        factors = df[['vwap_mom_50', 'vwap_mom_200', 'ofi_50', 'ofi_accel_100']]

        print("\n[因子统计] 原始因子描述：")
        print(factors.describe().T[['mean', 'std', 'min', 'max']])

        # 标准化处理
        z_scores = (factors - factors.mean()) / factors.std()
        print("\n[标准化统计] Z-Score分布：")
        print(z_scores.describe().T[['mean', 'std', 'min', 'max']])

        # 合成信号
        df['composite_signal'] = z_scores.mean(axis=1)
        print("\n[合成信号] 描述统计：")
        print(df['composite_signal'].describe(percentiles=[0.01, 0.25, 0.75, 0.99]))

        # 生成仓位
        df['position'] = 0
        df.loc[df['composite_signal'] > BacktestConfig.ENTRY_THRESHOLD, 'position'] = 1
        df.loc[df['composite_signal'] < BacktestConfig.EXIT_THRESHOLD, 'position'] = -1

        print("\n[持仓分布]")
        print(df['position'].value_counts(normalize=True))
        return df

    except Exception as e:
        raise RuntimeError(f"信号生成失败: {str(e)}")


def calculate_returns(df):
    """收益计算（增强鲁棒性）"""
    try:
        # 基础收益率
        df['price_return'] = df['price'].pct_change().fillna(0)

        # 策略收益率
        df['strategy_return'] = df['position'].shift(1) * df['price_return']

        # 交易成本计算
        position_changes = df['position'].diff().abs()
        df['transaction_cost'] = position_changes * BacktestConfig.FEE_RATE

        # 净收益率
        df['net_return'] = df['strategy_return'] - df['transaction_cost']

        # 处理异常值
        df['net_return'] = df['net_return'].replace([np.inf, -np.inf], np.nan)
        df['net_return'] = df['net_return'].ffill().fillna(0)

        print("\n[收益率统计]")
        print(df['net_return'].describe())
        return df

    except Exception as e:
        raise RuntimeError(f"收益计算失败: {str(e)}")


def evaluate_performance(return_series):
    """绩效评估（增强稳定性）"""
    if return_series.empty:
        raise ValueError("空收益率序列")

    try:
        # 累计收益
        cumulative_return = (1 + return_series).cumprod()

        # 年化计算参数
        total_minutes = len(return_series)
        annual_factor = 365 * 24 * 60 / total_minutes

        # 关键指标
        metrics = {
            '累计收益': cumulative_return.iloc[-1] - 1,
            '年化收益': cumulative_return.iloc[-1] ** annual_factor - 1,
            '波动率': return_series.std() * np.sqrt(annual_factor),
            '夏普比率': np.nan,
            '最大回撤': (cumulative_return / cumulative_return.cummax() - 1).min()
        }

        # 夏普比率计算保护
        if metrics['波动率'] > 1e-6:
            metrics['夏普比率'] = metrics['年化收益'] / metrics['波动率']

        return {k: float(v) for k, v in metrics.items()}

    except Exception as e:
        raise RuntimeError(f"绩效评估失败: {str(e)}")


def plot_results(df):
    """可视化（增强可读性）"""
    plt.figure(figsize=(16, 12))

    # 价格走势
    plt.subplot(3, 1, 1)
    df.set_index('transact_time')['price'].plot(title='Price Trend', color='navy')
    plt.ylabel('Price (USD)')

    # 合成信号
    plt.subplot(3, 1, 2)
    df.set_index('transact_time')['composite_signal'].plot(
        title='Composite Signal',
        color='darkgreen',
        alpha=0.7
    )
    plt.axhline(BacktestConfig.ENTRY_THRESHOLD, color='r', linestyle='--', label='Entry')
    plt.axhline(BacktestConfig.EXIT_THRESHOLD, color='g', linestyle='--', label='Exit')
    plt.ylabel('Signal Strength')
    plt.legend()

    # 收益对比
    plt.subplot(3, 1, 3)
    df.set_index('transact_time')['strategy_cumulative'].plot(
        label='Strategy',
        color='darkorange'
    )
    df.set_index('transact_time')['benchmark_cumulative'].plot(
        label='Buy & Hold',
        color='purple'
    )
    plt.title('Cumulative Returns')
    plt.ylabel('Return Multiple')
    plt.legend()

    plt.tight_layout()
    plt.savefig(Path(BacktestConfig.OUTPUT_DIR) / 'backtest_result.png')
    plt.close()


def main():
    """主流程（增强诊断）"""
    Path(BacktestConfig.OUTPUT_DIR).mkdir(exist_ok=True)

    try:
        print("\n" + "=" * 40)
        print(" 回测开始 ".center(40, '='))

        # 数据准备
        df = load_and_merge_data()
        print("\n[数据样例]")
        print(df[['transact_time', 'price', 'vwap_mom_50']].head(2))

        # 信号生成
        df = generate_signals(df)
        print("\n[信号样例]")
        print(df[['transact_time', 'composite_signal', 'position']].head())

        # 收益计算
        df = calculate_returns(df)

        # 绩效评估
        df['benchmark_cumulative'] = (1 + df['price_return']).cumprod()
        df['strategy_cumulative'] = (1 + df['net_return']).cumprod()
        performance = evaluate_performance(df['net_return'].dropna())

        print("\n[最终绩效]")
        for k, v in performance.items():
            print(f"{k:8}: {v:.4f}")

        # 可视化保存
        plot_results(df)
        df.to_parquet(Path(BacktestConfig.OUTPUT_DIR) / 'detailed_results.parquet')
        print("\n结果已保存至:", BacktestConfig.OUTPUT_DIR)

    except Exception as e:
        print("\n" + "!" * 40)
        print(f" 错误: {str(e)} ".center(40, '!'))
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 40)
        print(" 回测结束 ".center(40, '='))


if __name__ == "__main__":
    main()