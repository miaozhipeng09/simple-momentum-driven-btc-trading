import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from train import BacktestConfig  # 假设原始代码保存为单独文件
from train import ChunkTradingEnv  # 假设环境类已模块化
import os


def load_test_data():
    """专门加载测试数据"""

    def process_day(day):
        try:
            raw_df = pd.read_parquet(BacktestConfig.RAW_DATA_PATTERN.format(day))
            factor_df = pd.read_parquet(BacktestConfig.FACTOR_PATTERN.format(day))

            # 时间处理
            for df in [raw_df, factor_df]:
                df['transact_time'] = pd.to_datetime(df['transact_time'], unit='ms')
                df.sort_values('transact_time', inplace=True)

            merged = pd.merge_asof(raw_df, factor_df, on='transact_time',
                                   direction='nearest', tolerance=pd.Timedelta('1min'))
            merged = merged.dropna(subset=['vwap_mom_50'])

            # 特征工程
            features = merged[['vwap_mom_50', 'vwap_mom_200', 'ofi_50', 'ofi_accel_100']]
            features = (features - features.mean()) / features.std()
            merged['composite_signal'] = features.mean(axis=1)

            return merged
        except FileNotFoundError:
            return None

    test_chunks = []
    test_df = process_day(5)  # 只加载测试日数据
    if test_df is not None:
        for i in range(0, len(test_df), BacktestConfig.CHUNK_SIZE):
            chunk = test_df.iloc[i:i + BacktestConfig.CHUNK_SIZE]
            if len(chunk) > 100:
                test_chunks.append(chunk.reset_index(drop=True))
    return test_chunks


def run_backtest(model_path):
    """执行完整回测流程"""
    # 初始化配置
    os.makedirs(BacktestConfig.OUTPUT_DIR, exist_ok=True)

    # 加载数据
    test_chunks = load_test_data()
    print(f"Loaded {len(test_chunks)} test chunks")

    # 加载模型
    model = PPO.load(model_path)

    # 初始化环境
    env = ChunkTradingEnv(test_chunks)
    obs = env.reset()

    # 执行回测
    results = []
    all_trades = []

    for _ in range(len(test_chunks)):
        action, _ = model.predict(obs)
        obs, _, _, info = env.step(action)

        # 收集交易记录
        chunk_trades = env.trade_log[-info['num_trades']:] if info['num_trades'] > 0 else []
        all_trades.extend(chunk_trades)
        results.append(info)

    # 保存结果
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df.to_csv(os.path.join(BacktestConfig.OUTPUT_DIR, 'production_trades.csv'), index=False)

    # 生成报告
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(BacktestConfig.OUTPUT_DIR, 'production_results.csv'), index=False)

    # 可视化
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.plot(results_df['entry_threshold'], label='Entry Threshold')
    plt.plot(results_df['exit_threshold'], label='Exit Threshold')
    plt.title('Threshold Dynamics')
    plt.legend()

    plt.subplot(122)
    cumulative_returns = results_df['total_return_pct'].cumsum()
    plt.plot(cumulative_returns)
    plt.title(f'Total Cumulative Returns: {cumulative_returns.iloc[-1]:.2f}%')
    plt.savefig(os.path.join(BacktestConfig.OUTPUT_DIR, 'production_performance.png'))
    plt.close()

    print(f"Backtest completed. Results saved to {BacktestConfig.OUTPUT_DIR}")


if __name__ == "__main__":
    model_path = os.path.join(BacktestConfig.OUTPUT_DIR, "tuned_threshold_model")
    run_backtest(model_path)