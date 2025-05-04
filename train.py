"""
基于强化学习的量化交易策略优化系统

主要功能：
1. 使用PPO算法动态优化交易阈值参数
2. 支持分块数据处理，解决内存限制问题
3. 包含完整的训练、验证、回测流程
4. 提供可视化分析功能
"""

# 第三方库导入
import pandas as pd
import numpy as np
import gym  # 强化学习环境框架
from gym import spaces
from stable_baselines3 import PPO  # 近端策略优化算法实现
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt  # 可视化
import torch  # 神经网络框架
import os
from datetime import datetime


class BacktestConfig:
    """回测系统全局配置类"""
    FACTOR_PATTERN = 'processed_features/features_selected_2021-01-0{}.parquet'  # 特征数据路径模板
    RAW_DATA_PATTERN = 'BTCUSDT-aggTrade-2021-01-0{}.parquet'  # 原始数据路径模板
    OUTPUT_DIR = 'backtest_results'  # 结果输出目录
    FEE_RATE = 0.0001
    CHUNK_SIZE = 5000  # 数据分块大小


class ChunkTradingEnv(gym.Env):
    """分块式交易环境(支持动态阈值调整)

    功能特点：
    - 将大数据集分块处理，避免内存溢出
    - 动态调整交易信号阈值
    - 支持多日滚动训练
    - 集成交易成本计算
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_chunks):
        super(ChunkTradingEnv, self).__init__()
        self.data_chunks = data_chunks  # 分块数据集
        self.current_chunk_idx = -1  # 当前处理块索引
        self.stats_buffer = []  # 统计量缓存

        # 观察空间定义(标准化后范围)
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, -5, 0, 0.0, -4.0]),  # 信号统计量+阈值参数
            high=np.array([5, 5, 5, 5, 4.0, 0.0]),
            dtype=np.float32
        )

        # 动作空间定义(阈值调整幅度)
        self.action_space = spaces.Box(
            low=np.array([-0.2, -0.2]),  # [entry调整幅度, exit调整幅度]
            high=np.array([0.2, 0.2]),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        """环境重置方法"""
        # 滚动到下一个数据块
        self.current_chunk_idx = (self.current_chunk_idx + 1) % len(self.data_chunks)
        # 获取前一个块的统计量
        prev_stats = self.stats_buffer[-1] if self.stats_buffer else {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 1.0}

        # 构建观察向量
        self.observation = np.array([
            prev_stats['mean'],  # 信号均值
            prev_stats['max'],  # 信号最大值
            prev_stats['min'],  # 信号最小值
            prev_stats['std'],  # 信号标准差
            self.entry_threshold,  # 当前入场阈值
            self.exit_threshold  # 当前离场阈值
        ], dtype=np.float32)

        # 重置交易状态
        self.position = 0  # 当前持仓(1: 多, -1: 空, 0: 平)
        self.entry_price = None  # 入场价格
        self.trade_log = []  # 交易记录
        return self.observation

    def step(self, action):
        """执行交易动作的核心方法"""
        # 更新阈值参数(限制在[0,4]和[-4,0]区间)
        self.entry_threshold = np.clip(self.entry_threshold + action[0], 0.0, 4.0)
        self.exit_threshold = np.clip(self.exit_threshold + action[1], -4.0, 0.0)

        # 处理当前数据块
        prev_trade_count = len(self.trade_log)
        chunk = self.data_chunks[self.current_chunk_idx]
        current_position = 0

        # 遍历块内数据执行交易
        for idx, row in chunk.iterrows():
            price = row['price']
            signal = row['composite_signal']
            new_position = self._determine_position(signal, current_position)

            # 记录交易信息
            if new_position != current_position:
                trade = {
                    'timestamp': row['transact_time'],
                    'old_pos': current_position,
                    'new_pos': new_position,
                    'price': price,
                    'size_change': abs(new_position - current_position),
                    'fee_pct': abs(new_position - current_position) * BacktestConfig.FEE_RATE
                }

                # 计算收益
                if current_position != 0:
                    price_return = ((price - self.entry_price) / self.entry_price) * current_position
                    trade['return_pct'] = price_return - trade['fee_pct']
                else:
                    trade['return_pct'] = -trade['fee_pct']

                self.entry_price = price if new_position != 0 else None
                current_position = new_position
                self.trade_log.append(trade)

        # 记录信号统计量
        current_stats = {
            'mean': chunk['composite_signal'].mean(),
            'max': chunk['composite_signal'].max(),
            'min': chunk['composite_signal'].min(),
            'std': chunk['composite_signal'].std()
        }
        self.stats_buffer.append(current_stats)

        # 构建回报信息
        chunk_trades = self.trade_log[prev_trade_count:]
        total_return = sum(t['return_pct'] for t in chunk_trades) * 100
        avg_return = np.mean([t['return_pct'] for t in chunk_trades]) * 100 if chunk_trades else 0.0

        info = {
            'chunk_index': self.current_chunk_idx,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'total_return_pct': total_return,
            'avg_return_pct': avg_return,
            'num_trades': len(chunk_trades),
            'timestamp': chunk['transact_time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
        }

        # 计算奖励函数(收益主导+动作正则)
        reward = total_return * 5.0 - 0.001 * (action[0] ** 2 + action[1] ** 2)
        return self.reset(), reward, False, info

    def _determine_position(self, signal, current_pos):
        """基于信号和阈值的仓位决策逻辑"""
        if current_pos == 0:
            if signal > self.entry_threshold:
                return 1  # 开多仓
            elif signal < self.exit_threshold:
                return -1  # 开空仓
            else:
                return 0  # 保持空仓
        elif current_pos == 1:
            return 0 if signal < self.exit_threshold else 1  # 平多仓条件
        elif current_pos == -1:
            return 0 if signal > self.entry_threshold else -1  # 平空仓条件

    def render(self, mode='human'):
        """环境状态可视化"""
        if not self.trade_log:
            print("No trades executed yet")
            return
        last_trade = self.trade_log[-1]
        print(f"Thresholds: Entry={self.entry_threshold:.2f}, Exit={self.exit_threshold:.2f}")
        print(
            f"Last trade: {last_trade['timestamp']} | Position change: {last_trade['old_pos']}->{last_trade['new_pos']}")


def load_split_data():
    """数据加载与预处理模块

    功能：
    1. 合并原始交易数据和特征数据
    2. 生成复合交易信号
    3. 标准化特征数据
    4. 分割训练集(2-4日)和测试集(5日)
    """

    def process_day(day):
        """单日数据处理流水线"""
        try:
            # 数据加载
            raw_df = pd.read_parquet(BacktestConfig.RAW_DATA_PATTERN.format(day))
            factor_df = pd.read_parquet(BacktestConfig.FACTOR_PATTERN.format(day))

            # 时间处理
            for df in [raw_df, factor_df]:
                if not pd.api.types.is_datetime64_any_dtype(df['transact_time']):
                    df['transact_time'] = pd.to_datetime(df['transact_time'], unit='ms')
                df.sort_values('transact_time', inplace=True)

            # 时间对齐合并
            merged = pd.merge_asof(raw_df, factor_df, on='transact_time',
                                   direction='nearest', tolerance=pd.Timedelta('1min'))
            merged = merged.dropna(subset=['vwap_mom_50'])

            # 特征工程
            features = merged[['vwap_mom_50', 'vwap_mom_200', 'ofi_50', 'ofi_accel_100']]
            features = (features - features.mean()) / features.std()  # Z-score标准化
            merged['composite_signal'] = features.mean(axis=1)  # 合成信号

            return merged
        except FileNotFoundError:
            return None

    # 训练数据(2-4日)
    train_chunks = []
    for day in [2, 3, 4]:
        df = process_day(day)
        if df is not None:
            # 分块处理
            for i in range(0, len(df), BacktestConfig.CHUNK_SIZE):
                chunk = df.iloc[i:i + BacktestConfig.CHUNK_SIZE]
                if len(chunk) > 100:  # 过滤过小分块
                    train_chunks.append(chunk.reset_index(drop=True))

    # 测试数据(5日)
    test_chunks = []
    test_df = process_day(5)
    if test_df is not None:
        for i in range(0, len(test_df), BacktestConfig.CHUNK_SIZE):
            chunk = test_df.iloc[i:i + BacktestConfig.CHUNK_SIZE]
            if len(chunk) > 100:
                test_chunks.append(chunk.reset_index(drop=True))

    print(f"Loaded {len(train_chunks)} train chunks, {len(test_chunks)} test chunks")
    return train_chunks, test_chunks


class ThresholdCallback(BaseCallback):
    """阈值调整过程监控回调

    功能：
    1. 记录阈值调整历史
    2. 监控训练过程稳定性
    """

    def __init__(self, check_freq=500):
        super().__init__()
        self.check_freq = check_freq  # 监控频率
        self.threshold_history = []  # 历史记录存储

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            env = self.model.env.envs[0]
            self.threshold_history.append({
                'entry': env.entry_threshold,
                'exit': env.exit_threshold,
                'step': self.n_calls
            })
        return True


def backtest(test_chunks, model):
    """回测执行模块

    功能：
    1. 在测试集上评估模型表现
    2. 保存交易细节记录
    3. 生成可视化报告
    """
    env = ChunkTradingEnv(test_chunks)
    obs = env.reset()
    results = []
    all_trades = []

    # 遍历测试数据块
    for _ in range(len(test_chunks)):
        action, _ = model.predict(obs)
        obs, _, _, info = env.step(action)

        # 收集交易信息
        chunk_trades = env.trade_log[-info['num_trades']:] if info['num_trades'] > 0 else []
        all_trades.extend(chunk_trades)
        results.append(info)

    # 保存交易明细
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df.to_csv(os.path.join(BacktestConfig.OUTPUT_DIR, 'trades_details.csv'), index=False)

    # 可视化分析
    plt.figure(figsize=(14, 6))

    # 阈值动态图
    plt.subplot(1, 2, 1)
    entry_thresholds = [r['entry_threshold'] for r in results]
    exit_thresholds = [r['exit_threshold'] for r in results]
    plt.plot(entry_thresholds, label='Entry Threshold')
    plt.plot(exit_thresholds, label='Exit Threshold')
    plt.title(f'Threshold Dynamics\nFinal Entry: {entry_thresholds[-1]:.2f}, Exit: {exit_thresholds[-1]:.2f}')
    plt.ylim(-4.5, 4.5)
    plt.legend()

    # 收益曲线图
    plt.subplot(1, 2, 2)
    cumulative_returns = np.cumsum([r['total_return_pct'] for r in results])
    plt.plot(cumulative_returns)
    plt.title(f'Cumulative Returns: {cumulative_returns[-1]:.2f}%')
    plt.tight_layout()
    plt.savefig(os.path.join(BacktestConfig.OUTPUT_DIR, 'backtest_results.png'))

    return pd.DataFrame(results)


def main():
    """主执行流程"""
    os.makedirs(BacktestConfig.OUTPUT_DIR, exist_ok=True)
    train_chunks, test_chunks = load_split_data()

    # 模型训练配置
    env = DummyVecEnv([lambda: ChunkTradingEnv(train_chunks)])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "net_arch": [256, 256, 256],  # 3层256节点网络
            "activation_fn": torch.nn.LeakyReLU,  # 激活函数选择
            "optimizer_kwargs": {"weight_decay": 0.0001}  # L2正则化
        },
        learning_rate=1e-4,  # 学习率
        n_steps=4096,  # 每次更新的步数
        batch_size=256,  # 批量大小
        n_epochs=5,  # 每次更新的训练轮次
        gamma=0.995,  # 折扣因子
        verbose=1  # 训练过程输出
    )

    # 模型训练
    model.learn(
        total_timesteps=len(train_chunks) * 1,  # 总训练步数
        callback=ThresholdCallback(),  # 回调函数
        tb_log_name="extended_threshold"  # 实验名称
    )
    model.save(os.path.join(BacktestConfig.OUTPUT_DIR, "extended_threshold_model"))

    # 执行回测
    results_df = backtest(test_chunks, model)
    results_df.to_csv(os.path.join(BacktestConfig.OUTPUT_DIR, 'backtest_summary.csv'), index=False)
    print("Backtest completed. Results saved to output directory.")


if __name__ == "__main__":
    main()