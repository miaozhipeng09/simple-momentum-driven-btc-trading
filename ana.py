# factor_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


class AnalysisConfig:
    FEATURE_PATH = 'processed_features/features_20210105.parquet'
    OUTPUT_DIR = 'analysis_results'
    TARGET_WINDOW = 5  # 预测未来5个tick的收益
    QUANTILES = 5


def load_features():
    """加载特征数据"""
    if not Path(AnalysisConfig.FEATURE_PATH).exists():
        raise FileNotFoundError(f"特征文件 {AnalysisConfig.FEATURE_PATH} 不存在")
    return pd.read_parquet(AnalysisConfig.FEATURE_PATH)


def calculate_forward_returns(df):
    """计算前瞻收益"""
    df = df.sort_values('transact_time')
    df['future_return'] = df['price'].pct_change(AnalysisConfig.TARGET_WINDOW).shift(-AnalysisConfig.TARGET_WINDOW)
    return df.dropna()


def analyze_information_coefficient(df):
    """信息系数分析"""
    ic_results = {}
    features = [col for col in df.columns if col.startswith(('return_', 'vwap_', 'ofi_'))]

    for feature in features:
        valid_data = df[[feature, 'future_return']].dropna()
        ic = valid_data[feature].corr(valid_data['future_return'])
        ic_results[feature] = ic

    return pd.Series(ic_results).sort_values(ascending=False)


def plot_quantile_returns(df, feature_name):
    """分位数收益可视化"""
    df = df[[feature_name, 'future_return']].dropna()
    df['quantile'] = pd.qcut(df[feature_name], AnalysisConfig.QUANTILES, labels=False)

    plt.figure(figsize=(10, 6))
    df.groupby('quantile')['future_return'].mean().plot(kind='bar')
    plt.title(f'{feature_name} Quantile Returns')
    plt.xlabel('Quantile Group')
    plt.ylabel('Mean Return')
    plt.savefig(Path(AnalysisConfig.OUTPUT_DIR) / f'{feature_name}_quantiles.png')
    plt.close()


def plot_correlation_matrix(df, top_n=10):
    """特征相关性分析"""
    features = df.filter(like='return_', axis=1).columns[:top_n]
    corr_matrix = df[features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.savefig(Path(AnalysisConfig.OUTPUT_DIR) / 'correlation_matrix.png')
    plt.close()


def generate_analysis_report(ic_results):
    """生成文本报告"""
    report = [
        "因子有效性分析报告",
        "=" * 40,
        f"有效因子数量: {len(ic_results)}",
        f"平均信息系数: {ic_results.mean():.4f}",
        "\nTop 10因子:",
        ic_results.head(10).to_string(),
        "\nBottom 10因子:",
        ic_results.tail(10).to_string()
    ]
    return '\n'.join(report)


def main():
    # 初始化输出目录
    os.makedirs(AnalysisConfig.OUTPUT_DIR, exist_ok=True)

    # 加载数据
    df = load_features()
    df = calculate_forward_returns(df)

    # 执行分析
    ic_results = analyze_information_coefficient(df)

    # 生成可视化
    for feature in ic_results.index[:3]:  # 分析前3个因子
        plot_quantile_returns(df, feature)
    plot_correlation_matrix(df)

    # 生成报告
    report = generate_analysis_report(ic_results)
    print(report)

    # 保存结果
    with open(Path(AnalysisConfig.OUTPUT_DIR) / 'analysis_report.txt', 'w') as f:
        f.write(report)


if __name__ == "__main__":
    main()