import pandas as pd
from pathlib import Path


# ====================== 配置参数 ======================
class Config:
    # 输入输出配置
    INPUT_PATH = 'processed_features/features_20210110.parquet'
    OUTPUT_PATH = 'processed_features/features_selected_20210110.parquet'

    # 选定因子列表（根据分析结果）
    SELECTED_FACTORS = [
        'vwap_mom_50',
        'vwap_mom_200',
        'ofi_50',
        'ofi_accel_100',
        'transact_time'  # 保留时间戳用于后续分析
    ]


# ====================== 核心处理逻辑 ======================
def extract_and_save_factors():
    """因子提取与保存主函数"""
    try:
        # 1. 校验输入文件
        input_file = Path(Config.INPUT_PATH)
        if not input_file.exists():
            raise FileNotFoundError(f"输入文件 {Config.INPUT_PATH} 不存在")

        # 2. 读取数据
        df = pd.read_parquet(input_file)
        print(f"成功加载数据，原始维度：{df.shape}")

        # 3. 校验因子列
        missing_factors = [f for f in Config.SELECTED_FACTORS if f not in df.columns]
        if missing_factors:
            raise KeyError(f"以下因子不存在: {missing_factors}")

        # 4. 提取选定列
        selected_df = df[Config.SELECTED_FACTORS]
        print(f"提取后维度：{selected_df.shape}")

        # 5. 确保输出目录存在
        output_path = Path(Config.OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 6. 保存数据（使用高效压缩）
        selected_df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='zstd',
            index=False
        )
        print(f"成功保存至：{output_path.absolute()}")

    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise


# ====================== 执行入口 ======================
if __name__ == "__main__":
    extract_and_save_factors()