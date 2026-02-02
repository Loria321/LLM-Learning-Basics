from typing import List, Union, Optional, Tuple
import logging
import pandas as pd
from datetime import datetime
import os

# 1. 定义日志相关路径和文件名（xx采用日期命名，格式：struct_clean_20260202.log）
log_dir = r".\logs\struct_clean"  # 目标日志目录：logs\struct_clean
current_date = datetime.now().strftime("%Y%m%d")  # 获取当前日期，作为xx的替代（更实用）
log_filename = f"struct_clean_{current_date}.log"  # 日志文件名
log_full_path = os.path.join(log_dir, log_filename)  # 拼接完整日志路径，兼容跨平台

# 2. 自动创建多级目录（若不存在），避免FileHandler报错
os.makedirs(log_dir, exist_ok=True)

# 3. 配置日志：仅保存到本地文件，取消控制台输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_full_path, encoding="utf-8")]  # 使用拼接后的完整路径
)
logger = logging.getLogger(__name__)


def handle_high_missing_cols(
    df: pd.DataFrame,
    missing_col_threshold: float = 30.0
) -> pd.DataFrame:
    """
    删除缺失率超过阈值的列（直接修改原DataFrame）
    
    :param df: 输入DataFrame
    :param missing_col_threshold: 列缺失率阈值（%），范围需 0 ≤ threshold ≤ 100
    :return: 删除高缺失列后的原DataFrame
    :raises ValueError: 若阈值超出0-100范围
    """
    # 参数校验
    if not (0 <= missing_col_threshold <= 100):
        raise ValueError("缺失列阈值必须在0-100之间")
    
    # 计算缺失率（直接操作原数据）
    missing_rate = (df.isnull().sum() / len(df) * 100).round(2)
    cols_to_drop = missing_rate[missing_rate > missing_col_threshold].index.tolist()
    
    if cols_to_drop:
        logger.info(f"删除高缺失列：{cols_to_drop}，缺失率分别为：{missing_rate[cols_to_drop].to_dict()}")
        df.drop(columns=cols_to_drop, inplace=True)  # 直接修改原数据
    else:
        logger.info("无高缺失率列需要删除")
    
    return df


def fill_missing_values(
    df: pd.DataFrame,
    missing_fill_strategy: str = "auto"
) -> pd.DataFrame:
    """
    缺失值填充（直接修改原DataFrame）
    
    :param df: 输入DataFrame
    :param missing_fill_strategy: 填充策略
                                  - auto：数值列用中位数，类别列用众数
                                  - mean：数值列用均值
                                  - median：数值列用中位数
                                  - mode：类别列用众数
                                  - drop：删除所有缺失行
    :return: 填充缺失值后的原DataFrame
    :raises ValueError: 若填充策略不合法
    """
    # 校验填充策略
    valid_strategies = ["auto", "mean", "median", "mode", "drop"]
    if missing_fill_strategy not in valid_strategies:
        raise ValueError(f"无效填充策略：{missing_fill_strategy}，支持的策略：{valid_strategies}")
    
    # 删除缺失行（直接修改原数据）
    if missing_fill_strategy == "drop":
        before_rows = len(df)
        df.dropna(inplace=True)
        logger.info(f"删除缺失行：原行数{before_rows}，剩余行数{len(df)}")
        return df
    
    # 分离数值列和类别列
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 优化后（无警告，确保修改原数据）
    if missing_fill_strategy == "mean":
        fill_vals = df[numeric_cols].mean().round(2)
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(fill_vals)  # 改用.loc赋值
        logger.info(f"数值列({numeric_cols.tolist()})用均值填充：{fill_vals.to_dict()}")
    elif missing_fill_strategy in ["median", "auto"]:
        fill_vals = df[numeric_cols].median().round(2)  # 统一格式化精度
        df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(fill_vals)
        logger.info(f"数值列({numeric_cols.tolist()})用中位数填充：{fill_vals.to_dict()}")   
        # 类别列填充（兼容空众数场景，直接修改原数据）
        if not categorical_cols.empty and missing_fill_strategy in ["mode", "auto"]:
            for col in categorical_cols:
                # 处理众数为空的情况（比如列全为NaN）
                mode_series = df[col].mode(dropna=True)
                mode_val = mode_series.iloc[0] if not mode_series.empty else ""
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"类别列({col})用众数填充：{mode_val}")

        return df


def detect_and_handle_outliers(
    df: pd.DataFrame,
    outlier_method: str = "IQR",
    outlier_threshold: float = 5.0
) -> pd.DataFrame:
    """
    异常值检测与处理（直接修改原DataFrame）
    
    :param df: 输入DataFrame
    :param outlier_method: 异常值判定方法：IQR（四分位数）/3σ（标准差）
    :param outlier_threshold: 异常值占比阈值（%），范围0-100；超过则不处理，否则删除异常行
    :return: 处理异常值后的原DataFrame
    :raises ValueError: 若方法不合法或阈值超出范围
    """
    # 参数校验
    valid_methods = ["IQR", "3σ"]
    if outlier_method not in valid_methods:
        raise ValueError(f"无效异常值方法：{outlier_method}，支持的方法：{valid_methods}")
    if not (0 <= outlier_threshold <= 100):
        raise ValueError("异常值阈值必须在0-100之间")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if numeric_cols.empty:
        logger.info("无数值列，跳过异常值处理")
        return df
    
    for col in numeric_cols:
        # 计算异常值边界
        if outlier_method == "3σ":
            mean_val = df[col].mean()
            std_val = df[col].std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
        else:  # IQR（默认）
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        
        # 统计异常值
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outliers_mask.sum()
        outlier_rate = round((outlier_count / len(df) * 100), 2)
        
        # 异常值处理中，补充日志并确认删除逻辑
        if outlier_rate > 0:
            logger.info(f"列{col}：异常值数量{outlier_count}，异常率{outlier_rate}%，阈值{outlier_threshold}%，边界[{lower_bound:.2f}, {upper_bound:.2f}]")
            if outlier_rate <= outlier_threshold:
                df = df[~outliers_mask]  # 改用重新赋值（避免inplace=True的视图问题）
                logger.info(f"删除列{col}的异常行，剩余行数{len(df)}")
            else:
                logger.info(f"列{col}异常率超过阈值，跳过处理")

    return df


def standardize_formats(df: pd.DataFrame) -> pd.DataFrame:
    """
    格式标准化（直接修改原DataFrame）
    - 字符串列：去首尾空格、转大写（跳过NaN）
    - 时间列：自动识别（含time/date/dt关键词）并标准化为datetime格式
    
    :param df: 输入DataFrame
    :return: 格式标准化后的原DataFrame
    """
    # 字符串列处理（避免NaN被转为'NAN'，直接修改原数据）
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        # 仅处理非空值
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip().str.upper()
    logger.info(f"字符串列({str_cols.tolist()})已去空格、转大写")
    
    # 时间列自动识别并标准化（直接修改原数据）
    time_cols = [
        col for col in df.columns 
        if any(key in col.lower() for key in ['time', 'date', 'dt'])
    ]
    for col in time_cols:
        # 1. 保存转换前的原始值（用于统计失败数和提取失败示例）
        original_vals = df[col].copy()
        # 2. 执行时间格式转换（coerce将无效值转为NaT）
        df[col] = pd.to_datetime(df[col], errors='coerce')
        # 3. 统计转换失败的数量和具体值
        # 转换失败 = 原始非空值 但 转换后为NaT
        fail_mask = (original_vals.notna()) & (df[col].isna())
        fail_count = fail_mask.sum()
        # 提取失败值示例（最多展示5个，避免日志过长）
        fail_examples = original_vals[fail_mask].head(5).tolist()
    
        # 4. 输出分级日志
        if fail_count > 0:
            logger.warning(
                f"时间列({col})转换失败{fail_count}条，已转为NaT | 失败值示例：{fail_examples}"
            )
        else:
            logger.info(f"时间列({col})已标准化为datetime格式，无转换失败值")

    return df


def struct_clean_pipeline(
    data: pd.DataFrame,
    col: Optional[str] = None,
    duplicate_threshold: float = 5.0,
    missing_fill_strategy: str = "auto",
    missing_col_threshold: float = 30.0,
    outlier_method: str = "IQR",
    outlier_threshold: float = 5.0
) -> pd.DataFrame:
    """
    通用结构化数据清洗流水线（直接修改原DataFrame）
    
    :param data: 待处理DataFrame
    :param col: 可选，指定仅清洗某一列（None则清洗全量列）
    :param duplicate_threshold: 重复行占比阈值（%），范围0-100；超过则终止清洗
    :param missing_fill_strategy: 缺失值填充策略（参考fill_missing_values）
    :param missing_col_threshold: 列缺失率阈值（%），范围0-100；超过则删除列
    :param outlier_method: 异常值判定方法（参考detect_and_handle_outliers）
    :param outlier_threshold: 异常值占比阈值（%），范围0-100；超过则不处理
    :return: 清洗后的原DataFrame
    :raises TypeError: 若输入非DataFrame
    :raises ValueError: 若数据为空、阈值非法、重复率超阈值
    :raises KeyError: 若指定的col不存在于DataFrame
    """
    # 基础校验
    if not isinstance(data, pd.DataFrame):
        raise TypeError("仅支持DataFrame类型数据，当前输入类型：{}".format(type(data)))
    if data.empty:
        raise ValueError("清洗数据为空，无数据可清洗")
    
    # 阈值参数校验
    if not (0 <= duplicate_threshold <= 100):
        raise ValueError("重复行阈值必须在0-100之间")
    
    # 若指定单列，先校验并聚焦该列（直接操作原数据的视图）
    if col:
        if col not in data.columns:
            raise KeyError(f"指定列{col}不存在于DataFrame，列列表：{data.columns.tolist()}")
        # 临时保存其他列，清洗完成后合并（避免丢失非目标列）
        other_cols = [c for c in data.columns if c != col]
        temp_other_data = data[other_cols].copy() if other_cols else None
        data.drop(columns=other_cols, inplace=True) if other_cols else None
        logger.info(f"仅清洗指定列：{col}")
    
    # 重复行处理（直接修改原数据）
    duplicate_mask = data.duplicated()
    if duplicate_mask.any():
        duplicate_rate = round((duplicate_mask.sum() / len(data) * 100), 2)
        logger.info(f"检测到重复行：数量{duplicate_mask.sum()}，占比{duplicate_rate}%")
        
        if duplicate_rate > duplicate_threshold:
            # 恢复原数据（若指定了单列）
            if col and temp_other_data is not None:
                data = pd.concat([data, temp_other_data], axis=1)
            raise ValueError(f"重复行占比{duplicate_rate}%超过阈值{duplicate_threshold}%，终止清洗")
        
        # 删除重复行（保留首次出现，直接修改原数据）
        data.drop_duplicates(keep='first', inplace=True)
        data.reset_index(drop=True, inplace=True)  # 新增：重置索引
        logger.info(f"删除重复行后，剩余行数：{len(data)}")
    else:
        logger.info("无重复行，跳过重复处理")
    
    # 执行清洗流水线（均直接修改原数据）
    handle_high_missing_cols(data, missing_col_threshold)
    fill_missing_values(data, missing_fill_strategy)
    detect_and_handle_outliers(data, outlier_method, outlier_threshold)
    standardize_formats(data)
    
    # 恢复非目标列（若指定了单列）
    if col and temp_other_data is not None:
        # 按索引对齐合并（而非按行数截取）
        temp_other_data = temp_other_data.loc[data.index]
        data = pd.concat([data, temp_other_data], axis=1)
    
    logger.info("数据清洗完成，最终数据形状：{}".format(data.shape))
    return data

if __name__ == "__main__":
    # 测试数据
    test_df = pd.DataFrame({
        "name": ["  alice  ", "bob", "charlie", None, "bob"],
        "age": [20, 25, 100, 30, 25],
        "score": [85.5, 90.0, 200.0, None, 90.0],
        "create_time": ["2024-01-01", " 2024-02-01 ", "2024-03-01", "invalid_date", "2024-02-01"]
    })
    
    # 全量清洗
    try:
        cleaned_df = struct_clean_pipeline(
            data=test_df,
            duplicate_threshold=10.0,
            missing_fill_strategy="auto",
            missing_col_threshold=30.0,
            outlier_method="IQR",
            outlier_threshold=5.0
        )
        print("清洗后数据：")
        print(cleaned_df)
    except Exception as e:
        logger.error("清洗失败：{}".format(e))
    
    # 单列清洗
    try:
        cleaned_col_df = struct_clean_pipeline(
            data=test_df,
            col="age",
            duplicate_threshold=10.0
        )
        print("\n单列清洗后数据：")
        print(cleaned_col_df)
    except Exception as e:
        logger.error("单列清洗失败：{}".format(e))