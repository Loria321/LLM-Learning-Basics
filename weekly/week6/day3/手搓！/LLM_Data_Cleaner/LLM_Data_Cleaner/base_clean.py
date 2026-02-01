import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import argparse  # 新增：命令行参数支持
import traceback  # 新增：详细异常栈追踪

# ===================== 1. 日志配置（保留并优化） =====================
def setup_logger(log_path):
    """配置日志：同时输出到控制台和日志文件，记录清洗全流程"""
    # 日志文件名包含时间戳，避免覆盖
    log_file = f"{log_path}_清洗日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志格式
    logger = logging.getLogger("data_cleaner")
    logger.setLevel(logging.INFO)
    # 清空已有处理器（避免重复打印）
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger, log_file

# ===================== 2. 核心清洗函数（强化异常处理） =====================
def clean_csv_data(
    input_path,          # 输入CSV文件路径
    output_path,         # 输出清洗后CSV路径
    log_path="清洗日志",  # 日志文件基础路径
    duplicate_threshold=5.0,  # 重复行占比阈值（%），超过则终止
    missing_fill_strategy="auto",  # 缺失值填充策略：auto/mean/median/mode/drop
    missing_col_threshold=30.0,    # 列缺失率阈值（%），超过则删除列
    outlier_method="IQR",          # 异常值判定方法：IQR/3σ
    outlier_threshold=5.0          # 异常值占比阈值（%），超过则提示
):
    """
    通用CSV数据清洗函数（支持参数配置+强化异常处理）
    :param input_path: 输入CSV文件路径
    :param output_path: 输出清洗后CSV路径
    :param log_path: 日志文件保存基础路径
    :param duplicate_threshold: 重复行占比阈值（%），>该值则终止清洗
    :param missing_fill_strategy: 缺失值填充策略
                                  - auto：数值列用median，类别列用mode
                                  - mean：数值列用均值
                                  - median：数值列用中位数
                                  - mode：类别列用众数
                                  - drop：删除所有缺失行
    :param missing_col_threshold: 列缺失率阈值（%），>该值删除列
    :param outlier_method: 异常值判定方法（IQR/3σ）
    :param outlier_threshold: 异常值占比阈值（%），>该值仅提示不处理
    :return: 清洗后DataFrame、日志文件路径 | 异常时返回None, 日志路径
    """
    # 初始化日志
    logger, log_file = setup_logger(log_path)
    logger.info("="*50)
    logger.info("开始执行数据清洗流程")
    logger.info(f"输入文件：{input_path}")
    logger.info(f"配置参数：重复行阈值={duplicate_threshold}% | 缺失列阈值={missing_col_threshold}% | 缺失填充策略={missing_fill_strategy} | 异常值方法={outlier_method}")
    logger.info("="*50)

    try:
        # -------------------- 新增：前置格式校验 --------------------
        logger.info("【前置校验】检查文件格式和存在性")
        # 1. 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在：{input_path}")
        
        # 2. 检查输入文件格式（仅支持CSV）
        if not input_path.lower().endswith('.csv'):
            raise ValueError(f"输入文件格式错误！仅支持CSV文件，当前文件：{input_path}")
        
        # 3. 检查输出文件格式（仅支持CSV）
        if not output_path.lower().endswith('.csv'):
            raise ValueError(f"输出文件格式错误！仅支持CSV文件，当前文件：{output_path}")

        # -------------------- 步骤1：数据加载 --------------------
        logger.info("【步骤1：数据加载】")
        df = pd.read_csv(input_path, encoding='utf-8')
        original_shape = df.shape
        logger.info(f"原始数据维度：{original_shape[0]}行 × {original_shape[1]}列")
        
        if df.empty:
            raise ValueError("加载的CSV文件为空，无数据可清洗")

        # -------------------- 步骤2：探索性分析（日志记录，修复重复打印） --------------------
        logger.info("\n【步骤2：探索性分析】")
        # 数据类型
        logger.info(f"数据类型分布：\n{df.dtypes.to_string()}")
        # 缺失值统计
        missing_sum = df.isnull().sum()
        missing_rate = (missing_sum / len(df) * 100).round(2)
        missing_info = missing_rate[missing_rate > 0].to_string() if any(missing_rate > 0) else "无缺失值"
        logger.info(f"缺失值分布（列）：\n{missing_info}")
        # 描述性统计（仅打印一次）
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            desc_stats = df[numeric_cols].describe().round(2).to_string()
            logger.info(f"数值列描述性统计：\n{desc_stats}")

        # -------------------- 步骤3：去重 --------------------
        logger.info("\n【步骤3：去重处理】")
        duplicate_count = df.duplicated().sum()
        duplicate_rate = (duplicate_count / len(df) * 100).round(2)
        logger.info(f"重复行数量：{duplicate_count} | 重复行占比：{duplicate_rate}%")
        
        if duplicate_rate > duplicate_threshold:
            raise ValueError(f"重复行占比（{duplicate_rate}%）超过阈值（{duplicate_threshold}%），终止清洗")
        elif duplicate_count > 0:
            df = df.drop_duplicates(keep='first')
            logger.info(f"已删除重复行，当前数据维度：{df.shape[0]}行 × {df.shape[1]}列")

        # -------------------- 步骤4：缺失值处理 --------------------
        logger.info("\n【步骤4：缺失值处理】")
        # 先处理列缺失率超过阈值的列
        for col in df.columns:
            col_missing_rate = (df[col].isnull().sum() / len(df) * 100).round(2)
            if col_missing_rate > missing_col_threshold:
                logger.info(f"列[{col}]缺失率{col_missing_rate}% > 阈值{missing_col_threshold}%，删除该列")
                df = df.drop(columns=[col])
                continue
            
            # 处理列内缺失值
            if df[col].isnull().sum() == 0:
                continue
            
            if missing_fill_strategy == "drop":
                df = df.dropna(subset=[col])
                logger.info(f"列[{col}]：删除缺失行，当前行数：{len(df)}")
            else:
                # 根据策略选择填充值
                if df[col].dtype in ['int64', 'float64']:
                    if missing_fill_strategy == "mean":
                        fill_val = df[col].mean().round(2)
                    elif missing_fill_strategy == "median":
                        fill_val = df[col].median()
                    else:  # auto/默认
                        fill_val = df[col].median()
                else:
                    fill_val = df[col].mode()[0]  # 类别列用众数
                
                df[col] = df[col].fillna(fill_val)
                logger.info(f"列[{col}]：填充缺失值（策略={missing_fill_strategy} | 填充值={fill_val}）")

        # -------------------- 步骤5：异常值处理（仅数值列） --------------------
        logger.info("\n【步骤5：异常值处理】")
        for col in numeric_cols:
            if col not in df.columns:  # 避免列已被删除
                continue
            
            # 异常值判定
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
            
            # 筛选异常值
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_rate = round((outlier_count / len(df) * 100),2)
            logger.info(f"列[{col}]：异常值数量={outlier_count} | 占比={outlier_rate}% | 判定范围=[{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # 异常值处理：占比≤阈值则删除，超过则仅提示
            if outlier_rate > 0:
                if outlier_rate <= outlier_threshold:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    logger.info(f"列[{col}]：已删除异常行，当前行数：{len(df)}")
                else:
                    logger.warning(f"列[{col}]：异常值占比超过阈值（{outlier_threshold}%），请排查数据采集问题，暂不处理")

        # -------------------- 步骤6：格式标准化 --------------------
        logger.info("\n【步骤6：格式标准化】")
        # 字符串列：去空格、统一大写
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().str.upper()
            logger.info(f"列[{col}]：完成字符串标准化（去空格+大写）")
        
        # 时间列：自动识别并标准化
        time_cols = [col for col in df.columns if any(key in col.lower() for key in ['time', 'date', 'dt'])]
        for col in time_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logger.info(f"列[{col}]：标准化为datetime格式")

        # -------------------- 步骤7：数据保存 --------------------
        logger.info("\n【步骤7：数据保存】")
        # 新增：检查输出目录是否存在，不存在则创建
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"输出目录不存在，已创建：{output_dir}")
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        final_shape = df.shape
        logger.info(f"清洗完成！输出文件：{output_path}")
        logger.info(f"最终数据维度：{final_shape[0]}行 × {final_shape[1]}列")
        logger.info(f"数据清洗总览：删除重复行{original_shape[0]-df.shape[0]}行 | 保留列{final_shape[1]}列")
        logger.info("="*50)

        return df, log_file

    except Exception as e:
        # 新增：详细记录异常栈，便于排查
        logger.error(f"清洗过程出错：{str(e)}", exc_info=True)
        logger.error(f"异常详细栈信息：\n{traceback.format_exc()}")
        return None, log_file

# ===================== 3. 命令行参数配置（新增核心） =====================
def parse_args():
    """配置命令行参数，支持终端直接运行"""
    parser = argparse.ArgumentParser(description="通用CSV数据清洗脚本 V2.0（支持命令行参数+强化异常处理）")
    
    # 必选参数
    parser.add_argument('-i', '--input', required=True, help="输入CSV文件路径（必填），示例：./原始数据.csv")
    parser.add_argument('-o', '--output', required=True, help="输出清洗后CSV文件路径（必填），示例：./清洗后数据.csv")
    
    # 可选参数（均有默认值）
    parser.add_argument('-l', '--log', default="清洗日志", help="日志文件基础路径，默认：清洗日志")
    parser.add_argument('-dt', '--duplicate_threshold', type=float, default=5.0, help="重复行占比阈值（%），默认：5.0")
    parser.add_argument('-mfs', '--missing_fill_strategy', choices=['auto', 'mean', 'median', 'mode', 'drop'], 
                        default='auto', help="缺失值填充策略，可选：auto/mean/median/mode/drop，默认：auto")
    parser.add_argument('-mct', '--missing_col_threshold', type=float, default=30.0, help="列缺失率阈值（%），默认：30.0")
    parser.add_argument('-om', '--outlier_method', choices=['IQR', '3σ'], default='IQR', help="异常值判定方法，可选：IQR/3σ，默认：IQR")
    parser.add_argument('-ot', '--outlier_threshold', type=float, default=5.0, help="异常值占比阈值（%），默认：5.0")
    
    return parser.parse_args()


# ===================== 5. 主函数（整合命令行+清洗逻辑） =====================
def main():
    """主函数：解析命令行参数 → 执行清洗 → 输出结果"""
    # 解析命令行参数
    args = parse_args()
    
    # 执行清洗
    print("="*60)
    print("开始执行通用CSV数据清洗脚本 V2.0")
    print(f"输入文件：{args.input}")
    print(f"输出文件：{args.output}")
    print("="*60)
    
    cleaned_df, log_file = clean_csv_data(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        duplicate_threshold=args.duplicate_threshold,
        missing_fill_strategy=args.missing_fill_strategy,
        missing_col_threshold=args.missing_col_threshold,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold
    )
    
    # 输出最终结果
    if cleaned_df is not None:
        print(f"\n✅ 清洗成功！")
        print(f"📄 清洗后数据预览：\n{cleaned_df.head()}")
        print(f"📝 日志文件路径：{log_file}")
    else:
        print(f"\n❌ 清洗失败！")
        print(f"📝 错误日志已保存至：{log_file}，请查看日志排查问题")

# ===================== 6. 入口函数（支持命令行+脚本运行） =====================
if __name__ == "__main__":
    # 方式1：命令行运行（优先）
    try:
        main()
    # 方式2：脚本直接运行（测试用，保留原测试逻辑）
    except SystemExit:
        # 若未传入命令行参数，自动生成测试数据并运行
        print("\n⚠️  未传入命令行参数，自动执行测试模式...")
        # 1. 生成测试数据
        test_input = generate_test_student_data()
        # 2. 配置清洗参数（调高重复行阈值，避免终止）
        test_output = "学生成绩_清洗后数据.csv"
        # 3. 执行清洗
        cleaned_df, log_file = clean_csv_data(
            input_path=test_input,
            output_path=test_output,
            duplicate_threshold=10.0,  # 调高阈值，适配测试数据的9.09%重复率
            missing_fill_strategy="auto",
            missing_col_threshold=30.0,
            outlier_method="IQR",
            outlier_threshold=5.0
        )
        # 输出测试结果
        if cleaned_df is not None:
            print(f"\n✅ 测试模式 - 清洗成功！")
            print(f"📄 清洗后数据预览：\n{cleaned_df.head()}")
            print(f"📝 日志文件路径：{log_file}")
        else:
            print(f"\n❌ 测试模式 - 清洗失败！")
            print(f"📝 错误日志已保存至：{log_file}")
    except Exception as e:
        print(f"\n❌ 脚本运行出错：{str(e)}")
        print(f"📝 异常详情：\n{traceback.format_exc()}")