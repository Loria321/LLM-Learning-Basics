import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import argparse
import traceback
import re  # 新增：文本处理

# ===================== 1. 日志配置（保留） =====================
def setup_logger(log_path):
    log_file = f"{log_path}_IMDB清洗日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("data_cleaner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger, log_file

# ===================== 2. 核心清洗函数（适配IMDB评论） =====================
def clean_imdb_data(
    input_path,          
    output_path,         
    log_path="IMDB清洗日志",  
    duplicate_threshold=100.0,  # 评论去重不设严格阈值，调大避免终止
    missing_fill_strategy="drop",  # 空评论直接删除
    missing_col_threshold=30.0,    
    outlier_method="IQR",          
    outlier_threshold=5.0          
):
    """
    适配IMDB电影评论的清洗函数：去重+删空评论+统一编码
    """
    logger, log_file = setup_logger(log_path)
    logger.info("="*60)
    logger.info("开始执行IMDB电影评论数据清洗流程")
    logger.info(f"输入文件：{input_path} | 目标：去重+删空评论+统一文本编码")
    logger.info("="*60)

    try:
        # -------------------- 前置校验 --------------------
        logger.info("【前置校验】检查文件格式和存在性")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在：{input_path}")
        if not input_path.lower().endswith('.csv'):
            raise ValueError(f"输入文件格式错误！仅支持CSV文件")

        # -------------------- 步骤1：数据加载 --------------------
        logger.info("【步骤1：数据加载】")
        # 读取IMDB数据（处理编码异常）
        df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
        original_shape = df.shape
        logger.info(f"原始数据维度：{original_shape[0]}行 × {original_shape[1]}列")
        if df.empty:
            raise ValueError("加载的CSV文件为空，无数据可清洗")
        # 检查必要列（review/sentiment）
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("IMDB数据缺少必要列：review（评论）或sentiment（情感标签）")
        
        # -------------------- 步骤2：探索性分析（针对评论） --------------------
        logger.info("\n【步骤2：评论数据探索】")
        # 空评论统计
        empty_comment = df['review'].isna().sum() + (df['review'].str.strip() == '').sum()
        empty_rate = round((empty_comment / len(df) * 100), 2)
        # 重复评论统计
        duplicate_comment = df['review'].duplicated().sum()
        duplicate_rate = round((duplicate_comment / len(df) * 100), 2)
        logger.info(f"空评论数量：{empty_comment} | 占比：{empty_rate}%")
        logger.info(f"重复评论数量：{duplicate_comment} | 占比：{duplicate_rate}%")

        # -------------------- 步骤3：核心清洗（针对评论） --------------------
        logger.info("\n【步骤3：核心清洗】")
        # 3.1 去重（基于评论文本去重）
        df = df.drop_duplicates(subset=['review'], keep='first')
        logger.info(f"去重后数据量：{len(df)}条（删除重复评论{original_shape[0]-len(df)}条）")
        after_dup_shape = len(df)

        # 3.2 删除空评论（空值/空白字符串）
        df = df[df['review'].notna()]  # 删除空值
        df = df[df['review'].str.strip() != '']  # 删除空白字符串
        after_empty_shape = len(df)
        logger.info(f"删除空评论后数据量：{after_empty_shape}条（删除空评论{after_dup_shape - after_empty_shape}条）")

        # 3.3 统一文本编码（清理非UTF-8字符、特殊符号）
        def clean_review_encoding(review):
            """统一文本编码，清理异常字符"""
            # 转为UTF-8，忽略无法编码的字符
            review = review.encode('utf-8', 'ignore').decode('utf-8')
            # 清理多余空格/制表符（可选，提升文本整洁度）
            review = re.sub(r'\s+', ' ', review).strip()
            return review
        
        df['review'] = df['review'].apply(clean_review_encoding)
        logger.info("完成评论文本编码统一：转为UTF-8，清理异常字符")

        # -------------------- 步骤4：数据保存 --------------------
        logger.info("\n【步骤4：数据保存】")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(output_path, index=False, encoding='utf-8')
        final_shape = len(df)
        
        # -------------------- 清洗率计算 --------------------
        clean_rate = round(((original_shape[0] - final_shape) / original_shape[0]) * 100, 2)
        logger.info("="*60)
        logger.info("✅ IMDB评论清洗完成！")
        logger.info(f"原始数据量：{original_shape[0]}条")
        logger.info(f"清洗后数据量：{final_shape}条")
        logger.info(f"删除数据量：{original_shape[0] - final_shape}条")
        logger.info(f"清洗率：{clean_rate}%")
        logger.info(f"输出文件：{output_path}")
        logger.info("="*60)

        return df, log_file, clean_rate

    except Exception as e:
        logger.error(f"清洗过程出错：{str(e)}", exc_info=True)
        logger.error(f"异常详细栈信息：\n{traceback.format_exc()}")
        return None, log_file, 0.0

# ===================== 3. 命令行参数配置（保留） =====================
def parse_args():
    parser = argparse.ArgumentParser(description="IMDB电影评论清洗脚本（V2适配版）")
    parser.add_argument('-i', '--input', required=True, help="输入IMDB CSV文件路径，示例：./imdb_5000.csv")
    parser.add_argument('-o', '--output', required=True, help="输出清洗后CSV文件路径，示例：./imdb_5000_cleaned.csv")
    parser.add_argument('-l', '--log', default="IMDB清洗日志", help="日志文件基础路径，默认：IMDB清洗日志")
    return parser.parse_args()

# ===================== 4. 主函数 =====================
def main():
    args = parse_args()
    print("="*60)
    print("IMDB电影评论清洗脚本（V2适配版）")
    print(f"输入文件：{args.input}")
    print(f"输出文件：{args.output}")
    print("="*60)
    
    cleaned_df, log_file, clean_rate = clean_imdb_data(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log
    )
    
    if cleaned_df is not None:
        print(f"\n✅ 清洗成功！")
        print(f"📊 清洗统计：")
        print(f"   - 原始数据量：{pd.read_csv(args.input).shape[0]}条")
        print(f"   - 清洗后数据量：{len(cleaned_df)}条")
        print(f"   - 清洗率：{clean_rate}%")
        print(f"📝 日志文件：{log_file}")
    else:
        print(f"\n❌ 清洗失败！")
        print(f"📝 错误日志：{log_file}")

# ===================== 入口函数 =====================
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # 测试模式：自动处理5000条IMDB数据
        print("\n⚠️  未传入命令行参数，执行测试模式...")
        # 1. 自动下载/读取5000条IMDB数据
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/IMDB-Movie-Reviews-Dataset/master/IMDB%20Dataset.csv")
            df = df.head(5000)
            df.to_csv("imdb_5000.csv", index=False, encoding='utf-8')
            print(f"✅ 自动下载IMDB数据：imdb_5000.csv（{len(df)}条）")
        except Exception as e:
            print(f"❌ 自动下载失败，请手动下载数据集！错误：{e}")
            exit(1)
        # 2. 执行清洗
        cleaned_df, log_file, clean_rate = clean_imdb_data(
            input_path="imdb_5000.csv",
            output_path="imdb_5000_cleaned.csv",
            log_path="IMDB清洗日志"
        )
        # 3. 输出结果
        if cleaned_df is not None:
            print(f"\n✅ 测试模式 - 清洗成功！")
            print(f"📊 清洗统计：")
            print(f"   - 原始数据量：5000条")
            print(f"   - 清洗后数据量：{len(cleaned_df)}条")
            print(f"   - 清洗率：{clean_rate}%")
            print(f"📝 日志文件：{log_file}")
        else:
            print(f"\n❌ 测试模式 - 清洗失败！")
    except Exception as e:
        print(f"\n❌ 脚本运行出错：{str(e)}")
        print(f"📝 异常详情：\n{traceback.format_exc()}")