import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import argparse
import traceback
import re
import unicodedata

# ===================== 1. 日志配置 =====================
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

# ===================== 2. 核心文本去噪函数 =====================
def clean_review_noise(review, lower_case=False):
    """通用文本去噪函数：去除HTML标签、emoji、特殊符号、多余空格"""
    if pd.isna(review) or review is None:
        return ""
    # 统一编码
    review = unicodedata.normalize('NFKD', review).encode('utf-8', 'ignore').decode('utf-8')
    # 去除HTML标签
    review = re.sub(r'<.*?>', '', review)
    # 去除emoji
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002500-\U00002BEF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    review = emoji_pattern.sub(r'', review)
    # 去除特殊符号（保留基础标点）
    review = re.sub(r'[^\w\s.,!?\']', ' ', review)
    # 合并多余空格
    review = re.sub(r'\s+', ' ', review).strip()
    # 可选小写
    if lower_case:
        review = review.lower()
    return review

# ===================== 3. 核心清洗函数（集成去噪） =====================
def clean_imdb_data(
    input_path,          
    output_path,         
    log_path="IMDB清洗日志",  
    duplicate_threshold=100.0,
    missing_fill_strategy="drop",
    missing_col_threshold=30.0,    
    outlier_method="IQR",          
    outlier_threshold=5.0          
):
    logger, log_file = setup_logger(log_path)
    logger.info("="*60)
    logger.info("开始执行IMDB电影评论数据清洗流程（含文本去噪）")
    logger.info(f"输入文件：{input_path} | 核心：去重+删空+文本去噪+编码统一")
    logger.info("="*60)

    try:
        # 前置校验
        logger.info("【前置校验】检查文件格式和存在性")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在：{input_path}")
        if not input_path.lower().endswith('.csv'):
            raise ValueError(f"输入文件格式错误！仅支持CSV文件")

        # 数据加载
        logger.info("【步骤1：数据加载】")
        df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
        original_shape = df.shape
        logger.info(f"原始数据维度：{original_shape[0]}行 × {original_shape[1]}列")
        if df.empty:
            raise ValueError("加载的CSV文件为空，无数据可清洗")
        if 'review' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("IMDB数据缺少必要列：review（评论）或sentiment（情感标签）")
        
        # 探索性分析
        logger.info("\n【步骤2：评论数据探索】")
        empty_comment = df['review'].isna().sum() + (df['review'].str.strip() == '').sum()
        empty_rate = round((empty_comment / len(df) * 100), 2)
        duplicate_comment = df['review'].duplicated().sum()
        duplicate_rate = round((duplicate_comment / len(df) * 100), 2)
        logger.info(f"空评论数量：{empty_comment} | 占比：{empty_rate}%")
        logger.info(f"重复评论数量：{duplicate_comment} | 占比：{duplicate_rate}%")

        # 核心清洗
        logger.info("\n【步骤3：核心清洗】")
        # 3.1 去重（基于评论文本）
        df = df.drop_duplicates(subset=['review'], keep='first')
        logger.info(f"去重后数据量：{len(df)}条（删除重复评论{original_shape[0]-len(df)}条）")
        after_dup_shape = len(df)

        # 3.2 删除空评论
        df = df[df['review'].notna()]
        df = df[df['review'].str.strip() != '']
        after_empty_shape = len(df)
        logger.info(f"删除空评论后数据量：{after_empty_shape}条（删除空评论{after_dup_shape - after_empty_shape}条）")

        # 3.3 文本去噪（核心新增）
        logger.info("开始文本去噪：去除HTML标签、emoji、特殊符号、多余空格")
        # 对review列应用去噪函数，可选转为小写（lower_case=True）
        df['review'] = df['review'].apply(lambda x: clean_review_noise(x, lower_case=True))
        # 去噪后可能产生新的空字符串，再次过滤
        df = df[df['review'].str.strip() != '']
        after_noise_shape = len(df)
        logger.info(f"文本去噪后数据量：{after_noise_shape}条（去噪后删除空评论{after_empty_shape - after_noise_shape}条）")

        # 数据保存
        logger.info("\n【步骤4：数据保存】")
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df.to_csv(output_path, index=False, encoding='utf-8')
        final_shape = len(df)
        
        # 清洗率计算
        clean_rate = round(((original_shape[0] - final_shape) / original_shape[0]) * 100, 2)
        logger.info("="*60)
        logger.info("✅ IMDB评论清洗+去噪完成！")
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

# ===================== 4. 命令行参数配置 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="IMDB电影评论清洗脚本（含文本去噪）")
    parser.add_argument('-i', '--input', required=True, help="输入IMDB CSV文件路径")
    parser.add_argument('-o', '--output', required=True, help="输出清洗后CSV文件路径")
    parser.add_argument('-l', '--log', default="IMDB清洗日志", help="日志文件基础路径")
    return parser.parse_args()

# ===================== 5. 主函数 =====================
def main():
    args = parse_args()
    print("="*60)
    print("IMDB电影评论清洗脚本（含文本去噪）")
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
        # 预览去噪效果
        print(f"\n📝 去噪效果预览（前2条评论）：")
        original_df = pd.read_csv(args.input).head(2)['review']
        cleaned_review = cleaned_df.head(2)['review']
        for i in range(2):
            print(f"\n原始评论{i+1}：\n{original_df.iloc[i][:100]}...")
            print(f"去噪后评论{i+1}：\n{cleaned_review.iloc[i][:100]}...")
        print(f"\n📝 日志文件：{log_file}")
    else:
        print(f"\n❌ 清洗失败！")
        print(f"📝 错误日志：{log_file}")

# ===================== 入口函数 =====================
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # 测试模式
        print("\n⚠️  未传入命令行参数，执行测试模式...")
        try:
            df = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/IMDB-Movie-Reviews-Dataset/master/IMDB%20Dataset.csv")
            df = df.head(5000)
            df.to_csv("imdb_5000.csv", index=False, encoding='utf-8')
            print(f"✅ 自动下载IMDB数据：imdb_5000.csv（{len(df)}条）")
        except Exception as e:
            print(f"❌ 自动下载失败，请手动下载数据集！错误：{e}")
            exit(1)
        # 执行清洗
        cleaned_df, log_file, clean_rate = clean_imdb_data(
            input_path="imdb_5000.csv",
            output_path="imdb_5000_cleaned.csv",
            log_path="IMDB清洗日志"
        )
        # 输出结果
        if cleaned_df is not None:
            print(f"\n✅ 测试模式 - 清洗成功！")
            print(f"📊 清洗统计：")
            print(f"   - 原始数据量：5000条")
            print(f"   - 清洗后数据量：{len(cleaned_df)}条")
            print(f"   - 清洗率：{clean_rate}%")
            print(f"\n📝 去噪效果预览（第1条评论）：")
            print(f"原始：{pd.read_csv('imdb_5000.csv').head(1)['review'].iloc[0][:100]}...")
            print(f"去噪后：{cleaned_df.head(1)['review'].iloc[0][:100]}...")
        else:
            print(f"\n❌ 测试模式 - 清洗失败！")