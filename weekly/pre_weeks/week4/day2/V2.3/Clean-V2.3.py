import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import argparse
import traceback
import jieba

# ===================== 1. 停用词加载 =====================
def load_stopwords(file_path="cn_stopwords.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            stopwords = set([line.strip() for line in f.readlines() if line.strip()])
    except FileNotFoundError:
        stopwords = {'的', '了', '吗', '啊', '这', '那', '在', '是', '我', '你', '他', 
                     '很', '真的', '都', '也', '就', '又', '还', '吧', '呢', '哦', '哈',
                     '，', '。', '！', '？', '《', '》', '：', '；', '这部'}
    return stopwords

# ===================== 2. 文本处理函数 =====================
def cn_text_process(text, cut_mode="accurate"):
    if not text or text.strip() == "":
        return []
    stopwords = load_stopwords()
    if cut_mode == "accurate":
        tokens = jieba.lcut(text)
    elif cut_mode == "full":
        tokens = jieba.lcut(text, cut_all=True)
    elif cut_mode == "search":
        tokens = jieba.lcut_for_search(text)
    else:
        tokens = jieba.lcut(text)
    filtered_tokens = [word for word in tokens if word not in stopwords]
    # 可选：将分词结果拼接为字符串（便于保存到CSV）
    return " ".join(filtered_tokens)  # 拼接为空格分隔的字符串

# ===================== 3. 日志配置 =====================
def setup_logger(log_path):
    log_file = f"{log_path}_中文影评清洗日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("cn_text_cleaner")
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

# ===================== 4. 核心清洗函数 =====================
def clean_cn_movie_review(input_path, output_path, log_path="中文影评清洗日志"):
    logger, log_file = setup_logger(log_path)
    logger.info("="*60)
    logger.info("开始执行中文电影评论清洗流程（jieba分词+去停用词）")
    logger.info(f"输入文件：{input_path} | 输出文件：{output_path}")
    logger.info("="*60)

    try:
        # 1. 加载数据
        logger.info("【步骤1：加载数据】")
        df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
        original_count = len(df)
        logger.info(f"原始数据量：{original_count} 条")
        
        # 2. 检查必要列
        if 'comment' not in df.columns:
            raise ValueError("数据缺少 'comment' 列（评论内容）")
        
        # 3. 去重（基于评论内容）
        logger.info("【步骤2：评论去重】")
        df = df.drop_duplicates(subset=['comment'], keep='first')
        after_dup_count = len(df)
        logger.info(f"去重后数据量：{after_dup_count} 条（删除 {original_count - after_dup_count} 条重复评论）")
        
        # 4. 删除空评论
        logger.info("【步骤3：删除空评论】")
        df = df[df['comment'].notna()]
        df = df[df['comment'].str.strip() != '']
        after_empty_count = len(df)
        logger.info(f"删空后数据量：{after_empty_count} 条（删除 {after_dup_count - after_empty_count} 条空评论）")
        
        # 5. 分词 + 去停用词（核心）
        logger.info("【步骤4：jieba分词 + 去停用词】")
        df['cleaned_comment'] = df['comment'].apply(lambda x: cn_text_process(x))
        # 过滤分词后为空的评论
        df = df[df['cleaned_comment'].str.strip() != '']
        final_count = len(df)
        logger.info(f"分词去停用词后数据量：{final_count} 条（删除 {after_empty_count - final_count} 条无效评论）")
        
        # 6. 计算清洗率
        clean_rate = round(((original_count - final_count) / original_count) * 100, 2)
        logger.info("="*60)
        logger.info("✅ 中文影评清洗完成！")
        logger.info(f"原始数据量：{original_count} 条")
        logger.info(f"清洗后数据量：{final_count} 条")
        logger.info(f"清洗率：{clean_rate}%")
        logger.info("="*60)
        
        # 7. 保存结果
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"📁 清洗结果已保存至：{output_path}")
        
        return df, log_file, clean_rate

    except Exception as e:
        logger.error(f"❌ 清洗失败：{str(e)}", exc_info=True)
        logger.error(f"异常详情：\n{traceback.format_exc()}")
        return None, log_file, 0.0

# ===================== 5. 命令行运行 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="中文电影评论清洗脚本（jieba分词+去停用词）")
    parser.add_argument('-i', '--input', required=True, help="输入CSV文件路径（含comment列）")
    parser.add_argument('-o', '--output', required=True, help="输出清洗后CSV文件路径")
    parser.add_argument('-l', '--log', default="中文影评清洗日志", help="日志文件前缀")
    return parser.parse_args()

# ===================== 6. 主函数 =====================
def main():
    args = parse_args()
    df, log_file, clean_rate = clean_cn_movie_review(args.input, args.output, args.log)
    
    if df is not None:
        print(f"\n✅ 清洗成功！")
        print(f"📊 清洗统计：")
        print(f"   - 原始数据量：{pd.read_csv(args.input).shape[0]} 条")
        print(f"   - 清洗后数据量：{len(df)} 条")
        print(f"   - 清洗率：{clean_rate}%")
        print(f"\n📝 清洗效果预览（前2条）：")
        for i in range(min(2, len(df))):
            print(f"\n原始评论{i+1}：\n{df['comment'].iloc[i][:100]}...")
            print(f"清洗后评论{i+1}：\n{df['cleaned_comment'].iloc[i][:100]}...")
        print(f"\n📄 日志文件：{log_file}")
    else:
        print(f"\n❌ 清洗失败！请查看日志：{log_file}")

# ===================== 入口 =====================
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # 测试模式：生成模拟数据并清洗
        print("\n⚠️  未传入命令行参数，执行测试模式...")
        # 生成模拟中文影评数据
        test_data = {
            'comment': [
                "我喜欢看《流浪地球2》，这部科幻电影的特效太棒了吗？真的超震撼！",
                "《满江红》的剧情反转很多，张艺谋的导演手法太绝了！",
                "",  # 空评论
                "我喜欢看《流浪地球2》，这部科幻电影的特效太棒了吗？真的超震撼！",  # 重复评论
                "《无名》的演员演技在线，王一博的表现超出预期！"
            ]
        }
        test_df = pd.DataFrame(test_data)
        test_df.to_csv("test_cn_comments.csv", index=False, encoding='utf-8')
        print("✅ 生成模拟测试数据：test_cn_comments.csv")
        
        # 执行清洗
        df, log_file, clean_rate = clean_cn_movie_review("test_cn_comments.csv", "test_cn_comments_cleaned.csv")
        if df is not None:
            print(f"\n✅ 测试模式清洗成功！")
            print(f"📊 清洗统计：原始5条 → 清洗后{len(df)}条 | 清洗率{clean_rate}%")
            print(f"📝 清洗后评论示例：{df['cleaned_comment'].iloc[0]}")

    except Exception as e:
        print(f"\n❌ 运行出错：{str(e)}")
        print(f"异常详情：\n{traceback.format_exc()}")