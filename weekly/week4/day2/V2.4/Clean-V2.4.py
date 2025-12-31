import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import argparse
import traceback
import re
import jieba

# ===================== 全局配置 =====================
STOPWORDS_FILE = "cn_stopwords.txt"
DEFAULT_ENABLE_NOISE = True
DEFAULT_ENABLE_CUT = True
DEFAULT_ENABLE_STOPWORDS = True

# ===================== 初始化：加载自定义词典提升分词精准度 =====================
# 自定义词典（解决“上一代”“ABC公司”等分词错误）
custom_dict = """
ABC公司 10 n
上一代 10 n
2025年 10 n
3月5日 10 n
覆盖全国 10 n
人工智能行业 10 n
"""
# 将自定义词典写入临时文件并加载
with open("custom_dict.txt", "w", encoding="utf-8") as f:
    f.write(custom_dict.strip())
jieba.load_userdict("custom_dict.txt")

# ===================== 1. 正则去噪（保留核心逻辑，微调数字处理） =====================
def clean_text_noise(text):
    if pd.isna(text) or text is None or text.strip() == "":
        return ""
    
    # 删所有HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 删Emoji
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    # 删极端特殊符号
    special_noise = r'★|■|◆|●|△|▲|※|§|№|＃|＆|＄|％|＠|～|｀|＾|｜|＼|／'
    text = re.sub(special_noise, '', text)
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 微调：数字+中文之间不加空格（如“2025 年”→“2025年”）
    text = re.sub(r'(\d+) +([年月日])', r'\1\2', text)
    
    return text

# ===================== 2. 停用词加载（大幅扩充，解决过滤不彻底） =====================
def load_stopwords(file_path=STOPWORDS_FILE):
    """扩充停用词表，覆盖残留的“本报记者”“该”“从”等"""
    default_stopwords = {
        # 基础停用词
        '的', '了', '吗', '啊', '这', '那', '在', '是', '我', '你', '他', 
        '很', '真的', '都', '也', '就', '又', '还', '吧', '呢', '哦', '哈',
        # 介词/连词（新增）
        '从', '于', '和', '与', '或', '及', '对', '对于', '关于', '把', '被', '为', '因', '由',
        # 量词/助词（新增）
        '个', '等', '所', '之', '其',
        # 副词（新增）
        '超', '已', '将', '才', '仅', '只', '都', '全', '均', '共',
        # 标点/空格
        '，', '。', '！', '？', '、', '：', '；', '（', '）', ' ', '"', "'",
        # 新闻专属停用词（扩充）
        '本报', '记者', '本报记者', '报道', '据悉', '近日', '目前', '相关', '部门', '表示',
        '认为', '指出', '强调', '发布', '公告', '通知', '称', '据了解', '据介绍', '该',
        # HTML标签碎片
        'p', 'br', 'a', 'href', 'com', 'https', 'http', 'example'
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            stopwords = set([line.strip() for line in f.readlines() if line.strip()])
        stopwords = stopwords.union(default_stopwords)
        logging.info(f"✅ 加载停用词 {len(stopwords)} 个（扩充版）")
    except FileNotFoundError:
        logging.warning(f"⚠️  使用默认停用词表（扩充版）")
        stopwords = default_stopwords
    return stopwords

# ===================== 3. 分词+去停用词（优化后） =====================
def cn_text_cut(text, enable_cut=True, enable_stopwords=True):
    if not text or text.strip() == "" or len(text.strip()) < 2:
        return ""
    
    if not enable_cut:
        return text.strip()
    
    # 精准分词（加载自定义词典后）
    tokens = jieba.lcut(text.strip())
    
    # 去停用词（扩充版）
    if enable_stopwords:
        stopwords = load_stopwords()
        tokens = [word for word in tokens if word not in stopwords and word.strip() != ""]
    
    return " ".join(tokens)

# ===================== 4. 日志配置 =====================
def setup_logger(log_path):
    log_file = f"{log_path}_文本清洗日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("text_cleaner")
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

# ===================== 5. 核心清洗函数 =====================
def clean_text_data(
    input_path, 
    output_path, 
    log_path="文本清洗日志",
    enable_noise=DEFAULT_ENABLE_NOISE,
    enable_cut=DEFAULT_ENABLE_CUT,
    enable_stopwords=DEFAULT_ENABLE_STOPWORDS,
    text_column="content"
):
    logger, log_file = setup_logger(log_path)
    logger.info("="*80)
    logger.info("开始执行通用文本清洗流程（最终优化版）")
    logger.info(f"配置：正则去噪={enable_noise} | 分词={enable_cut} | 去停用词={enable_stopwords}")
    logger.info(f"输入文件：{input_path} | 输出文件：{output_path} | 文本列：{text_column}")
    logger.info("="*80)

    try:
        # 加载数据
        logger.info("【步骤1：加载原始数据】")
        df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
        original_count = len(df)
        logger.info(f"原始数据量：{original_count} 条")
        
        if text_column not in df.columns:
            raise ValueError(f"缺少文本列：{text_column}")
        
        # 基础清洗
        logger.info("【步骤2：基础清洗（去重+删空）】")
        df = df.drop_duplicates(subset=[text_column], keep='first')
        df = df[df[text_column].notna()]
        df = df[df[text_column].str.strip() != '']
        basic_clean_count = len(df)
        logger.info(f"基础清洗后：{basic_clean_count} 条（删除 {original_count - basic_clean_count} 条）")
        
        # 正则去噪
        if enable_noise:
            logger.info("【步骤3：正则去噪（最终版）】")
            df['cleaned_noise'] = df[text_column].apply(clean_text_noise)
            logger.info(f"第一条去噪后文本：{df['cleaned_noise'].iloc[0][:100]}")
            df = df[df['cleaned_noise'].str.strip() != '']
            noise_clean_count = len(df)
            logger.info(f"正则去噪后：{noise_clean_count} 条（删除 {basic_clean_count - noise_clean_count} 条）")
            temp_text_col = 'cleaned_noise'
        else:
            logger.info("【步骤3：跳过正则去噪】")
            df['cleaned_noise'] = df[text_column]
            temp_text_col = text_column
            noise_clean_count = basic_clean_count
        
        # 分词+去停用词
        if enable_cut:
            logger.info(f"【步骤4：jieba分词（自定义词典+扩充停用词）】")
            df['final_cleaned'] = df[temp_text_col].apply(
                lambda x: cn_text_cut(x, enable_cut=True, enable_stopwords=enable_stopwords)
            )
        else:
            logger.info("【步骤4：跳过分词】")
            df['final_cleaned'] = df[temp_text_col]
        
        # 最终过滤
        df = df[df['final_cleaned'].str.strip() != '']
        final_count = len(df)
        logger.info(f"最终清洗后：{final_count} 条")
        
        # 统计+保存
        clean_rate = round(((original_count - final_count) / original_count) * 100, 2)
        logger.info("="*80)
        logger.info(f"✅ 清洗完成！原始 {original_count} → 最终 {final_count} | 清洗率 {clean_rate}%")
        logger.info("="*80)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"📁 结果保存至：{output_path}")
        
        return df, log_file, clean_rate

    except Exception as e:
        logger.error(f"❌ 清洗失败：{str(e)}", exc_info=True)
        return None, log_file, 0.0

# ===================== 6. 命令行参数 =====================
def parse_args():
    parser = argparse.ArgumentParser(description="通用文本清洗脚本（最终优化版）")
    parser.add_argument('-i', '--input', required=True, help="输入CSV路径")
    parser.add_argument('-o', '--output', required=True, help="输出CSV路径")
    parser.add_argument('-l', '--log', default="文本清洗日志", help="日志前缀")
    parser.add_argument('-c', '--column', default="content", help="文本列名")
    parser.add_argument('--disable-noise', action='store_false', dest='enable_noise')
    parser.add_argument('--disable-cut', action='store_false', dest='enable_cut')
    parser.add_argument('--disable-stopwords', action='store_false', dest='enable_stopwords')
    return parser.parse_args()

# ===================== 7. 主函数 =====================
def main():
    args = parse_args()
    df, log_file, clean_rate = clean_text_data(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        enable_noise=args.enable_noise,
        enable_cut=args.enable_cut,
        enable_stopwords=args.enable_stopwords,
        text_column=args.column
    )
    
    if df is not None:
        print(f"\n✅ 清洗成功！")
        print(f"📊 统计：原始 {pd.read_csv(args.input).shape[0]} → 最终 {len(df)} | 清洗率 {clean_rate}%")
        print(f"\n📝 清洗效果预览：")
        for i in range(min(2, len(df))):
            print(f"\n【原始文本{i+1}】：\n{df[args.column].iloc[i][:150]}...")
            print(f"【清洗后{i+1}】：\n{df['final_cleaned'].iloc[i][:150]}...")
        print(f"\n📄 日志：{log_file}")
    else:
        print(f"\n❌ 清洗失败！查看日志：{log_file}")

# ===================== 8. 测试模块 =====================
def test_cn_news_clean():
    print("="*80)
    print("📌 执行中文新闻清洗测试（最终优化版）")
    print("="*80)
    
    # 生成测试数据
    test_news_data = {
        'title': ["【央视新闻】2025年全国两会开幕😀", "科技巨头发布新款AI芯片★", "", "重复新闻", "楼市新政"],
        'content': [
            """<p>本报记者从国务院新闻办获悉<br/>，2025年全国两会于3月5日在北京召开，相关部门表示，今年将重点关注民生、就业等领域。</p> 据悉，此次会议参会代表超2000人，覆盖全国31个省市区。""",
            """<a href="https://tech.example.com">科技巨头ABC公司近日发布了新款AI芯片，该芯片的算力相比上一代提升50%，相关专家认为，这将推动人工智能行业的发展。</a> 目前，该芯片已开始量产。""",
            "",
            """重复文本""",
            """近日，上海、广州、深圳等多地发布了新的楼市调控政策，相关部门指出，新政将支持刚性和改善性购房需求，同时保持房地产市场的稳定。"""
        ]
    }
    test_input = "test_cn_news.csv"
    test_output = "test_cn_news_cleaned.csv"
    pd.DataFrame(test_news_data).to_csv(test_input, index=False, encoding='utf-8')
    print(f"✅ 生成测试数据：{test_input}")
    
    # 执行清洗
    df, log_file, clean_rate = clean_text_data(
        input_path=test_input,
        output_path=test_output,
        log_path="中文新闻清洗测试",
        enable_noise=True,
        enable_cut=True,
        enable_stopwords=True,
        text_column="content"
    )
    
    if df is not None:
        print(f"\n✅ 测试完成！")
        print(f"📊 统计：原始5条 → 最终{len(df)}条 | 清洗率{clean_rate}%")
        print(f"\n📝 第一条新闻清洗效果：")
        print(f"原始：\n{df['content'].iloc[0][:200]}...")
        print(f"清洗后：\n{df['final_cleaned'].iloc[0]}")
        print(f"\n📝 第二条新闻清洗效果：")
        print(f"原始：\n{df['content'].iloc[1][:200]}...")
        print(f"清洗后：\n{df['final_cleaned'].iloc[1]}")
        print(f"\n📁 结果文件：{test_output}")

# ===================== 入口 =====================
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        test_cn_news_clean()
    except Exception as e:
        print(f"\n❌ 出错：{str(e)}")
        print(f"详情：\n{traceback.format_exc()}")
    # 清理临时自定义词典文件
    if os.path.exists("custom_dict.txt"):
        os.remove("custom_dict.txt")