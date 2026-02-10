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

# ===================== 初始化：加载自定义词典（新增校园术语） =====================
custom_dict = """
# 核心校园实体
学分 10 n
选课 10 n
GPA 10 n
辅导员 10 n
教务处 10 n
毕业论文 10 n
开题报告 10 n
答辩 10 n
补考 10 n
重修 10 n
综测 10 n
保研 10 n
考研 10 n
奖学金 10 n
助学金 10 n
选课系统 10 n
教务系统 10 n
学分绩点 10 n
通识课 10 n
专业课 10 n
选修课 10 n
必修课 10 n

# 校园场景短语
期末考核 10 n
开学时间 10 n
放假安排 10 n
宿舍申请 10 n
社团招新 10 n
学术讲座 10 n
交换项目 10 n
四六级 10 n
计算机二级 10 n
体测 10 n
"""
# 将自定义词典写入临时文件并加载
with open("custom_dict.txt", "w", encoding="utf-8") as f:
    f.write(custom_dict.strip())
jieba.load_userdict("custom_dict.txt")

# ===================== 1. 正则去噪（新增校园问答专属规则） =====================
def clean_text_noise(text):
    if pd.isna(text) or text is None or text.strip() == "":
        return ""
    
    # 原有基础去噪
    text = re.sub(r'<[^>]+>', '', text)  # 删HTML标签
    emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', flags=re.UNICODE)
    text = emoji_pattern.sub('', text)  # 删Emoji
    special_noise = r'★|■|◆|●|△|▲|※|§|№|＃|＆|＄|％|＠|～|｀|＾|｜|＼|／'
    text = re.sub(special_noise, '', text)  # 删极端特殊符号
    text = re.sub(r'\s+', ' ', text).strip()  # 合并多余空格
    text = re.sub(r'(\d+) +([年月日])', r'\1\2', text)  # 数字+中文连写
    
    # 校园问答专属去噪
    text = re.sub(r'Q:|A:|提问：|回答：|【问题】|【回复】', '', text)  # 去除问答标记残留
    text = re.sub(r'[1-9]\d{4,10}', '', text)  # 去除学号/QQ号
    text = re.sub(r'1\d{10}', '', text)  # 去除手机号
    text = re.sub(r'\w+@(stu\.)?\w+\.edu\.cn', '', text)  # 去除校园邮箱
    text = re.sub(r'https?://(jw.|xy.|www.)?\w+\.edu\.cn', '', text)  # 去除校园网址
    text = re.sub(r'[一二三四五六七八九]?[、.）)]', '', text)  # 去除列表序号残留（如"1." "一、"）
    
    return text

# ===================== 2. 停用词加载（新增校园场景停用词） =====================
def load_stopwords(file_path=STOPWORDS_FILE):
    """扩充校园场景停用词，覆盖问答中的冗余表达"""
    default_stopwords = {
        # 原有基础停用词
        '的', '了', '吗', '啊', '这', '那', '在', '是', '我', '你', '他', 
        '很', '真的', '都', '也', '就', '又', '还', '吧', '呢', '哦', '哈',
        '从', '于', '和', '与', '或', '及', '对', '对于', '关于', '把', '被', '为', '因', '由',
        '个', '等', '所', '之', '其', '超', '已', '将', '才', '仅', '只', '全', '均', '共',
        '，', '。', '！', '？', '、', '：', '；', '（', '）', ' ', '"', "'",
        
        # 校园问答专属停用词
        '同学', '老师', '请问', '您好', '谢谢', '麻烦', '请问一下', '你好', '谢谢啦',
        '问题', '提问', '回答', '回复', '想问', '想知道', '告知', '咨询', '了解',
        '学校', '学院', '系里', '这里', '那里', '这边', '那边', '哪个', '哪些',
        '什么', '怎么', '为什么', '多少', '何时', '何地', '怎样', '如何',
        '可以', '能', '会', '有没有', '是不是', '有没有人', '能不能', '会不会'
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            stopwords = set([line.strip() for line in f.readlines() if line.strip()])
        stopwords = stopwords.union(default_stopwords)
        logging.info(f"✅ 加载停用词 {len(stopwords)} 个（含校园专属）")
    except FileNotFoundError:
        logging.warning(f"⚠️  使用默认停用词表（含校园专属）")
        stopwords = default_stopwords
    return stopwords

# ===================== 3. 分词+去停用词（保持逻辑，适配新词典） =====================
def cn_text_cut(text, enable_cut=True, enable_stopwords=True):
    if not text or text.strip() == "" or len(text.strip()) < 2:
        return ""
    
    if not enable_cut:
        return text.strip()
    
    # 精准分词（已加载校园专属词典）
    tokens = jieba.lcut(text.strip())
    
    # 去停用词（含校园专属）
    if enable_stopwords:
        stopwords = load_stopwords()
        tokens = [word for word in tokens if word not in stopwords and word.strip() != ""]
    
    return " ".join(tokens)

# ===================== 4. 日志配置（保持不变） =====================
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

# ===================== 5. 核心清洗函数（保持逻辑，适配场景） =====================
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
    logger.info("开始执行校园问答文本清洗流程")
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
        
        # 基础清洗（强化去重逻辑，适应重复提问场景）
        logger.info("【步骤2：基础清洗（去重+删空）】")
        # 先去除完全重复
        df = df.drop_duplicates(subset=[text_column], keep='first')
        # 再去除空白内容
        df = df[df[text_column].notna()]
        df = df[df[text_column].str.strip() != '']
        # 针对校园问答：去除过短文本（小于5字的可能是无效提问）
        df = df[df[text_column].str.len() >= 5]
        basic_clean_count = len(df)
        logger.info(f"基础清洗后：{basic_clean_count} 条（删除 {original_count - basic_clean_count} 条）")
        
        # 正则去噪（使用校园专属规则）
        if enable_noise:
            logger.info("【步骤3：正则去噪（校园专属）】")
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
        
        # 分词+去停用词（使用校园词典和停用词）
        if enable_cut:
            logger.info(f"【步骤4：jieba分词（校园专属词典+停用词）】")
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

# ===================== 6. 命令行参数（保持不变） =====================
def parse_args():
    parser = argparse.ArgumentParser(description="校园问答文本清洗脚本")
    parser.add_argument('-i', '--input', required=True, help="输入CSV路径")
    parser.add_argument('-o', '--output', required=True, help="输出CSV路径")
    parser.add_argument('-l', '--log', default="校园问答清洗日志", help="日志前缀")
    parser.add_argument('-c', '--column', default="content", help="文本列名")
    parser.add_argument('--disable-noise', action='store_false', dest='enable_noise')
    parser.add_argument('--disable-cut', action='store_false', dest='enable_cut')
    parser.add_argument('--disable-stopwords', action='store_false', dest='enable_stopwords')
    return parser.parse_args()

# ===================== 7. 主函数（保持不变） =====================
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

# ===================== 8. 测试模块（改为校园问答测试） =====================
def test_campus_qa_clean():
    print("="*80)
    print("📌 执行校园问答清洗测试")
    print("="*80)
    
    # 生成校园问答测试数据
    test_qa_data = {
        'id': [1, 2, 3, 4, 5],
        'content': [
            """Q: 老师您好！请问2025年的选课时间是什么时候呀？我是计算机学院的同学，学号是202201001，谢谢！<br/>""",
            """【问题】ABC大学的GPA怎么计算呢？有没有包含选修课成绩？麻烦告知一下，邮箱是stu123@xxx.edu.cn<br/>""",
            """请问重修的课程能算入综测吗？之前问过辅导员但没记清楚... https://jw.abc.edu.cn/faq""",
            """短文本""",  # 过短内容（会被过滤）
            """重复问题 请问重修的课程能算入综测吗？之前问过辅导员但没记清楚"""  # 重复内容（会被去重）
        ]
    }
    test_input = "test_campus_qa.csv"
    test_output = "test_campus_qa_cleaned.csv"
    pd.DataFrame(test_qa_data).to_csv(test_input, index=False, encoding='utf-8')
    print(f"✅ 生成测试数据：{test_input}")
    
    # 执行清洗
    df, log_file, clean_rate = clean_text_data(
        input_path=test_input,
        output_path=test_output,
        log_path="校园问答清洗测试",
        enable_noise=True,
        enable_cut=True,
        enable_stopwords=True,
        text_column="content"
    )
    
    if df is not None:
        print(f"\n✅ 测试完成！")
        print(f"📊 统计：原始5条 → 最终{len(df)}条 | 清洗率{clean_rate}%")
        print(f"\n📝 第一条问答清洗效果：")
        print(f"原始：\n{df['content'].iloc[0][:200]}...")
        print(f"清洗后：\n{df['final_cleaned'].iloc[0]}")
        print(f"\n📝 第二条问答清洗效果：")
        print(f"原始：\n{df['content'].iloc[1][:200]}...")
        print(f"清洗后：\n{df['final_cleaned'].iloc[1]}")
        print(f"\n📁 结果文件：{test_output}")

# ===================== 入口 =====================
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        test_campus_qa_clean()
    except Exception as e:
        print(f"\n❌ 出错：{str(e)}")
        print(f"详情：\n{traceback.format_exc()}")
    # 清理临时自定义词典文件
    if os.path.exists("custom_dict.txt"):
        os.remove("custom_dict.txt")