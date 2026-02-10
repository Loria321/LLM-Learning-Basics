import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
import re
from collections import Counter

# 设置中文字体（避免画图乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 基础配置（复用清洗脚本的停用词） =====================
def load_stopwords():
    """加载停用词表（含校园专属）"""
    stopwords = {
        # 基础停用词
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
    return stopwords

STOPWORDS = load_stopwords()

# ===================== 2. 核心评估函数 =====================
def text_quality_evaluate(text_series, is_cleaned=False):
    """
    文本质量评估函数
    :param text_series: pd.Series，待评估的文本列（清洗前/后）
    :param is_cleaned: bool，是否为清洗后文本（清洗后已分词，用空格分隔）
    :return: dict，评估结果
    """
    # 初始化评估结果
    eval_result = {
        "样本总数": len(text_series),
        "平均字符长度": 0.0,
        "平均词汇数": 0.0,
        "有效词汇占比(%)": 0.0,
        "长度分布": {}
    }
    
    # 过滤空文本
    text_series = text_series[text_series.notna() & (text_series.str.strip() != "")]
    valid_count = len(text_series)
    if valid_count == 0:
        return eval_result
    
    # -------------------- 指标1：文本长度分布 + 平均字符长度 --------------------
    # 计算每个文本的字符长度
    char_lengths = text_series.apply(lambda x: len(x.strip()))
    eval_result["平均字符长度"] = round(char_lengths.mean(), 2)
    
    # 统计长度分布（区间：0-10, 10-20, 20-50, 50+）
    length_bins = [0, 10, 20, 50, float('inf')]
    length_labels = ["0-10字", "10-20字", "20-50字", "50字以上"]
    length_cut = pd.cut(char_lengths, bins=length_bins, labels=length_labels, right=False)
    length_dist = length_cut.value_counts().sort_index()
    eval_result["长度分布"] = {label: int(length_dist.get(label, 0)) for label in length_labels}
    
    # -------------------- 指标2：平均词汇数 + 有效词汇占比 --------------------
    total_words = 0  # 总词汇数
    valid_words = 0  # 有效词汇数
    
    for text in text_series:
        if text.strip() == "":
            continue
        
        # 分词（清洗后已分词，直接按空格切分；清洗前需先分词）
        if is_cleaned:
            words = text.strip().split()
        else:
            # 清洗前先做基础去噪，再分词
            clean_text = re.sub(r'<[^>]+>|[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', '', text)
            words = jieba.lcut(clean_text.strip())
        
        # 统计总词汇数和有效词汇数
        total_words += len(words)
        # 有效词汇：非停用词、非空、非纯数字/符号
        for word in words:
            if (word not in STOPWORDS) and (word.strip() != "") and (not re.match(r'^\d+(\.\d+)?$', word)):
                valid_words += 1
    
    # 计算平均词汇数和有效词汇占比
    eval_result["平均词汇数"] = round(total_words / valid_count, 2) if valid_count > 0 else 0.0
    eval_result["有效词汇占比(%)"] = round((valid_words / total_words) * 100, 2) if total_words > 0 else 0.0
    
    return eval_result

# ===================== 3. 可视化对比函数（修复形状不匹配问题） =====================
def plot_quality_comparison(before_eval, after_eval):
    """可视化清洗前后的指标对比（修复轴长度不匹配）"""
    # 子图1：长度分布对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # -------- 子图1：长度分布（4个区间） --------
    length_labels = list(before_eval["长度分布"].keys())
    before_length = [before_eval["长度分布"][label] for label in length_labels]
    after_length = [after_eval["长度分布"][label] for label in length_labels]
    
    # 为长度分布生成4个x轴位置
    x1 = np.arange(len(length_labels))
    width = 0.35
    ax1.bar(x1 - width/2, before_length, width, label='清洗前', color='#ff7f7f')
    ax1.bar(x1 + width/2, after_length, width, label='清洗后', color='#7fbf7f')
    ax1.set_xlabel('文本长度区间')
    ax1.set_ylabel('文本数量')
    ax1.set_title('清洗前后文本长度分布对比')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(length_labels)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # -------- 子图2：核心指标对比（3个指标） --------
    metrics = ['平均字符长度', '平均词汇数', '有效词汇占比(%)']
    before_metrics = [before_eval[metric] for metric in metrics]
    after_metrics = [after_eval[metric] for metric in metrics]
    
    # 为核心指标生成3个x轴位置（关键修复：匹配指标数量）
    x2 = np.arange(len(metrics))
    ax2.bar(x2 - width/2, before_metrics, width, label='清洗前', color='#ff7f7f')
    ax2.bar(x2 + width/2, after_metrics, width, label='清洗后', color='#7fbf7f')
    ax2.set_xlabel('评估指标')
    ax2.set_ylabel('指标值')
    ax2.set_title('清洗前后核心指标对比')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (b, a) in enumerate(zip(before_metrics, after_metrics)):
        ax2.text(i - width/2, b + 0.5, f'{b}', ha='center', va='bottom', fontsize=10)
        ax2.text(i + width/2, a + 0.5, f'{a}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('文本质量评估对比图.png', dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 4. 主执行逻辑 =====================
def main():
    # 1. 加载数据（替换为你的文件路径）
    before_file = "weekly\week4\day3\HDUtieba_raw_simulation.csv"  # 清洗前数据
    after_file = "weekly\week4\day3\HDUtieba_cleaned_simulation.csv"  # 清洗后数据
    
    try:
        df_before = pd.read_csv(before_file, encoding='utf-8')
        df_after = pd.read_csv(after_file, encoding='utf-8')
        print("✅ 数据加载成功！")
    except FileNotFoundError as e:
        print(f"❌ 文件未找到：{e}")
        return
    
    # 2. 提取文本列（根据你的实际列名调整！）
    text_col_before = "content"  # 清洗前的文本列
    text_col_after = "final_cleaned"  # 清洗后的文本列（确认是你的实际列名）
    
    # 检查列名是否存在
    if text_col_after not in df_after.columns:
        print(f"⚠️  清洗后文件无 {text_col_after} 列，自动检测文本列...")
        # 自动匹配可能的清洗后列名
        possible_cols = ['final_cleaned', 'cleaned_content', 'content_cleaned', 'content']
        for col in possible_cols:
            if col in df_after.columns:
                text_col_after = col
                print(f"✅ 自动匹配到清洗后文本列：{col}")
                break
    
    # 3. 计算清洗前后的评估指标
    print("\n📊 开始计算清洗前的文本质量指标...")
    eval_before = text_quality_evaluate(df_before[text_col_before], is_cleaned=False)
    print("\n📊 开始计算清洗后的文本质量指标...")
    eval_after = text_quality_evaluate(df_after[text_col_after], is_cleaned=True)
    
    # 4. 打印评估结果
    print("\n" + "="*80)
    print("📈 清洗前文本质量评估结果")
    print("="*80)
    for k, v in eval_before.items():
        if k == "长度分布":
            print(f"{k}：{v}")
        else:
            print(f"{k}：{v}")
    
    print("\n" + "="*80)
    print("📈 清洗后文本质量评估结果")
    print("="*80)
    for k, v in eval_after.items():
        if k == "长度分布":
            print(f"{k}：{v}")
        else:
            print(f"{k}：{v}")
    
    # 5. 可视化对比（修复后）
    print("\n📊 生成可视化对比图...")
    plot_quality_comparison(eval_before, eval_after)
    
    # 6. 总结指标变化
    print("\n" + "="*80)
    print("🔍 指标变化总结")
    print("="*80)
    print(f"样本总数：{eval_before['样本总数']} → {eval_after['样本总数']}（删除无效数据 {eval_before['样本总数'] - eval_after['样本总数']} 条）")
    print(f"平均字符长度：{eval_before['平均字符长度']} → {eval_after['平均字符长度']}（去除冗余后更聚焦核心）")
    print(f"平均词汇数：{eval_before['平均词汇数']} → {eval_after['平均词汇数']}（过滤停用词后词汇数更精简）")
    print(f"有效词汇占比：{eval_before['有效词汇占比(%)']}% → {eval_after['有效词汇占比(%)']}%（核心信息占比提升）")

if __name__ == "__main__":
    main()