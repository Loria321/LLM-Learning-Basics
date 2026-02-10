# 导入所需库
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba

# 步骤1：准备校园问答样本数据（构造6条典型校园问答文本）
campus_qa = [
    "我想选一门计算机相关的课程，请问有哪些推荐",
    "如何申请本学期的选修课，截止日期是什么时候",
    "图书馆的图书可以借阅多久，逾期会有什么处罚",
    "我需要请假三天，请假流程是什么样的",
    "图书馆的自习室需要提前预约吗，预约方式是什么",
    "请假需要提交什么材料，多久能审批通过"
]

# 步骤2：中文分词处理（定义分词函数）
def chinese_word_cut(text):
    # jieba.cut返回生成器，转换为列表后用空格拼接
    return " ".join(jieba.lcut(text))

# 对所有校园问答文本进行分词预处理
processed_qa = [chinese_word_cut(qa) for qa in campus_qa]
print("分词后的文本示例：")
for i, qa in enumerate(processed_qa[:2]):
    print(f"{i+1}. {qa}")

# 步骤3：使用CountVectorizer实现词袋模型
# 初始化CountVectorizer（不过滤停用词，先直观查看结果）
count_vec = CountVectorizer()

# 拟合（构建词汇表）+ 转换（生成频率矩阵）
bow_matrix = count_vec.fit_transform(processed_qa)

# 输出结果分析
print("\n" + "="*50)
print("【词袋模型结果】")
# 1. 输出构建的词汇表（所有不重复词汇）
vocab_list = count_vec.get_feature_names_out()
print(f"词汇表大小：{len(vocab_list)}")
print(f"词汇表示例（前10个）：{vocab_list[:10]}")

# 2. 输出词袋频率矩阵（转换为数组，方便查看）
bow_array = bow_matrix.toarray()
print(f"\n词袋矩阵形状（文档数×词汇表大小）：{bow_array.shape}")
print(f"\n词袋矩阵详情：")
for i, (original_qa, bow_vec) in enumerate(zip(campus_qa, bow_array)):
    print(f"文档{i+1}：{original_qa[:20]}...")
    print(f"对应向量（非零值）：{dict(zip(vocab_list[bow_vec>0], bow_vec[bow_vec>0]))}\n")

# 步骤4：使用TfidfVectorizer实现TF-IDF特征提取
# 初始化TfidfVectorizer（默认进行TF-IDF标准化，权重范围0-1）
tfidf_vec = TfidfVectorizer()

# 拟合（构建词汇表）+ 转换（生成TF-IDF权重矩阵）
tfidf_matrix = tfidf_vec.fit_transform(processed_qa)

# 输出结果分析
print("="*50)
print("【TF-IDF模型结果】")
# 1. 输出构建的词汇表
tfidf_vocab_list = tfidf_vec.get_feature_names_out()
print(f"词汇表大小：{len(tfidf_vocab_list)}")
print(f"词汇表示例（前10个）：{tfidf_vocab_list[:10]}")

# 2. 输出TF-IDF权重矩阵（转换为数组，方便查看）
tfidf_array = tfidf_matrix.toarray()
print(f"\nTF-IDF矩阵形状（文档数×词汇表大小）：{tfidf_array.shape}")
print(f"\nTF-IDF矩阵详情（保留3位小数）：")
for i, (original_qa, tfidf_vec) in enumerate(zip(campus_qa, tfidf_array)):
    print(f"文档{i+1}：{original_qa[:20]}...")
    # 只输出非零且权重前5的词汇（更清晰）
    non_zero_indices = tfidf_vec > 0
    top5_words = sorted(
        zip(tfidf_vocab_list[non_zero_indices], tfidf_vec[non_zero_indices]),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    print(f"对应高权重词汇：{dict((w, round(v, 3)) for w, v in top5_words)}\n")

# 步骤5：优化 - 加入停用词过滤
# 定义中文校园场景常用停用词列表（可根据需求扩展）
stop_words = ["的", "我", "你", "他", "她", "它", "什么", "如何", "请问", "吗", "呢", "这", "那", "是", "有"]

# 初始化带停用词过滤的TfidfVectorizer
tfidf_vec_opt = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix_opt = tfidf_vec_opt.fit_transform(processed_qa)

# 输出优化后的结果
print("="*50)
print("【优化后TF-IDF结果（过滤停用词）】")
tfidf_vocab_opt = tfidf_vec_opt.get_feature_names_out()
print(f"优化后词汇表大小：{len(tfidf_vocab_opt)}")
print(f"优化后词汇表示例：{tfidf_vocab_opt[:10]}")