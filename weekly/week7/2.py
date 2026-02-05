# 导入所需库（新增特征选择和降维相关工具）
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import jieba
import numpy as np

# 步骤1：复用之前的校园问答数据和预处理流程
# 1. 校园问答样本数据
campus_qa = [
    "我想选一门计算机相关的课程，请问有哪些推荐",
    "如何申请本学期的选修课，截止日期是什么时候",
    "图书馆的图书可以借阅多久，逾期会有什么处罚",
    "我需要请假三天，请假流程是什么样的",
    "图书馆的自习室需要提前预约吗，预约方式是什么",
    "请假需要提交什么材料，多久能审批通过"
]

# 2. 中文分词函数
def chinese_word_cut(text):
    return " ".join(jieba.lcut(text))

# 3. 分词预处理
processed_qa = [chinese_word_cut(qa) for qa in campus_qa]

# 4. 生成TF-IDF特征矩阵（复用之前的逻辑，无停用词过滤，保持和之前结果一致）
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(processed_qa)
tfidf_array = tfidf_matrix.toarray()  # 转换为数组，方便后续处理
vocab_list = tfidf_vec.get_feature_names_out()

# 输出原始TF-IDF矩阵信息
print("="*60)
print("【原始TF-IDF矩阵信息】")
print(f"原始特征维度（词汇表大小）：{tfidf_array.shape[1]}")
print(f"原始矩阵形状（文档数×特征维度）：{tfidf_array.shape}")

# 步骤2：特征选择 - 方差过滤
# 初始化方差过滤器，默认threshold=0（过滤方差为0的特征）
vt = VarianceThreshold(threshold=0.0)

# 对TF-IDF矩阵进行方差过滤（仅支持稠密矩阵，稀疏矩阵需先转换为数组）
tfidf_filtered_vt = vt.fit_transform(tfidf_array)

# 输出方差过滤结果
print("\n" + "="*60)
print("【方差过滤结果（threshold=0.0）】")
print(f"过滤后特征维度：{tfidf_filtered_vt.shape[1]}")
print(f"过滤后矩阵形状：{tfidf_filtered_vt.shape}")
print(f"被过滤的特征数量：{tfidf_array.shape[1] - tfidf_filtered_vt.shape[1]}")

# （可选）查看被过滤的特征（词汇）
filtered_indices = np.where(vt.variances_ <= 0.0)[0]
if len(filtered_indices) > 0:
    print(f"被过滤的词汇：{vocab_list[filtered_indices]}")
else:
    print("无方差为0的特征，所有特征均被保留")

# 步骤3：特征选择 - 互信息法（有监督，需先标注标签）
# 1. 给校园问答数据标注标签（0=选课，1=图书馆，2=请假）
labels = [0, 0, 1, 2, 1, 2]
label_names = ["选课", "图书馆", "请假"]

# 2. 初始化：选择互信息值前10名的特征（SelectKBest + mutual_info_classif）
# mutual_info_classif：计算特征与标签的互信息值（分类任务）
skb = SelectKBest(score_func=mutual_info_classif, k=10)

# 3. 拟合+转换（对TF-IDF矩阵进行高互信息特征筛选）
tfidf_filtered_mi = skb.fit_transform(tfidf_array, labels)

# 4. 输出互信息法结果
print("\n" + "="*60)
print("【互信息法筛选结果（选择前10个高互信息特征）】")
print(f"筛选后特征维度：{tfidf_filtered_mi.shape[1]}")
print(f"筛选后矩阵形状：{tfidf_filtered_mi.shape}")

# 5. 查看高互信息特征（词汇）及其互信息值
mi_scores = skb.scores_
mi_top10_indices = skb.get_support(indices=True)
mi_top10_words = vocab_list[mi_top10_indices]
mi_top10_scores = mi_scores[mi_top10_indices]

print(f"\n前10个高互信息词汇及对应得分（保留3位小数）：")
for word, score in sorted(zip(mi_top10_words, mi_top10_scores), key=lambda x: x[1], reverse=True):
    print(f"{word}: {round(score, 3)}")

# 步骤4：降维 - PCA（保留90%的信息）
# 初始化PCA，n_components=0.9表示保留累计解释方差比≥90%的主成分
pca = PCA(n_components=0.9, random_state=42)

# 对TF-IDF矩阵进行PCA降维（支持稠密矩阵，直接传入tfidf_array）
tfidf_pca = pca.fit_transform(tfidf_array)

# 输出PCA降维结果
print("\n" + "="*60)
print("【PCA降维结果（保留90%信息）】")
print(f"降维后特征维度（主成分数量）：{tfidf_pca.shape[1]}")
print(f"降维后矩阵形状（文档数×主成分数量）：{tfidf_pca.shape}")
print(f"各主成分的解释方差比（保留3位小数）：{[round(x, 3) for x in pca.explained_variance_ratio_]}")
print(f"累计解释方差比（保留3位小数）：{round(sum(pca.explained_variance_ratio_), 3)}")

# 查看降维后的主成分矩阵（每个文档对应各主成分的取值）
print(f"\n降维后主成分矩阵（保留3位小数）：")
print(np.round(tfidf_pca, 3))