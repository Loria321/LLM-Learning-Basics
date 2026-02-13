import os
import json
import timeit
import numpy as np
from openai import OpenAI
# from dotenv import load_dotenv
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# 加载环境变量（若需重新生成通义千问向量则启用）
# load_dotenv()

# -------------------------- 1. 核心工具函数 --------------------------
def cosine_similarity(vec1, vec2):
    """计算余弦相似度（兼容归一化/非归一化向量）"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_qwen_embeddings():
    """加载之前保存的通义千问向量数据"""
    try:
        with open("campus_qa_with_both_embeddings.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        # 提取问题和通义千问向量（转为numpy数组）
        questions = [item["question"] for item in data]
        qwen_embeddings = [np.array(item["qwen_embedding"]) for item in data]
        return questions, np.array(qwen_embeddings)  # 转为二维数组（n_samples, 1536）
    except FileNotFoundError:
        print("❌ 未找到向量文件，请先运行闭源Embedding脚本生成！")
        exit(1)

def optimize_embeddings(embeddings, target_dim=5):
    """
    对Embedding做优化：归一化 + PCA降维
    【修复点】：目标维度改为5（≤样本数6-1），适配测试样本少的场景
    :param embeddings: 原始1536维通义千问向量（二维数组）
    :param target_dim: 降维目标维度（测试场景设为5，真实场景可改回768）
    :return: 优化后的向量、PCA模型（用于后续新向量降维）
    """
    # 步骤1：归一化（L2归一化，让向量范数=1，提升相似度计算准确性）
    normalized_emb = normalize(embeddings, norm='l2', axis=1)
    
    # 步骤2：PCA降维（修复：目标维度≤样本数-1）
    # 先校验目标维度，避免再次报错
    max_possible_dim = min(len(embeddings)-1, embeddings.shape[1])
    if target_dim > max_possible_dim:
        print(f"⚠️ 目标维度{target_dim}超过最大可降维维度{max_possible_dim}，自动调整为{max_possible_dim}")
        target_dim = max_possible_dim
    
    pca = PCA(n_components=target_dim, random_state=42)  # 固定随机种子保证可复现
    optimized_emb = pca.fit_transform(normalized_emb)
    
    # 输出PCA降维的信息保留率
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"✅ PCA降维完成：{embeddings.shape[1]}维→{target_dim}维，信息保留率：{explained_variance:.4f}（越高越好）")
    
    return optimized_emb, pca

def retrieve_similar_question(query_emb, corpus_embs, corpus_questions, top_k=1):
    """
    检索与查询向量最相似的问题（核心检索逻辑）
    :param query_emb: 查询向量（一维）
    :param corpus_embs: 语料库向量（二维）
    :param corpus_questions: 语料库问题列表
    :param top_k: 返回最相似的top-k个问题
    :return: 最相似的问题、相似度值
    """
    # 计算查询向量与所有语料向量的相似度
    similarities = [cosine_similarity(query_emb, emb) for emb in corpus_embs]
    # 按相似度排序，取top-k
    sorted_idx = np.argsort(similarities)[::-1][:top_k]
    top_question = corpus_questions[sorted_idx[0]]
    top_similarity = similarities[sorted_idx[0]]
    return top_question, top_similarity

# -------------------------- 2. 加载数据 & 优化向量 --------------------------
# 加载原始通义千问向量（1536维）
corpus_questions, original_embs = load_qwen_embeddings()
print(f"📊 原始通义千问向量：{original_embs.shape}（样本数×维度）")

# 优化向量（归一化+PCA降维，适配样本数少的场景）
optimized_embs, pca_model = optimize_embeddings(original_embs, target_dim=5)
print(f"📊 优化后向量：{optimized_embs.shape}（样本数×维度）")

# -------------------------- 3. 构建检索测试集（验证准确率） --------------------------
# 测试集：查询问题 → 预期匹配的目标问题（语义一致）
test_queries = [
    {"query": "图书馆早上几点开门？", "expected": "图书馆几点开门？"},
    {"query": "奖学金怎么申请？", "expected": "如何申请奖学金？"},
    {"query": "食堂都有什么吃的？", "expected": "食堂有哪些窗口？"},
    {"query": "办理校园网需要什么？", "expected": "校园网怎么办理？"},
    {"query": "图书馆周末开放吗？", "expected": "图书馆开放时间？"}
]

# 生成查询问题的向量（原始+优化）
# 初始化OpenAI客户端（生成查询向量）
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 提取查询文本，批量生成原始向量
query_texts = [item["query"] for item in test_queries]
query_original_embs = [
    np.array(item.embedding) for item in client.embeddings.create(model="text-embedding-v2", input=query_texts).data
]
# 对查询向量做相同的优化（归一化+PCA降维）
query_normalized_embs = normalize(np.array(query_original_embs), norm='l2', axis=1)
query_optimized_embs = pca_model.transform(query_normalized_embs)

# -------------------------- 4. 对比检索速度（多次运行取平均） --------------------------
print("\n=== 检索速度对比（运行100次取平均）===")

# 定义速度测试函数
def test_retrieval_speed(embs_type):
    """测试检索速度：embs_type='original'/'optimized'"""
    total_time = 0
    run_times = 100  # 多次运行减少误差
    for i in range(run_times):
        start = timeit.default_timer()
        # 随机选一个查询向量测试
        idx = np.random.randint(0, len(query_original_embs))
        if embs_type == "original":
            retrieve_similar_question(query_original_embs[idx], original_embs, corpus_questions)
        else:
            retrieve_similar_question(query_optimized_embs[idx], optimized_embs, corpus_questions)
        end = timeit.default_timer()
        total_time += (end - start)
    avg_time = (total_time / run_times) * 1000  # 转毫秒
    return avg_time

# 测试原始向量检索速度
original_speed = test_retrieval_speed("original")
# 测试优化后向量检索速度
optimized_speed = test_retrieval_speed("optimized")

print(f"原始{original_embs.shape[1]}维向量检索平均耗时：{original_speed:.4f} 毫秒")
print(f"优化后{optimized_embs.shape[1]}维向量检索平均耗时：{optimized_speed:.4f} 毫秒")
print(f"速度提升：{((original_speed - optimized_speed)/original_speed)*100:.2f}%")

# -------------------------- 5. 对比检索准确率（Top-1命中率） --------------------------
print("\n=== 检索准确率对比（Top-1命中率）===")

def calculate_accuracy(query_embs, corpus_embs, test_queries, corpus_questions):
    """计算Top-1检索准确率"""
    correct = 0
    total = len(test_queries)
    for idx, test_item in enumerate(test_queries):
        retrieved_q, _ = retrieve_similar_question(query_embs[idx], corpus_embs, corpus_questions)
        if retrieved_q == test_item["expected"]:
            correct += 1
        else:
            print(f"❌ 检索错误：查询「{test_item['query']}」→ 检索到「{retrieved_q}」（预期：{test_item['expected']}）")
    accuracy = (correct / total) * 100
    return accuracy

# 计算原始向量准确率
original_acc = calculate_accuracy(query_original_embs, original_embs, test_queries, corpus_questions)
# 计算优化后向量准确率
optimized_acc = calculate_accuracy(query_optimized_embs, optimized_embs, test_queries, corpus_questions)

print(f"\n原始{original_embs.shape[1]}维向量检索准确率：{original_acc:.2f}%")
print(f"优化后{optimized_embs.shape[1]}维向量检索准确率：{optimized_acc:.2f}%")
print(f"准确率变化：{optimized_acc - original_acc:.2f} 个百分点")

# -------------------------- 6. 保存优化后的向量（可选） --------------------------
# 合并优化向量到原数据并保存
with open("campus_qa_with_both_embeddings.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)
for idx, item in enumerate(original_data):
    item["qwen_optimized_embedding"] = optimized_embs[idx].tolist()
with open("campus_qa_with_optimized_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(original_data, f, ensure_ascii=False, indent=2)
print("\n✅ 优化后的向量已保存至：campus_qa_with_optimized_embeddings.json")