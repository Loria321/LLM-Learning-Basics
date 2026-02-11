from sentence_transformers import SentenceTransformer
import numpy as np
import json

# -------------------------- 1. 初始化模型 --------------------------
# 加载text2vec中文模型（全局加载，仅加载一次）
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print(f"模型加载完成，向量维度：{model.get_sentence_embedding_dimension()}")

# -------------------------- 2. 构造/加载校园问答数据 --------------------------
# 示例校园问答数据（实际场景可从文件读取，如json/csv）
campus_qa_data = [
    {"question": "图书馆几点开门？", "answer": "图书馆工作日8:00开门，周末9:00开门"},
    {"question": "图书馆开放时间？", "answer": "图书馆工作日8:00-22:00，周末9:00-20:00"},
    {"question": "食堂有哪些窗口？", "answer": "一食堂有川菜、面食、快餐窗口；二食堂有清真、套餐窗口"},
    {"question": "如何申请奖学金？", "answer": "登录教务处官网，填写奖学金申请表，提交至辅导员处"},
    {"question": "奖学金申请流程是什么？", "answer": "1.官网填表 2.提交材料 3.学院审核 4.学校公示"},
    {"question": "校园网怎么办理？", "answer": "携带学生证到网络中心办理，需缴纳50元开户费"}
]

# -------------------------- 3. 批量生成Embedding向量 --------------------------
def generate_campus_qa_embeddings(qa_data, model):
    """
    为校园问答数据批量生成Embedding向量（仅对问题生成，问答场景核心是匹配问题）
    :param qa_data: 校园问答列表（含question/answer）
    :param model: text2vec模型实例
    :return: 带Embedding的问答数据
    """
    # 提取所有问题文本
    questions = [item["question"] for item in qa_data]
    
    # 批量生成向量（比单条生成效率高）
    embeddings = model.encode(questions, normalize_embeddings=True)
    
    # 将向量合并回原数据
    qa_data_with_emb = []
    for idx, item in enumerate(qa_data):
        item["embedding"] = embeddings[idx].tolist()  # 转list方便保存
        qa_data_with_emb.append(item)
    
    return qa_data_with_emb

# 执行批量生成
campus_qa_with_emb = generate_campus_qa_embeddings(campus_qa_data, model)

# -------------------------- 4. 验证：相似问题向量相似度高 --------------------------
def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 验证："图书馆几点开门？" 和 "图书馆开放时间？" 相似度
q1_emb = np.array(campus_qa_with_emb[0]["embedding"])
q2_emb = np.array(campus_qa_with_emb[1]["embedding"])
sim_q1_q2 = cosine_similarity(q1_emb, q2_emb)

# 验证："图书馆几点开门？" 和 "食堂有哪些窗口？" 相似度
q3_emb = np.array(campus_qa_with_emb[2]["embedding"])
sim_q1_q3 = cosine_similarity(q1_emb, q3_emb)

print("\n=== 语义相似度验证 ===")
print(f"'图书馆几点开门？' vs '图书馆开放时间？' 相似度：{sim_q1_q2:.4f}")  # 约0.9+（高相似）
print(f"'图书馆几点开门？' vs '食堂有哪些窗口？' 相似度：{sim_q1_q3:.4f}")    # 约0.1-（低相似）

# -------------------------- 5. 保存向量数据（可选） --------------------------
# 将带Embedding的问答数据保存为json文件，方便后续复用
with open("campus_qa_with_embedding.json", "w", encoding="utf-8") as f:
    json.dump(campus_qa_with_emb, f, ensure_ascii=False, indent=2)
print("\n带Embedding的校园问答数据已保存至：campus_qa_with_embedding.json")