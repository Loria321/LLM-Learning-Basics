from text2vec import SentenceModel, cos_sim

# 1. 加载预训练模型（中文优先）
model = SentenceModel('shibing624/text2vec-base-chinese')

# 2. 准备待转换的文本
texts = [
    "我想吃苹果",
    "我想吃香蕉",
    "今天天气很好"
]

# 3. 生成Embedding向量（输出维度：768）
embeddings = model.encode(texts)
print("向量维度：", embeddings.shape)  # 输出：(3, 768) → 3个文本，每个文本768维向量

# 4. 计算余弦相似度（验证语义相近性）
sim_1_2 = cos_sim(embeddings[0], embeddings[1])  # 苹果 vs 香蕉
sim_1_3 = cos_sim(embeddings[0], embeddings[2])  # 苹果 vs 天气

print("苹果 vs 香蕉 相似度：", sim_1_2.item())  # 输出≈0.8+（高相似）
print("苹果 vs 天气 相似度：", sim_1_3.item())  # 输出≈0.1-（低相似）