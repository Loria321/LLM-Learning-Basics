import os
import json
import numpy as np
from openai import OpenAI

# -------------------------- 1. 核心工具函数 --------------------------
def cosine_similarity(vec1, vec2):
    """
    计算两个同维度向量的余弦相似度
    注：余弦相似度仅关注向量方向，必须保证两个向量维度完全一致
    """
    # 增加维度一致性校验，提前抛出明确错误
    if len(vec1) != len(vec2):
        raise ValueError(f"向量维度不匹配：vec1({len(vec1)}维) vs vec2({len(vec2)}维)")
    
    # 归一化向量（消除长度影响，提升相似度准确性）
    vec1_norm = vec1 / np.linalg.norm(vec1)
    vec2_norm = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1_norm, vec2_norm)

def get_qwen_embeddings(texts):
    """调用通义千问Embedding API生成闭源向量"""
    try:
        # 初始化OpenAI兼容客户端
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 调用Embedding API（通义千问1024维模型）
        response = client.embeddings.create(
            model="text-embedding-v2",  # 固定模型名，输出1024维向量
            input=texts
        )
        
        # 提取向量（转为numpy数组，方便计算）
        embeddings = [np.array(item.embedding) for item in response.data]
        return embeddings
    
    except Exception as e:
        print(f"通义千问API调用失败：{e}")
        raise

# -------------------------- 2. 加载校园问答数据（带text2vec向量） --------------------------
# 读取之前保存的开源向量数据
try:
    with open("campus_qa_with_embedding.json", "r", encoding="utf-8") as f:
        campus_qa_data = json.load(f)
    print("✅ 成功加载校园问答数据（含text2vec开源向量）")
except FileNotFoundError:
    print("❌ 未找到campus_qa_with_embedding.json文件，请先运行开源Embedding脚本生成！")
    exit(1)

# 提取问题文本和text2vec向量
questions = [item["question"] for item in campus_qa_data]
text2vec_embeddings = [np.array(item["embedding"]) for item in campus_qa_data]

# -------------------------- 3. 调用通义千问生成闭源向量 --------------------------
print("\n📡 开始调用通义千问Embedding API...")
qwen_embeddings = get_qwen_embeddings(questions)
print("✅ 通义千问闭源向量生成完成")

# -------------------------- 4. 对比开源/闭源向量维度 --------------------------
print("\n=== 向量维度对比 ===")
text2vec_dim = len(text2vec_embeddings[0])
qwen_dim = len(qwen_embeddings[0])
print(f"text2vec（开源）向量维度：{text2vec_dim} 维")
print(f"通义千问（闭源）向量维度：{qwen_dim} 维")

# -------------------------- 5. 验证语义一致性（余弦相似度） --------------------------
print("\n=== 语义一致性验证（余弦相似度） ===")

# 定义要验证的问题对（相似问题+无关问题）
test_cases = [
    # (问题1索引, 问题2索引, 问题描述)
    (0, 1, "'图书馆几点开门？' vs '图书馆开放时间？'（相似问题）"),
    (0, 2, "'图书馆几点开门？' vs '食堂有哪些窗口？'（无关问题）"),
    (3, 4, "'如何申请奖学金？' vs '奖学金申请流程是什么？'（相似问题）")
]

# 分别计算开源/闭源向量的相似度，对比结果
for idx1, idx2, desc in test_cases:
    # 开源向量相似度（text2vec）
    sim_text2vec = cosine_similarity(text2vec_embeddings[idx1], text2vec_embeddings[idx2])
    # 闭源向量相似度（通义千问）
    sim_qwen = cosine_similarity(qwen_embeddings[idx1], qwen_embeddings[idx2])
    
    print(f"\n{desc}：")
    print(f"  text2vec相似度：{sim_text2vec:.4f}")
    print(f"  通义千问相似度：{sim_qwen:.4f}")

# -------------------------- 6. 保存闭源向量数据（可选） --------------------------
# 将通义千问向量合并到原数据并保存
campus_qa_with_both_emb = []
for idx, item in enumerate(campus_qa_data):
    item["qwen_embedding"] = qwen_embeddings[idx].tolist()  # 转list方便保存
    campus_qa_with_both_emb.append(item)

with open("campus_qa_with_both_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(campus_qa_with_both_emb, f, ensure_ascii=False, indent=2)
print("\n✅ 带开源+闭源向量的校园问答数据已保存至：campus_qa_with_both_embeddings.json")