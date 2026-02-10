import os
import numpy as np
from openai import OpenAI
from typing import List

# -------------------------- 基础配置（保留你提供的核心配置） --------------------------
client = OpenAI(
    # 从环境变量读取API Key（推荐方式，避免硬编码泄露）
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 北京地域base-url（新加坡地域替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1）
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 待生成Embedding的3个指定文本
TEXT_LIST = [
    "杭电智科专业大二课程",
    "杭州电子科技大学智能科学与技术专业二年级课程",
    "华为大模型应用开发岗位要求"
]

# -------------------------- 核心工具函数 --------------------------
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个Embedding向量的余弦相似度（值域[-1,1]，越接近1语义越相似）
    :param vec1: 第一个向量
    :param vec2: 第二个向量
    :return: 余弦相似度值
    """
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    # 避免除零错误
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # 计算余弦相似度
    similarity = np.dot(vec1_np, vec2_np) / (norm1 * norm2)
    return float(similarity)

# -------------------------- 主执行流程 --------------------------
if __name__ == "__main__":
    # 1. 批量生成3个文本的Embedding（兼容OpenAI接口格式）
    print("正在批量生成文本Embedding...")
    resp = client.embeddings.create(
        model="text-embedding-v4",  # 通义千问Embedding V4模型
        input=TEXT_LIST,  # 批量传入3个文本
        dimensions=256  # 向量维度设置为256
    )
    print("Embedding生成完成！\n")

    # 2. 提取每个文本对应的Embedding向量（保持与TEXT_LIST的顺序一致）
    embedding_list = [item.embedding for item in resp.data]
    emb1, emb2, emb3 = embedding_list  # 分别对应3个文本的向量

    # 3. 输出每个文本的基本信息和向量维度
    for idx, (text, embedding) in enumerate(zip(TEXT_LIST, embedding_list), 1):
        print(f"文本{idx}：{text}")
        print(f"文本{idx} Embedding向量维度：{len(embedding)}\n")

    # 4. 计算并输出语义相似度
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)
    sim_2_3 = cosine_similarity(emb2, emb3)

    print("=" * 60)
    print("语义相似度计算结果（余弦相似度）：")
    print(f"文本1 ↔ 文本2：{sim_1_2:.4f}（语义高度相似，预期接近1）")
    print(f"文本1 ↔ 文本3：{sim_1_3:.4f}（语义不相似，预期接近0）")
    print(f"文本2 ↔ 文本3：{sim_2_3:.4f}（语义不相似，预期接近0）")
    print("=" * 60)