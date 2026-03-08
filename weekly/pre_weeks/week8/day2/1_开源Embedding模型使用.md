你现在聚焦“开源Embedding模型使用”的学习，核心任务是安装sentence-transformers库并加载text2vec模型、编写文本转Embedding向量的脚本，以及实操为校园问答数据生成对应的开源Embedding向量。

### 一、环境准备（安装依赖）
text2vec底层依赖`sentence-transformers`库，先通过pip完成安装（建议在虚拟环境中操作）：
```bash
# 安装核心依赖
pip install sentence-transformers text2vec numpy
# 若安装慢，可换国内源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sentence-transformers text2vec numpy
```

### 二、基础脚本：输入文本→生成Embedding向量
先编写最基础的脚本，实现“输入任意文本→输出对应的Embedding向量”，理解核心调用逻辑：
```python
from sentence_transformers import SentenceTransformer
import numpy as np

def text_to_embedding(text, model=None):
    """
    将单条文本转换为Embedding向量
    :param text: 输入文本（字符串）
    :param model: 加载好的text2vec模型实例
    :return: 一维Embedding向量（numpy数组）
    """
    # 校验输入文本
    if not text or not isinstance(text, str):
        raise ValueError("输入文本必须是非空字符串！")
    
    # 若未传入模型，则加载text2vec中文基础模型
    if model is None:
        # 加载开源text2vec模型（自动下载，首次运行需等待）
        model = SentenceTransformer('shibing624/text2vec-base-chinese')
    
    # 生成Embedding向量（encode方法返回numpy数组）
    embedding = model.encode(text, normalize_embeddings=True)  # 归一化向量，便于后续计算相似度
    return embedding

# 测试基础功能
if __name__ == "__main__":
    # 1. 加载模型（全局加载一次，避免重复加载耗时）
    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    print("模型加载完成，向量维度：", model.get_sentence_embedding_dimension())  # 输出768（text2vec-base-chinese默认维度）
    
    # 2. 输入文本并生成向量
    input_text = input("请输入要生成Embedding的文本：")
    try:
        embedding = text_to_embedding(input_text, model)
        print(f"\n生成的Embedding向量（前10维）：{embedding[:10]}")
        print(f"向量总维度：{len(embedding)}")
    except Exception as e:
        print(f"生成向量失败：{e}")
```

### 三、实操：为校园问答数据生成Embedding向量
校园问答数据是典型的Embedding应用场景（如智能问答、相似问题检索），以下脚本实现“批量加载校园问答数据→生成Embedding向量→保存向量（可选）→验证语义相关性”：

#### 完整实操脚本
```python
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
```

### 关键代码解释
1. **模型加载**：`SentenceTransformer('shibing624/text2vec-base-chinese')`是text2vec的中文基础模型，首次运行会自动下载（约1GB），后续运行直接加载本地缓存；
2. **批量编码**：`model.encode(questions)`支持批量输入文本列表，比循环单条编码效率提升5-10倍，`normalize_embeddings=True`会将向量归一化，让余弦相似度计算更准确；
3. **校园问答适配**：仅对“问题”生成Embedding（问答场景中，用户输入的是问题，需匹配已有问题的Embedding），答案仅作为配套信息；
4. **相似度验证**：通过余弦相似度验证“语义相近的问题向量距离近”，符合Embedding核心原理，也验证了text2vec模型的效果；
5. **数据保存**：将向量转为list并保存为JSON，避免numpy数组序列化问题，方便后续用于智能问答、相似问题检索等场景。

### 总结
1. 开源Embedding使用核心步骤：**安装sentence-transformers→加载text2vec模型→调用encode方法生成向量**，normalize_embeddings=True是提升相似度计算准确性的常用优化；
2. 校园问答场景中，优先对**问题文本**生成Embedding，批量编码比单条编码更高效；
3. 可通过余弦相似度验证生成的Embedding效果：语义相近的校园问题（如图书馆开放时间相关）相似度应显著高于无关问题（如图书馆vs食堂）。