你本周聚焦“大模型Embedding生成与优化”主题，今日核心学习任务是掌握Embedding的核心原理、对比主流Embedding模型（开源的BERT-base/text2vec、闭源的通义千问/OpenAI Embedding），并将text2vec（开源免费）和通义千问Embedding（效果优）作为重点学习对象。

### 一、Embedding核心原理（通俗版）
Embedding（嵌入）的本质是**把人类能理解的文本（文字、句子、段落）转换成计算机能处理的数字向量**，核心规则是：**语义越相似的文本，转换后的向量在高维空间中的距离越近**。

举个简单例子：
- 句子A：“我想吃苹果”
- 句子B：“我想吃香蕉”
- 句子C：“今天天气很好”

转换后的向量中，A和B的距离会非常近（都是表达想吃水果），而A/C、B/C的距离会很远（语义无关）。

衡量向量距离最常用的是**余弦相似度**（取值范围[-1,1]）：值越接近1，语义越相似；越接近0/负数，语义越无关。

### 二、主流Embedding模型对比
先通过表格清晰对比你提到的4类模型核心特征：

| 模型类型       | 开源性 | 免费性       | 效果（通用场景） | 部署难度 | 适用场景                     |
|----------------|--------|--------------|------------------|----------|------------------------------|
| BERT-base      | 开源   | 完全免费     | 中等             | 较高     | 定制化改造、学术研究         |
| text2vec       | 开源   | 完全免费     | 中高             | 较低     | 中小规模业务、本地化部署     |
| 通义千问Embedding | 闭源   | 有免费额度   | 高               | 极低     | 大规模业务、追求开箱即用效果 |
| OpenAI Embedding | 闭源  | 有免费额度   | 高               | 极低     | 海外场景、多语言需求         |

#### 重点学习：text2vec（开源）& 通义千问Embedding（闭源）
##### 1. text2vec 使用示例（Python）
text2vec是基于BERT等模型优化的中文Embedding工具，开箱即用，适合新手。

**前置条件**：安装依赖
```bash
pip install text2vec
```

**完整使用代码**：
```python
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
```

##### 2. 通义千问Embedding 使用示例（Python）
通义千问Embedding是阿里云推出的闭源API，效果优于多数开源模型，有免费调用额度，适合快速验证效果。

**前置条件**：
- 注册阿里云账号，获取API Key（https://dashscope.aliyun.com/）
- 安装依赖
```bash
pip install dashscope
```

**完整使用代码**：
```python
import os
import dashscope
from dashscope import TextEmbedding
from numpy import dot
from numpy.linalg import norm

# 1. 配置API Key（替换成你的实际Key）
dashscope.api_key = "your-api-key-here"  # 关键：替换为自己的API Key

# 2. 定义余弦相似度计算函数
def cosine_similarity(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# 3. 调用通义千问Embedding API生成向量
def get_qwen_embedding(text):
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v2,  # 推荐使用v2版本
        input=text
    )
    if resp.status_code == 200:
        return resp.output['embeddings'][0]['embedding']
    else:
        raise Exception(f"API调用失败：{resp.message}")

# 4. 测试示例
if __name__ == "__main__":
    texts = [
        "我想吃苹果",
        "我想吃香蕉",
        "今天天气很好"
    ]
    
    # 生成向量
    embeddings = [get_qwen_embedding(t) for t in texts]
    
    # 计算相似度
    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
    
    print("苹果 vs 香蕉 相似度：", sim_1_2)  # 输出≈0.9+（比text2vec更优）
    print("苹果 vs 天气 相似度：", sim_1_3)  # 输出≈0.05-
```

### 总结
1. **Embedding核心**：文本转高维向量，语义相近则向量余弦相似度高（距离近）；
2. **模型选型关键**：追求免费/本地化部署选text2vec，追求效果/低成本快速落地选通义千问Embedding；
3. **实践要点**：text2vec需本地加载模型，通义千问需申请API Key，两者都可通过余弦相似度验证向量语义关联性。