你现在需要开发一个可复用的Embedding生成工具模块，核心目标是将开源（text2vec）/闭源（通义千问）的Embedding生成、优化逻辑封装为灵活的函数，支持模型选择、维度调整、归一化配置，并能无缝集成到数据清洗工具箱中作为独立模块。

### 一、工具模块设计思路
我会采用**模块化+面向对象**的设计，保证工具的可复用性和可配置性：
1. 封装核心函数：开源/闭源Embedding生成、归一化+PCA降维优化；
2. 统一调用接口：通过`EmbeddingGenerator`类封装所有功能，参数可配置；
3. 鲁棒性保障：参数校验、异常处理、适配不同维度的向量；
4. 集成友好：作为独立模块，可被数据清洗工具箱导入调用。

### 二、完整Embedding工具模块代码
将以下代码保存为`embedding_toolkit.py`（作为数据清洗工具箱的子模块）：

```python
"""
Embedding生成工具模块
集成到数据清洗工具箱的Embedding子模块，支持：
1. 开源(text2vec)/闭源(通义千问)Embedding生成
2. 归一化、PCA降维优化
3. 灵活配置模型、维度、归一化开关
"""
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# -------------------------- 配置常量（可根据需求修改） --------------------------
# 模型名称映射（统一外部调用的模型名）
MODEL_MAPPING = {
    "text2vec": "shibing624/text2vec-base-chinese",  # 开源text2vec
    "qwen": "text-embedding-v2"                      # 通义千问Embedding
}
# 默认参数
DEFAULT_CONFIG = {
    "normalize": True,        # 默认开启归一化
    "target_dim": None,       # 默认不降维（None）
    "random_state": 42        # PCA随机种子，保证可复现
}

# -------------------------- 核心工具函数 --------------------------
def load_qwen_client():
    """加载通义千问OpenAI兼容客户端（单例模式，避免重复初始化）"""
    load_dotenv()  # 加载环境变量中的DASHSCOPE_API_KEY
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未配置DASHSCOPE_API_KEY环境变量！请先配置阿里云百炼API Key")
    
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

def generate_open_source_embeddings(texts, model_name="text2vec"):
    """
    生成开源Embedding（text2vec）
    :param texts: 文本列表（str/list[str]）
    :param model_name: 模型名（仅支持text2vec）
    :return: 二维numpy数组（n_samples, 768）
    """
    if not isinstance(texts, list):
        texts = [texts]  # 兼容单文本输入
    
    try:
        # 加载text2vec模型（首次运行自动下载）
        model = SentenceTransformer(MODEL_MAPPING[model_name])
        # 生成向量并归一化（基础归一化）
        embeddings = model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)
    except Exception as e:
        raise RuntimeError(f"开源Embedding生成失败：{e}")

def generate_closed_source_embeddings(texts, model_name="qwen"):
    """
    生成闭源Embedding（通义千问）
    :param texts: 文本列表（str/list[str]）
    :param model_name: 模型名（仅支持qwen）
    :return: 二维numpy数组（n_samples, 1536/1024）
    """
    if not isinstance(texts, list):
        texts = [texts]  # 兼容单文本输入
    
    try:
        client = load_qwen_client()
        # 调用通义千问Embedding API
        response = client.embeddings.create(
            model=MODEL_MAPPING[model_name],
            input=texts
        )
        # 提取向量并转为numpy数组
        embeddings = [np.array(item.embedding) for item in response.data]
        return np.array(embeddings)
    except Exception as e:
        raise RuntimeError(f"闭源Embedding生成失败：{e}")

def optimize_embeddings(embeddings, normalize=True, target_dim=None):
    """
    Embedding优化：归一化 + PCA降维
    :param embeddings: 原始Embedding向量（二维numpy数组）
    :param normalize: 是否开启L2归一化（bool）
    :param target_dim: 降维目标维度（None=不降维，int=指定维度）
    :return: 优化后的向量（二维numpy数组）、PCA模型（None=未降维）
    """
    # 步骤1：归一化（可选）
    if normalize:
        optimized_embs = normalize(embeddings, norm='l2', axis=1)
    else:
        optimized_embs = embeddings.copy()
    
    # 步骤2：PCA降维（可选）
    pca_model = None
    if target_dim is not None and target_dim < embeddings.shape[1]:
        # 校验降维维度（不能超过样本数-1）
        n_samples, n_features = embeddings.shape
        max_possible_dim = min(n_samples - 1, n_features)
        if target_dim > max_possible_dim:
            print(f"⚠️ 目标维度{target_dim}超出最大可降维维度{max_possible_dim}，自动调整为{max_possible_dim}")
            target_dim = max_possible_dim
        
        # 执行PCA降维
        pca_model = PCA(n_components=target_dim, random_state=DEFAULT_CONFIG["random_state"])
        optimized_embs = pca_model.fit_transform(optimized_embs)
    
    return optimized_embs, pca_model

# -------------------------- 工具主类（统一调用接口） --------------------------
class EmbeddingGenerator:
    """Embedding生成工具主类，封装所有功能"""
    def __init__(self, model_type="text2vec", normalize=True, target_dim=None):
        """
        初始化工具
        :param model_type: 模型类型（text2vec/qwen）
        :param normalize: 是否开启归一化（默认True）
        :param target_dim: 降维目标维度（默认None=不降维）
        """
        if model_type not in MODEL_MAPPING.keys():
            raise ValueError(f"不支持的模型类型！仅支持：{list(MODEL_MAPPING.keys())}")
        
        self.model_type = model_type
        self.normalize = normalize
        self.target_dim = target_dim
        self.pca_model = None  # 保存训练好的PCA模型，用于新向量降维

    def generate(self, texts):
        """
        生成并优化Embedding
        :param texts: 输入文本（str/list[str]）
        :return: 优化后的Embedding向量（二维numpy数组）
        """
        # 1. 生成原始Embedding
        if self.model_type == "text2vec":
            original_embs = generate_open_source_embeddings(texts, self.model_type)
        else:
            original_embs = generate_closed_source_embeddings(texts, self.model_type)
        
        # 2. 优化Embedding（归一化+降维）
        optimized_embs, self.pca_model = optimize_embeddings(
            original_embs,
            normalize=self.normalize,
            target_dim=self.target_dim
        )
        
        print(f"✅ Embedding生成完成：{len(texts)}条文本 → 维度{optimized_embs.shape[1]}")
        return optimized_embs

    def transform_new_texts(self, texts):
        """
        对新文本使用已训练的PCA模型生成优化向量（避免重复训练PCA）
        :param texts: 新文本（str/list[str]）
        :return: 优化后的Embedding向量（二维numpy数组）
        """
        if self.pca_model is None:
            raise RuntimeError("未训练PCA模型！请先调用generate方法生成过向量")
        
        # 生成原始向量
        if self.model_type == "text2vec":
            original_embs = generate_open_source_embeddings(texts, self.model_type)
        else:
            original_embs = generate_closed_source_embeddings(texts, self.model_type)
        
        # 归一化 + PCA降维（复用已训练的PCA模型）
        if self.normalize:
            normalized_embs = normalize(original_embs, norm='l2', axis=1)
        else:
            normalized_embs = original_embs
        
        optimized_embs = self.pca_model.transform(normalized_embs)
        return optimized_embs

# -------------------------- 集成到数据清洗工具箱的示例 --------------------------
# 假设数据清洗工具箱的主模块为data_cleaning_toolkit.py，可这样集成：
# from .embedding_toolkit import EmbeddingGenerator
class DataCleaningToolkit:
    """数据清洗工具箱主类（示例）"""
    def __init__(self):
        self.embedding_generator = None  # 延迟初始化Embedding工具
    
    def init_embedding_tool(self, model_type="text2vec", normalize=True, target_dim=None):
        """初始化Embedding生成工具"""
        self.embedding_generator = EmbeddingGenerator(model_type, normalize, target_dim)
        print(f"🔧 数据清洗工具箱 - Embedding模块初始化完成（模型：{model_type}）")
    
    def add_embedding_to_data(self, data, text_col="text"):
        """
        为数据添加Embedding向量列
        :param data: 待处理数据（list[dict]，如校园问答数据）
        :param text_col: 文本列名（默认text）
        :return: 带Embedding的新数据
        """
        if self.embedding_generator is None:
            raise RuntimeError("请先调用init_embedding_tool初始化Embedding工具")
        
        # 提取文本列
        texts = [item[text_col] for item in data]
        # 生成Embedding
        embeddings = self.embedding_generator.generate(texts)
        # 合并向量到数据
        new_data = []
        for idx, item in enumerate(data):
            item["embedding"] = embeddings[idx].tolist()  # 转list方便保存
            new_data.append(item)
        
        return new_data
```

### 三、工具使用示例（测试+集成）
创建`test_embedding_tool.py`，测试工具功能并演示集成到数据清洗工具箱的用法：

```python
from embedding_toolkit import EmbeddingGenerator, DataCleaningToolkit

# -------------------------- 1. 基础使用：单独调用Embedding生成工具 --------------------------
if __name__ == "__main__":
    # 示例校园问答数据
    campus_qa_data = [
        {"question": "图书馆几点开门？", "answer": "图书馆工作日8:00开门，周末9:00开门"},
        {"question": "如何申请奖学金？", "answer": "登录教务处官网，填写奖学金申请表，提交至辅导员处"},
        {"question": "校园网怎么办理？", "answer": "携带学生证到网络中心办理，需缴纳50元开户费"}
    ]
    
    # ------------ 示例1：使用text2vec生成768维向量，降维到3维（适配样本数） ------------
    print("=== 示例1：开源text2vec生成优化向量 ===")
    generator = EmbeddingGenerator(model_type="text2vec", normalize=True, target_dim=3)
    texts = [item["question"] for item in campus_qa_data]
    embeddings = generator.generate(texts)
    print(f"生成的向量维度：{embeddings.shape}")
    
    # ------------ 示例2：使用通义千问生成1536维向量，降维到2维 ------------
    print("\n=== 示例2：闭源通义千问生成优化向量 ===")
    generator_qwen = EmbeddingGenerator(model_type="qwen", normalize=True, target_dim=2)
    embeddings_qwen = generator_qwen.generate(texts)
    print(f"生成的向量维度：{embeddings_qwen.shape}")
    
    # ------------ 示例3：集成到数据清洗工具箱使用 ------------
    print("\n=== 示例3：集成到数据清洗工具箱 ===")
    # 初始化数据清洗工具箱
    toolkit = DataCleaningToolkit()
    # 初始化Embedding模块（使用text2vec，降维到3维）
    toolkit.init_embedding_tool(model_type="text2vec", normalize=True, target_dim=3)
    # 为校园问答数据添加Embedding列
    data_with_emb = toolkit.add_embedding_to_data(campus_qa_data, text_col="question")
    # 输出结果
    for item in data_with_emb:
        print(f"问题：{item['question']} → Embedding维度：{len(item['embedding'])}")
```

### 四、核心功能解释
1. **模块化封装**：
   - 核心函数（生成/优化）独立封装，可单独调用；
   - `EmbeddingGenerator`类提供统一接口，简化外部调用；
   - 常量配置（`MODEL_MAPPING`/`DEFAULT_CONFIG`）集中管理，便于维护。

2. **灵活配置项**：
   - `model_type`：支持选择`text2vec`（开源）/`qwen`（闭源）；
   - `normalize`：开关式控制是否开启L2归一化；
   - `target_dim`：可指定降维目标维度（None=不降维），自动校验维度合法性。

3. **集成友好性**：
   - 作为独立模块`embedding_toolkit.py`，可通过`import`导入到数据清洗工具箱；
   - `DataCleaningToolkit`示例演示了如何将Embedding模块作为子功能集成，保持工具箱的整体性；
   - 输出向量转为`list`格式，避免numpy数组序列化问题，方便保存/传输。

4. **鲁棒性设计**：
   - 兼容单文本/文本列表输入；
   - 完善的异常捕获与提示，快速定位问题（如API Key缺失、模型加载失败）；
   - PCA维度自动校验，避免样本数不足导致的报错。

### 总结
1. **工具核心特性**：
   - 支持开源（text2vec）/闭源（通义千问）双模型，参数统一配置；
   - 内置归一化+PCA降维优化，维度自动适配，无需手动处理；
   - 面向对象封装，既支持单独调用，也可无缝集成到数据清洗工具箱。
2. **使用关键点**：
   - 通义千问需提前配置`DASHSCOPE_API_KEY`环境变量；
   - 降维维度需≤样本数-1（测试场景），真实场景扩充样本后可设为768维；
   - `transform_new_texts`方法可复用PCA模型，避免新文本重复训练。
3. **集成方式**：
   - 将`embedding_toolkit.py`放入数据清洗工具箱目录；
   - 通过`from .embedding_toolkit import EmbeddingGenerator`导入，作为工具箱的子模块使用。