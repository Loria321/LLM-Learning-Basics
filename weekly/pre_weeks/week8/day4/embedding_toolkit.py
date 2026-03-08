"""
Embedding生成工具模块
集成到数据清洗工具箱的Embedding子模块，支持：
1. 开源(text2vec)/闭源(通义千问)Embedding生成
2. 归一化、PCA降维优化
3. 灵活配置模型、维度、归一化开关
"""
import os
import numpy as np
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