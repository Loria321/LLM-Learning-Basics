# text_feature_engineer.py
"""
传统文本特征工程模块
功能：集成词袋模型/TF-IDF + 特征选择（方差过滤/互信息法） + PCA降维
适用场景：中文文本特征提取，可集成到数据清洗工具箱
"""

import jieba
import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

# ====================== 基础预处理函数 ======================
def chinese_text_preprocess(text: str, stop_words: Optional[List[str]] = None) -> str:
    """
    中文文本预处理：分词 + 过滤停用词
    Args:
        text: 原始中文文本字符串
        stop_words: 停用词列表，None则不过滤
    Returns:
        分词后用空格拼接的字符串
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # 分词（精确模式）
    words = jieba.lcut(text.strip())
    
    # 过滤停用词和空字符串
    if stop_words is not None:
        words = [w for w in words if w not in stop_words and len(w) > 0]
    else:
        words = [w for w in words if len(w) > 0]
    
    return " ".join(words)

# ====================== 核心特征提取类 ======================
class TraditionalTextFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    传统文本特征工程类（兼容sklearn Pipeline）
    集成：特征提取（词袋/TF-IDF） + 特征选择 + PCA降维
    """
    def __init__(
        self,
        feature_type: str = "tfidf",  # 特征类型：count(词袋)/tfidf
        stop_words: Optional[List[str]] = None,  # 停用词列表
        variance_threshold: float = 0.0,  # 方差过滤阈值，0则过滤方差为0的特征
        use_mutual_info: bool = False,  # 是否使用互信息法筛选特征
        mutual_info_k: int = 200,  # 互信息法保留特征数
        use_pca: bool = True,  # 是否使用PCA降维
        pca_n_components: Union[float, int] = 0.9  # PCA保留信息比例（0-1）或维度数
    ):
        # 输入参数校验
        if feature_type not in ["count", "tfidf"]:
            raise ValueError(f"feature_type必须是count或tfidf，当前为{feature_type}")
        if variance_threshold < 0:
            raise ValueError("variance_threshold不能为负数")
        if use_mutual_info and mutual_info_k <= 0:
            raise ValueError("use_mutual_info=True时，mutual_info_k必须大于0")
        if use_pca:
            if isinstance(pca_n_components, float):
                if not (0 < pca_n_components <= 1):
                    raise ValueError("pca_n_components为浮点数时，必须在(0,1]区间")
            elif isinstance(pca_n_components, int):
                if pca_n_components <= 0:
                    raise ValueError("pca_n_components为整数时，必须大于0")
            else:
                raise TypeError("pca_n_components必须是浮点数或整数")
        
        # 初始化参数
        self.feature_type = feature_type
        self.stop_words = stop_words
        self.variance_threshold = variance_threshold
        self.use_mutual_info = use_mutual_info
        self.mutual_info_k = mutual_info_k
        self.use_pca = use_pca
        self.pca_n_components = pca_n_components
        
        # 初始化工具对象（延迟初始化，避免提前占用内存）
        self.vectorizer = None
        self.vt = None
        self.skb = None
        self.pca = None
        
    def fit(self, X: List[str], y: Optional[List[int]] = None) -> "TraditionalTextFeatureEngineer":
        """
        拟合特征提取器/选择器/降维器
        Args:
            X: 原始中文文本列表
            y: 标签列表（互信息法需要）
        Returns:
            self
        """
        # 1. 文本预处理
        print("开始文本预处理...")
        X_processed = [chinese_text_preprocess(text, self.stop_words) for text in X]
        
        # 2. 特征提取（词袋/TF-IDF）
        print(f"开始{self.feature_type.upper()}特征提取...")
        if self.feature_type == "count":
            self.vectorizer = CountVectorizer()
        else:
            self.vectorizer = TfidfVectorizer()
        
        X_feature = self.vectorizer.fit_transform(X_processed).toarray()
        print(f"原始特征维度：{X_feature.shape[1]}")
        
        # 3. 特征选择：方差过滤
        print(f"开始方差过滤（阈值={self.variance_threshold}）...")
        self.vt = VarianceThreshold(threshold=self.variance_threshold)
        X_selected = self.vt.fit_transform(X_feature)
        print(f"方差过滤后维度：{X_selected.shape[1]}")
        
        # 4. 特征选择：互信息法（有监督，需要标签）
        if self.use_mutual_info:
            if y is None:
                raise ValueError("使用互信息法时必须传入标签y")
            print(f"开始互信息法筛选（保留前{self.mutual_info_k}个特征）...")
            self.skb = SelectKBest(score_func=mutual_info_classif, k=self.mutual_info_k)
            X_selected = self.skb.fit_transform(X_selected, y)
            print(f"互信息法筛选后维度：{X_selected.shape[1]}")
        
        # 5. PCA降维
        if self.use_pca:
            print(f"开始PCA降维（保留{self.pca_n_components}信息/维度）...")
            self.pca = PCA(n_components=self.pca_n_components, random_state=42)
            self.pca.fit(X_selected)
            print(f"PCA解释方差比：{np.round(self.pca.explained_variance_ratio_, 3)}")
            print(f"PCA累计解释方差比：{np.round(sum(self.pca.explained_variance_ratio_), 3)}")
        
        return self
    
    def transform(self, X: List[str]) -> np.ndarray:
        """
        转换文本数据为低维特征矩阵
        Args:
            X: 原始中文文本列表
        Returns:
            低维稠密特征矩阵（n_samples, n_components）
        """
        if self.vectorizer is None:
            raise RuntimeError("请先调用fit()方法拟合模型")
        
        # 1. 文本预处理
        X_processed = [chinese_text_preprocess(text, self.stop_words) for text in X]
        
        # 2. 特征提取
        X_feature = self.vectorizer.transform(X_processed).toarray()
        
        # 3. 方差过滤
        X_selected = self.vt.transform(X_feature)
        
        # 4. 互信息法筛选
        if self.use_mutual_info:
            X_selected = self.skb.transform(X_selected)
        
        # 5. PCA降维
        if self.use_pca:
            X_final = self.pca.transform(X_selected)
        else:
            X_final = X_selected
        
        print(f"最终输出特征维度：{X_final.shape[1]}")
        return X_final
    
    def fit_transform(self, X: List[str], y: Optional[List[int]] = None) -> np.ndarray:
        """
        拟合并转换
        """
        self.fit(X, y)
        return self.transform(X)

# ====================== 工具箱集成示例（简化版） ======================
class DataCleaningToolbox:
    """
    数据清洗工具箱（简化版）：集成文本特征提取模块
    """
    def __init__(self):
        # 预设中文停用词（可扩展）
        self.default_stop_words = [
            "的", "地", "得", "我", "你", "他", "她", "它", "什么", "如何",
            "请问", "吗", "呢", "这", "那", "是", "有", "在", "了", "就", "都"
        ]
    
    def extract_text_features(
        self,
        text_data: List[str],
        labels: Optional[List[int]] = None,
        feature_type: str = "tfidf",
        use_mutual_info: bool = False,
        pca_n_components: Union[float, int] = 0.9
    ) -> np.ndarray:
        """
        文本特征提取接口（对外暴露）
        Args:
            text_data: 原始中文文本列表
            labels: 标签列表（互信息法需要）
            feature_type: count/tfidf
            use_mutual_info: 是否使用互信息法
            pca_n_components: PCA参数
        Returns:
            低维特征矩阵
        """
        # 参数校验
        if not isinstance(text_data, list) or len(text_data) == 0:
            raise ValueError("text_data必须是非空列表")
        if use_mutual_info and (labels is None or len(labels) != len(text_data)):
            raise ValueError("use_mutual_info=True时，labels必须非空且长度匹配")
        
        # 初始化特征工程器
        feature_engineer = TraditionalTextFeatureEngineer(
            feature_type=feature_type,
            stop_words=self.default_stop_words,
            use_mutual_info=use_mutual_info,
            mutual_info_k=min(200, len(text_data)//2),  # 自适应k值（不超过样本数的一半）
            use_pca=True,
            pca_n_components=pca_n_components
        )
        
        # 提取特征
        features = feature_engineer.fit_transform(text_data, labels)
        return features

# ====================== 测试代码 ======================
if __name__ == "__main__":
    # 测试数据：校园问答样本（复用之前的扩充样本）
    course_qa = [
        "我想选一门计算机相关的课程，请问有哪些推荐",
        "如何申请本学期的选修课，截止日期是什么时候"
    ]
    library_qa = [
        "图书馆的图书可以借阅多久，逾期会有什么处罚",
        "图书馆的自习室需要提前预约吗，预约方式是什么"
    ]
    leave_qa = [
        "我需要请假三天，请假流程是什么样的",
        "请假需要提交什么材料，多久能审批通过"
    ]
    test_texts = course_qa + library_qa + leave_qa
    test_labels = [0, 0, 1, 1, 2, 2]
    
    # 1. 直接使用特征工程类
    print("=== 测试特征工程类 ===")
    engineer = TraditionalTextFeatureEngineer(
        feature_type="tfidf",
        use_mutual_info=True,
        mutual_info_k=5,
        pca_n_components=0.9
    )
    features = engineer.fit_transform(test_texts, test_labels)
    print(f"特征矩阵形状：{features.shape}")
    print(f"特征矩阵示例：\n{np.round(features, 3)}")
    
    # 2. 集成到数据清洗工具箱
    print("\n=== 测试数据清洗工具箱 ===")
    toolbox = DataCleaningToolbox()
    toolbox_features = toolbox.extract_text_features(
        text_data=test_texts,
        labels=test_labels,
        feature_type="tfidf",
        use_mutual_info=True,
        pca_n_components=0.9
    )
    print(f"工具箱输出特征形状：{toolbox_features.shape}")
    print(f"工具箱输出特征示例：\n{np.round(toolbox_features, 3)}")

