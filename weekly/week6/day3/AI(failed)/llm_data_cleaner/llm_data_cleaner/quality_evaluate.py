import re
import jieba
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Callable
from .utils import read_file, save_file

def calculate_text_smoothness(
    text: Union[str, List[str]],
    smoothness_func: Optional[Callable] = None
) -> Union[float, List[float]]:
    """
    文本通顺度评分（简易版：基于标点符号合理性+无乱码）
    :param text: 待评估文本/文本列表
    :param smoothness_func: 自定义通顺度函数（输入文本，输出0-1评分）
    :return: 通顺度评分（0-1）/评分列表
    """
    # 默认通顺度函数
    def _default_smoothness(t):
        t = t.strip()
        if not t:
            return 0.0
        # 乱码检测（含大量非中文字符/特殊字符）
        chinese_ratio = len(re.findall(r"[\u4e00-\u9fa5]", t)) / len(t)
        if chinese_ratio < 0.5:
            return 0.2
        # 标点符号合理性（避免连续标点）
        if re.search(r"[，。！？；：]{2,}", t):
            return 0.5
        return 1.0
    
    smoothness_func = smoothness_func or _default_smoothness
    
    if isinstance(text, str):
        return smoothness_func(text)
    elif isinstance(text, list):
        return [smoothness_func(t) for t in text]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def calculate_duplicate_rate(
    text_list: List[str],
    ngram: int = 3
) -> float:
    """
    文本重复度评估（基于ngram相似度）
    :param text_list: 文本列表
    :param ngram: ngram长度
    :return: 重复度（0-1，越高重复越多）
    """
    if len(text_list) < 2:
        return 0.0
    
    # 生成ngram集合
    def _get_ngrams(t):
        t = t.strip()
        if len(t) < ngram:
            return set()
        return set([t[i:i+ngram] for i in range(len(t)-ngram+1)])
    
    ngram_sets = [_get_ngrams(t) for t in text_list if t.strip()]
    if not ngram_sets:
        return 0.0
    
    # 计算平均交集率
    total_similarity = 0.0
    count = 0
    for i in range(len(ngram_sets)):
        for j in range(i+1, len(ngram_sets)):
            if ngram_sets[i] and ngram_sets[j]:
                intersection = len(ngram_sets[i] & ngram_sets[j])
                union = len(ngram_sets[i] | ngram_sets[j])
                total_similarity += intersection / union if union else 0
                count += 1
    return total_similarity / count if count else 0.0

def calculate_keyword_coverage(
    text: Union[str, List[str]],
    keywords: List[str],
    match_type: str = "exact"  # exact:精确匹配, fuzzy:模糊匹配
) -> Union[float, List[float]]:
    """
    关键词覆盖率评分（0-1）
    :param text: 待评估文本/文本列表
    :param keywords: 关键词列表
    :param match_type: 匹配类型（exact:精确，fuzzy:模糊）
    :return: 覆盖率/覆盖率列表
    """
    if not keywords:
        raise ValueError("关键词列表不能为空")
    
    def _coverage(t):
        t = t.lower()
        match_count = 0
        for kw in keywords:
            kw = kw.lower()
            if match_type == "exact" and kw in t:
                match_count += 1
            elif match_type == "fuzzy" and re.search(kw, t):
                match_count += 1
        return match_count / len(keywords)
    
    if isinstance(text, str):
        return _coverage(text)
    elif isinstance(text, list):
        return [_coverage(t) for t in text]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def calculate_data_completeness(
    data: pd.DataFrame,
    required_cols: List[str]
) -> float:
    """
    结构化数据完整性评分（0-1）
    :param data: 待评估DataFrame
    :param required_cols: 必选列列表
    :return: 完整性评分
    """
    if not required_cols:
        raise ValueError("必选列列表不能为空")
    # 检查必选列是否存在
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"缺失必选列：{missing_cols}")
    # 计算非空值比例
    completeness = data[required_cols].notna().mean().mean()
    return float(completeness)

def generate_quality_report(
    data: Union[pd.DataFrame, List[str]],
    col: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    required_cols: Optional[List[str]] = None,
    report_path: Optional[str] = None
) -> Dict:
    """
    生成数据质量评估报告
    :param data: 待评估数据
    :param col: 文本列名（DataFrame时必填）
    :param keywords: 关键词列表（用于覆盖率计算）
    :param required_cols: 必选列列表（DataFrame完整性计算）
    :param report_path: 报告保存路径
    :return: 质量报告字典
    """
    report = {}
    
    # 基础统计
    if isinstance(data, pd.DataFrame):
        report["total_rows"] = len(data)
        report["non_null_rows"] = len(data.dropna(how="all"))
        report["null_rate"] = 1 - report["non_null_rows"] / report["total_rows"]
        # 完整性评分
        if required_cols:
            report["completeness_score"] = calculate_data_completeness(data, required_cols)
        # 提取文本列表
        text_list = data[col].tolist() if col else []
    elif isinstance(data, list):
        report["total_texts"] = len(data)
        report["non_empty_texts"] = len([t for t in data if t.strip()])
        report["empty_rate"] = 1 - report["non_empty_texts"] / report["total_texts"]
        text_list = data
    else:
        raise TypeError("仅支持DataFrame或字符串列表类型")
    
    # 通顺度评分
    if text_list:
        smoothness_scores = calculate_text_smoothness(text_list)
        report["smoothness_avg"] = np.mean(smoothness_scores)
        report["smoothness_min"] = np.min(smoothness_scores)
        report["smoothness_max"] = np.max(smoothness_scores)
    
    # 重复度评分
    if text_list and len(text_list) >= 2:
        report["duplicate_rate"] = calculate_duplicate_rate(text_list)
    
    # 关键词覆盖率
    if text_list and keywords:
        coverage_scores = calculate_keyword_coverage(text_list, keywords)
        report["keyword_coverage_avg"] = np.mean(coverage_scores)
    
    # 整体质量分（加权计算）
    weights = {
        "smoothness_avg": 0.4,
        "duplicate_rate": -0.3,  # 重复度越高，质量分越低
        "completeness_score": 0.2,
        "keyword_coverage_avg": 0.1
    }
    total_score = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        if key in report:
            if key == "duplicate_rate":
                total_score += (1 - report[key]) * weight
            else:
                total_score += report[key] * weight
            weight_sum += abs(weight)
    report["overall_quality_score"] = total_score / weight_sum if weight_sum else 0.0
    
    # 保存报告
    if report_path:
        save_file(report, report_path)
    
    return report