import jieba
import opencc
import re
import pandas as pd
from typing import List, Union, Optional
from .utils import read_file, save_file

# 加载停用词（可替换为自定义停用词路径）
def load_stopwords(stopword_path: Optional[str] = None) -> set:
    """
    加载停用词表
    :param stopword_path: 停用词文件路径（每行一个停用词）
    :return: 停用词集合
    """
    if stopword_path is None:
        # 默认停用词（精简版）
        return {"的", "了", "是", "我", "你", "他", "她", "它", "们", "在", "有", "就", "不", "和", "也", "都"}
    with open(stopword_path, "r", encoding="utf-8") as f:
        return set([line.strip() for line in f if line.strip()])

def split_text(
    text: Union[str, List[str]],
    cut_all: bool = False,
    use_jieba: bool = True
) -> Union[List[str], List[List[str]]]:
    """
    文本分词
    :param text: 待分词文本/文本列表
    :param cut_all: jieba全模式（True/False）
    :param use_jieba: 是否使用jieba分词（False则按空格分割）
    :return: 分词结果/分词结果列表
    """
    def _cut(t):
        if use_jieba:
            return jieba.lcut(t, cut_all=cut_all)
        else:
            return t.split()
    
    if isinstance(text, str):
        return _cut(text)
    elif isinstance(text, list):
        return [_cut(t) for t in text]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def remove_stopwords(
    words: Union[List[str], List[List[str]]],
    stopwords: Optional[set] = None,
    stopword_path: Optional[str] = None
) -> Union[List[str], List[List[str]]]:
    """
    去除停用词
    :param words: 分词结果/分词结果列表
    :param stopwords: 停用词集合（优先级高于stopword_path）
    :param stopword_path: 停用词文件路径
    :return: 去停用词后结果
    """
    stopwords = stopwords or load_stopwords(stopword_path)
    
    def _remove(ws):
        return [w for w in ws if w not in stopwords and w.strip()]
    
    if isinstance(words[0], str):
        return _remove(words)
    elif isinstance(words[0], list):
        return [_remove(ws) for ws in words]
    else:
        raise TypeError("仅支持分词列表或分词列表的列表")

def filter_sensitive_words(
    text: Union[str, List[str]],
    sensitive_words: Optional[set] = None,
    sensitive_path: Optional[str] = None,
    replace_char: str = "*"
) -> Union[str, List[str]]:
    """
    敏感词过滤
    :param text: 待处理文本/文本列表
    :param sensitive_words: 敏感词集合（优先级高于sensitive_path）
    :param sensitive_path: 敏感词文件路径（每行一个）
    :param replace_char: 替换字符
    :return: 过滤后文本/文本列表
    """
    if sensitive_words is None:
        if sensitive_path:
            with open(sensitive_path, "r", encoding="utf-8") as f:
                sensitive_words = set([line.strip() for line in f if line.strip()])
        else:
            sensitive_words = set()  # 空集合，不过滤
    
    def _filter(t):
        for word in sensitive_words:
            if word in t:
                t = t.replace(word, replace_char * len(word))
        return t
    
    if isinstance(text, str):
        return _filter(text)
    elif isinstance(text, list):
        return [_filter(t) for t in text]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def filter_text_length(
    text: Union[str, List[str]],
    min_len: int = 1,
    max_len: Optional[int] = None
) -> Union[str, List[str]]:
    """
    文本长度过滤
    :param text: 待处理文本/文本列表
    :param min_len: 最小长度
    :param max_len: 最大长度（None则不限制）
    :return: 过滤后文本/文本列表
    """
    def _filter(t):
        length = len(t.strip())
        if length < min_len:
            return ""
        if max_len and length > max_len:
            return t.strip()[:max_len]
        return t.strip()
    
    if isinstance(text, str):
        return _filter(text)
    elif isinstance(text, list):
        return [_filter(t) for t in text if _filter(t)]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def convert_traditional_simplified(
    text: Union[str, List[str]],
    convert_type: str = "s2t"  # s2t:简转繁, t2s:繁转简
) -> Union[str, List[str]]:
    """
    繁简转换
    :param text: 待处理文本/文本列表
    :param convert_type: 转换类型（s2t:简转繁，t2s:繁转简）
    :return: 转换后文本/文本列表
    """
    converter = opencc.OpenCC(convert_type)
    
    def _convert(t):
        return converter.convert(t)
    
    if isinstance(text, str):
        return _convert(text)
    elif isinstance(text, list):
        return [_convert(t) for t in text]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def text_special_pipeline(
    data: Union[pd.DataFrame, List[str]],
    col: Optional[str] = None,
    do_split: bool = False,
    remove_stopwords_flag: bool = False,
    filter_sensitive: bool = False,
    filter_length: bool = True,
    min_len: int = 5,
    max_len: int = 1000,
    convert_t2s: bool = False,
    sensitive_path: Optional[str] = r"config\sensitive_words.txt",
    stopword_path: Optional[str] = r"config\stopwords.txt"
) -> Union[pd.DataFrame, List[str], List[List[str]]]:
    """
    文本专项处理流水线
    :param data: 待处理数据
    :param col: DataFrame处理列名
    :param do_split: 是否分词
    :param remove_stopwords_flag: 是否去停用词
    :param filter_sensitive: 是否过滤敏感词
    :param filter_length: 是否过滤长度
    :param min_len: 最小长度
    :param max_len: 最大长度
    :param convert_t2s: 是否繁转简
    :param sensitive_path: 敏感词路径
    :param stopword_path: 停用词路径
    :return: 处理后数据
    """
    # 统一格式为列表
    if isinstance(data, pd.DataFrame) and col:
        text_list = data[col].tolist()
    elif isinstance(data, list):
        text_list = data
    else:
        raise TypeError("仅支持DataFrame或列表类型")
    
    # 繁简转换
    if convert_t2s:
        text_list = convert_traditional_simplified(text_list, "t2s")
    
    # 敏感词过滤
    if filter_sensitive:
        text_list = filter_sensitive_words(text_list, sensitive_path=sensitive_path)
    
    # 长度过滤
    if filter_length:
        text_list = filter_text_length(text_list, min_len, max_len)
    
    # 分词
    if do_split:
        text_list = split_text(text_list)
        # 去停用词
        if remove_stopwords_flag:
            text_list = remove_stopwords(text_list, stopword_path=stopword_path)
    
    # 回填到DataFrame
    if isinstance(data, pd.DataFrame) and col:
        data[col] = text_list
        return data
    return text_list