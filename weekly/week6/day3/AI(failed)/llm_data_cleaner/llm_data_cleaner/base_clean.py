import re
import pandas as pd
import numpy as np
from typing import List, Union, Optional
from .utils import read_file, save_file

def remove_duplicates(
    data: Union[pd.DataFrame, List[str]],
    col: Optional[str] = None
) -> Union[pd.DataFrame, List[str]]:
    """
    去重处理
    :param data: 待处理数据（DataFrame/文本列表）
    :param col: DataFrame去重的列名，None则按全列去重
    :return: 去重后数据
    """
    if isinstance(data, pd.DataFrame):
        return data.drop_duplicates(subset=col, keep="first").reset_index(drop=True)
    elif isinstance(data, list):
        return list(dict.fromkeys(data))  # 保留顺序去重
    else:
        raise TypeError("仅支持DataFrame或列表类型数据")

def remove_null_values(
    data: Union[pd.DataFrame, List[str]],
    col: Optional[str] = None
) -> Union[pd.DataFrame, List[str]]:
    """
    去除空值
    :param data: 待处理数据（DataFrame/文本列表）
    :param col: DataFrame处理的列名
    :return: 去空值后数据
    """
    if isinstance(data, pd.DataFrame):
        if col:
            return data[data[col].notna() & (data[col] != "")].reset_index(drop=True)
        else:
            return data.dropna(how="all").reset_index(drop=True)
    elif isinstance(data, list):
        return [x for x in data if x is not None and x != ""]
    else:
        raise TypeError("仅支持DataFrame或列表类型数据")

def remove_special_chars(
    text: Union[str, List[str]],
    keep_chars: str = "，。！？；：""''（）【】《》a-zA-Z0-9\u4e00-\u9fa5",
    replace_char: str = ""
) -> Union[str, List[str]]:
    """
    去除特殊字符，仅保留指定字符
    :param text: 待处理文本/文本列表
    :param keep_chars: 保留的字符集
    :param replace_char: 替换字符，默认空字符串
    :return: 清洗后文本/文本列表
    """
    pattern = f"[^{keep_chars}]"
    if isinstance(text, str):
        return re.sub(pattern, replace_char, text)
    elif isinstance(text, list):
        return [re.sub(pattern, replace_char, t) for t in text]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def convert_case(
    text: Union[str, List[str]],
    case_type: str = "lower"  # lower/upper/title
) -> Union[str, List[str]]:
    """
    大小写转换
    :param text: 待处理文本/文本列表
    :param case_type: 转换类型（lower:小写，upper:大写，title:首字母大写）
    :return: 转换后文本/文本列表
    """
    def _convert(t):
        if case_type == "lower":
            return t.lower()
        elif case_type == "upper":
            return t.upper()
        elif case_type == "title":
            return t.title()
        else:
            raise ValueError("case_type仅支持lower/upper/title")
    
    if isinstance(text, str):
        return _convert(text)
    elif isinstance(text, list):
        return [_convert(t) for t in text]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def base_clean_pipeline(
    data: Union[pd.DataFrame, List[str]],
    col: Optional[str] = None,
    remove_dup: bool = True,
    remove_null: bool = True,
    remove_spec_chars: bool = True,
    case_convert: Optional[str] = None,
    keep_chars: str = "，。！？；：""''（）【】《》a-zA-Z0-9\u4e00-\u9fa5"
) -> Union[pd.DataFrame, List[str]]:
    """
    基础清洗流水线
    :param data: 待处理数据
    :param col: DataFrame处理列名
    :param remove_dup: 是否去重
    :param remove_null: 是否去空值
    :param remove_spec_chars: 是否去特殊字符
    :param case_convert: 大小写转换类型（None则不转换）
    :param keep_chars: 保留字符集
    :return: 清洗后数据
    """
    # 去空值
    if remove_null:
        data = remove_null_values(data, col)
    # 去特殊字符
    if remove_spec_chars:
        if isinstance(data, pd.DataFrame) and col:
            data[col] = remove_special_chars(data[col].tolist(), keep_chars)
        elif isinstance(data, list):
            data = remove_special_chars(data, keep_chars)
    # 大小写转换
    if case_convert:
        if isinstance(data, pd.DataFrame) and col:
            data[col] = convert_case(data[col].tolist(), case_convert)
        elif isinstance(data, list):
            data = convert_case(data, case_convert)
    # 去重
    if remove_dup:
        data = remove_duplicates(data, col)
    return data