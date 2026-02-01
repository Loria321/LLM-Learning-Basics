import json
import pandas as pd
import re
from typing import List, Dict, Union, Optional
from .utils import read_file, save_file

def parse_json_str(
    json_str: Union[str, List[str]],
    default_val: Optional[Dict] = None
) -> Union[Dict, List[Dict]]:
    """
    JSON字符串解析为结构化字典
    :param json_str: JSON字符串/字符串列表
    :param default_val: 解析失败时的默认值
    :return: 解析后的字典/字典列表
    """
    default_val = default_val or {}
    
    def _parse(s):
        try:
            return json.loads(s.strip())
        except (json.JSONDecodeError, AttributeError):
            return default_val
    
    if isinstance(json_str, str):
        return _parse(json_str)
    elif isinstance(json_str, list):
        return [_parse(s) for s in json_str]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def extract_structured_data(
    text: Union[str, List[str]],
    pattern: str,
    group_names: List[str],
    default_val: str = ""
) -> Union[Dict, List[Dict]]:
    """
    正则提取文本中的结构化数据
    :param text: 待提取文本/文本列表
    :param pattern: 正则表达式（含命名分组）
    :param group_names: 分组名列表（与正则分组对应）
    :param default_val: 提取失败时的默认值
    :return: 结构化字典/字典列表
    """
    regex = re.compile(pattern, re.S)
    
    def _extract(t):
        match = regex.search(t)
        if match:
            return {name: match.group(name) or default_val for name in group_names}
        else:
            return {name: default_val for name in group_names}
    
    if isinstance(text, str):
        return _extract(text)
    elif isinstance(text, list):
        return [_extract(t) for t in text]
    else:
        raise TypeError("仅支持字符串或字符串列表类型")

def struct_data_validate(
    struct_data: Union[Dict, List[Dict]],
    required_fields: List[str],
    field_rules: Optional[Dict[str, callable]] = None
) -> Union[bool, List[bool]]:
    """
    结构化数据校验
    :param struct_data: 结构化字典/字典列表
    :param required_fields: 必选字段列表
    :param field_rules: 字段校验规则（{字段名: 校验函数}）
    :return: 校验结果（True/False）/结果列表
    """
    field_rules = field_rules or {}
    
    def _validate(d):
        # 校验必选字段
        for field in required_fields:
            if field not in d or d[field] is None or d[field] == "":
                return False
        # 校验字段规则
        for field, rule in field_rules.items():
            if field in d and not rule(d[field]):
                return False
        return True
    
    if isinstance(struct_data, dict):
        return _validate(struct_data)
    elif isinstance(struct_data, list):
        return [_validate(d) for d in struct_data]
    else:
        raise TypeError("仅支持字典或字典列表类型")

def text_to_struct(
    text: Union[str, List[str]],
    struct_type: str = "json",  # json/regex
    **kwargs
) -> Union[Dict, List[Dict]]:
    """
    文本转结构化数据统一接口
    :param text: 待处理文本/文本列表
    :param struct_type: 转换类型（json/regex）
    :param kwargs: 其他参数（对应parse_json_str/extract_structured_data的参数）
    :return: 结构化数据
    """
    if struct_type == "json":
        return parse_json_str(text, **kwargs)
    elif struct_type == "regex":
        return extract_structured_data(text, **kwargs)
    else:
        raise ValueError("struct_type仅支持json/regex")

def struct_process_pipeline(
    data: Union[pd.DataFrame, List[str]],
    col: Optional[str] = None,
    struct_type: str = "json",
    validate: bool = True,
    required_fields: Optional[List[str]] = None,
    **kwargs
) -> Union[pd.DataFrame, List[Dict]]:
    """
    结构化处理流水线
    :param data: 待处理数据
    :param col: DataFrame处理列名
    :param struct_type: 转换类型（json/regex）
    :param validate: 是否校验结构化数据
    :param required_fields: 必选字段（校验用）
    :param kwargs: 文本转结构化的参数
    :return: 处理后数据（含结构化列/结构化列表）
    """
    # 统一格式为列表
    if isinstance(data, pd.DataFrame) and col:
        text_list = data[col].tolist()
    elif isinstance(data, list):
        text_list = data
    else:
        raise TypeError("仅支持DataFrame或列表类型")
    
    # 文本转结构化
    struct_list = text_to_struct(text_list, struct_type=struct_type, **kwargs)
    
    # 结构化校验
    if validate and required_fields:
        validate_result = struct_data_validate(struct_list, required_fields)
        # 过滤校验失败的数据
        struct_list = [s for s, v in zip(struct_list, validate_result) if v]
        text_list = [t for t, v in zip(text_list, validate_result) if v]
    
    # 回填到DataFrame
    if isinstance(data, pd.DataFrame) and col:
        # 过滤原DataFrame
        data = data.iloc[[i for i, v in enumerate(validate_result) if v]].reset_index(drop=True)
        # 添加结构化列
        data["structured_data"] = struct_list
        return data
    return struct_list