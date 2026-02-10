import pandas as pd
from typing import List, Union, Optional

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

def base_clean_pipeline(
    data: Union[pd.DataFrame, List[str]],
    col: Optional[str] = None,
    remove_dup: bool = True,
    remove_null: bool = True,
) -> Union[pd.DataFrame, List[str]]:
    """
    基础清洗流水线
    :param data: 待处理数据
    :param col: DataFrame处理的列名
    :param remove_dup: 是否去重
    :param remove_null: 是否去空值
    :return: 清洗后数据
    """
    # 去空值
    if remove_null:
        data = remove_null_values(data, col)
    # 去重
    if remove_dup:
        data = remove_duplicates(data, col)
    return data