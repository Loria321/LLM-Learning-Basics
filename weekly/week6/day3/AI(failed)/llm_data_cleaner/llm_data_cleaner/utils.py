import os
import json
import pandas as pd
from typing import List, Dict, Union, Optional

def read_file(file_path: str, encoding: str = "utf-8") -> Union[str, pd.DataFrame]:
    """
    通用文件读取函数，支持txt/csv/json/xlsx格式
    :param file_path: 文件路径
    :param encoding: 编码格式，默认utf-8
    :return: 读取结果（文本/DataFrame）
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            return f.read()
    elif ext == ".csv":
        return pd.read_csv(file_path, encoding=encoding)
    elif ext == ".json":
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式：{ext}，仅支持txt/csv/json/xlsx")

def save_file(
    data: Union[str, pd.DataFrame, Dict],
    save_path: str,
    encoding: str = "utf-8",
    indent: int = 4
) -> None:
    """
    通用文件保存函数，支持txt/csv/json/xlsx格式
    :param data: 待保存数据
    :param save_path: 保存路径
    :param encoding: 编码格式
    :param indent: json缩进
    """
    ext = os.path.splitext(save_path)[1].lower()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if ext == ".txt":
        with open(save_path, "w", encoding=encoding) as f:
            f.write(str(data))
    elif ext == ".csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(save_path, index=False, encoding=encoding)
        else:
            raise TypeError("csv格式仅支持DataFrame数据")
    elif ext == ".json":
        with open(save_path, "w", encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    elif ext in [".xlsx", ".xls"]:
        if isinstance(data, pd.DataFrame):
            data.to_excel(save_path, index=False)
        else:
            raise TypeError("xlsx格式仅支持DataFrame数据")
    else:
        raise ValueError(f"不支持的文件格式：{ext}，仅支持txt/csv/json/xlsx")

def get_file_list(folder_path: str, ext_list: Optional[List[str]] = None) -> List[str]:
    """
    获取指定文件夹下的指定格式文件列表
    :param folder_path: 文件夹路径
    :param ext_list: 扩展名列表，如[".txt", ".csv"]，None则返回所有文件
    :return: 文件路径列表
    """
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if ext_list is None or os.path.splitext(file)[1].lower() in ext_list:
                file_list.append(os.path.join(root, file))
    return file_list