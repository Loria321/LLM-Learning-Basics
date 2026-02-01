import os
import multiprocessing
from typing import List, Callable, Union, Optional
import pandas as pd
from .utils import read_file, save_file, get_file_list
from .base_clean import base_clean_pipeline
from .text_special import text_special_pipeline
from .struct_process import struct_process_pipeline

def process_single_file(
    file_path: str,
    process_func: Callable,
    save_path: Optional[str] = None,
    **kwargs
) -> Union[str, pd.DataFrame, List]:
    """
    处理单个文件
    :param file_path: 输入文件路径
    :param process_func: 处理函数（base_clean_pipeline/text_special_pipeline/struct_process_pipeline）
    :param save_path: 保存路径（None则不保存）
    :param kwargs: 处理函数的参数
    :return: 处理后数据
    """
    # 读取文件
    data = read_file(file_path)
    # 处理数据
    if isinstance(data, pd.DataFrame):
        col = kwargs.get("col")
        if col and col not in data.columns:
            raise ValueError(f"文件{file_path}中不存在列：{col}")
        processed_data = process_func(data, **kwargs)
    elif isinstance(data, str):
        processed_data = process_func([data], **kwargs)[0]
    else:
        processed_data = process_func(data, **kwargs)
    # 保存文件
    if save_path:
        save_file(processed_data, save_path)
    return processed_data

def batch_process_files(
    input_folder: str,
    output_folder: str,
    process_func: Callable,
    ext_list: List[str] = [".txt", ".csv", ".json"],
    num_workers: int = 4,
    **kwargs
) -> List[Union[str, pd.DataFrame, List]]:
    """
    批量处理文件夹下的文件（多进程）
    :param input_folder: 输入文件夹
    :param output_folder: 输出文件夹
    :param process_func: 处理函数
    :param ext_list: 处理的文件扩展名
    :param num_workers: 进程数
    :param kwargs: 处理函数的参数
    :return: 所有文件的处理结果列表
    """
    # 获取文件列表
    file_list = get_file_list(input_folder, ext_list)
    if not file_list:
        raise ValueError(f"文件夹{input_folder}下无{ext_list}格式文件")
    
    # 构建参数列表
    args_list = []
    for file_path in file_list:
        # 构建输出路径
        rel_path = os.path.relpath(file_path, input_folder)
        save_path = os.path.join(output_folder, rel_path)
        args_list.append((file_path, process_func, save_path, kwargs))
    
    # 多进程处理
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(process_single_file, args_list)
    
    return results

def batch_summary_report(
    input_folder: str,
    output_folder: str,
    report_path: str = "batch_summary.json"
) -> Dict:
    """
    批量处理结果汇总报告
    :param input_folder: 输入文件夹
    :param output_folder: 输出文件夹
    :param report_path: 报告保存路径
    :return: 汇总报告字典
    """
    input_files = get_file_list(input_folder)
    output_files = get_file_list(output_folder)
    
    report = {
        "input_file_count": len(input_files),
        "output_file_count": len(output_files),
        "success_rate": len(output_files)/len(input_files) if input_files else 0,
        "input_files": input_files,
        "output_files": output_files
    }
    
    save_file(report, os.path.join(output_folder, report_path))
    return report