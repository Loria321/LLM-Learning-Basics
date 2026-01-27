import pandas as pd
import json
import os

def excel_to_csv(excel_path, csv_path, sheet_name=0, encoding="utf-8"):
    """
    将Excel文件转换为CSV文件
    :param excel_path: 输入Excel文件路径（支持.xlsx/.xls）
    :param csv_path: 输出CSV文件路径
    :param sheet_name: 读取Excel的工作表（默认第1个）
    :param encoding: 文件编码（默认utf-8，避免中文乱码）
    :return: 转换成功返回True，失败返回False
    """
    try:
        # 检查Excel文件是否存在
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel文件不存在：{excel_path}")
        
        # 读取Excel文件
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # 保存为CSV文件（index=False不保存行索引，避免冗余）
        df.to_csv(csv_path, index=False, encoding=encoding)
        print(f"Excel转CSV成功！输出路径：{csv_path}")
        return True
    
    except Exception as e:
        print(f"Excel转CSV失败：{str(e)}")
        return False

def csv_to_jsonl(csv_path, jsonl_path, encoding="utf-8"):
    """
    将CSV文件转换为大模型微调专用的JSONL文件（字段固定为question/answer）
    :param csv_path: 输入CSV文件路径（必须包含question和answer列）
    :param jsonl_path: 输出JSONL文件路径
    :param encoding: 文件编码（默认utf-8）
    :return: 转换成功返回True，失败返回False
    """
    try:
        # 检查CSV文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在：{csv_path}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_path, encoding=encoding)
        
        # 检查必要列是否存在
        required_columns = ["question", "answer"]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"CSV文件缺少必要列：{missing_cols}，必须包含question和answer列")
        
        # 逐行转换为JSON并写入JSONL文件
        with open(jsonl_path, "w", encoding=encoding) as f:
            for _, row in df.iterrows():
                # 构造指定格式的JSON对象
                json_obj = {
                    "question": str(row["question"]).strip(),  # 去空格，确保字符串类型
                    "answer": str(row["answer"]).strip()
                }
                # 写入一行JSON（json.dumps确保格式合法）
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        
        print(f"CSV转JSONL成功！输出路径：{jsonl_path}")
        return True
    
    except Exception as e:
        print(f"CSV转JSONL失败：{str(e)}")
        return False

# 测试示例（取消注释即可运行）
if __name__ == "__main__":
    # 1. Excel转CSV（替换为你的文件路径）
    excel_to_csv(
        excel_path="test_data.xlsx",  # 输入Excel文件
        csv_path="test_data.csv"      # 输出CSV文件
    )
    
    # 2. CSV转JSONL（替换为你的文件路径）
    csv_to_jsonl(
        csv_path="test_data.csv",     # 输入CSV文件
        jsonl_path="test_data.jsonl"  # 输出JSONL文件
    )