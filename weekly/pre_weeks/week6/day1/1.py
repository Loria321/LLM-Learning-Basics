import os
import pandas as pd

#1.文件夹遍历
def traverse_folder(root_folder):
    """
    traverse_folder 的 Docstring
    递归遍历指定根文件夹下的所有文件，返回完整文件路径列表
    :param root_folder: 根文件夹路径
    """
    file_path_list = []

    if not os.path.exists(root_folder):
        print(f"错误：文件夹 {root_folder} 不存在！")
        return file_path_list
    
    for root,dirs,files in os.walk(root_folder):
        for file_name in files:
            full_file_path = os.path.join(root,file_name)
            file_path_list.append(full_file_path)

    return file_path_list

#2.格式识别与对应清洗逻辑
def get_file_format(file_path):
    """
    get_file_format 的 Docstring
    提取文件后缀名，判断格式
    :param file_path: 说明
    """
    #分离文件名和后缀名，获取后缀并转为小写
    _,file_ext = os.path.splitext(file_path)
    file_ext = file_ext.lower()

    if file_ext == ".csv":
        return "csv"
    elif file_ext == ".json":
        return "json"
    elif file_ext in [".xlsx",".xls"]:
        return "excel"
    else:
        return "unknown"

def clean_csv(file_path):
    """
    clean_csv 的 Docstring
    CSV 文件读取与基础清洗
    :param file_path: 说明
    """
    try:
        #读取 CSV 文件
        df = pd.read_csv(file_path,encoding="utf-8-sig")    # utf-8-sig 兼容含 BOM 的 UTF-8 文件
    except Exception as e:
        print(f"错误：读取 CSV 文件 {file_path} 失败 - {str(e)}")
        return None
    
    #基础清洗逻辑
    cleaned_df = df.copy()
    # 1. 去除空数据
    cleaned_df = cleaned_df.dropna(how="all",axis=0).dropna(how="all",axis=1)
    # 2. 去重
    cleaned_df = cleaned_df.drop_duplicates()
    # 3. 重置索引
    cleaned_df = cleaned_df.reset_index(drop=True)

    print(f"成功清洗 CSV 文件： {file_path} ,原始行数： {len(df)} ,清洗后行数： {len(cleaned_df)}")
    return cleaned_df

def clean_json(file_path):
    """
    clean_json 的 Docstring
    JSON 文件的读取与基础清洗
    :param file_path: 说明
    """
    try:
        # 读取 JSON 文件(此处使用常规格式读取)
        df = pd.read_json(file_path,encoding="utf-8-sig")
    except Exception as e:
        print(f"错误：读取 JSON 文件 {file_path} 失败 -  {str(e)}")
        return None
    
    #基础清洗逻辑
    cleaned_df = df.copy()
    # 1. 去除空数据
    cleaned_df = cleaned_df.dropna(how="all",axis=0).dropna(how="all",axis=1)
    # 2. 去重
    cleaned_df = cleaned_df.drop_duplicates()
    # 3. 重置索引
    cleaned_df = cleaned_df.reset_index(drop=True)

    print(f"成功清洗 JSON 文件： {file_path} ,原始行数： {len(df)} ,清洗后行数： {len(cleaned_df)}")
    return cleaned_df

def clean_excel(file_path):
    try:
        # 读取 Excel 文件(默认读取第一个工作表)
        df = pd.read_excel(file_path, engine="openpyxl" if file_path.endswith(".xlsx") else "xlrd")
    except Exception as e:
        print(f"错误：读取 Excel 文件 {file_path} 失败 -  {str(e)}")
        return None
    
    #基础清洗逻辑
    cleaned_df = df.copy()
    # 1. 去除空数据
    cleaned_df = cleaned_df.dropna(how="all",axis=0).dropna(how="all",axis=1)
    # 2. 去重
    cleaned_df = cleaned_df.drop_duplicates()
    # 3. 重置索引
    cleaned_df = cleaned_df.reset_index(drop=True)

    print(f"成功清洗 Excel 文件： {file_path} ,原始行数： {len(df)} ,清洗后行数： {len(cleaned_df)}")
    return cleaned_df

# 2. 批量处理主函数
def batch_process_files(root_folder,output_folder="cleaned_output"):
    """
    batch_process_files 的 Docstring
    批量处理指定文件夹下所有 CSV/JSON/Excel 文件
    :param root_folder: 说明
    :param output_folder: 说明
    """
    # 1. 遍历文件夹，获取所有文件路径
    file_list = traverse_folder(root_folder)
    if not file_list:
        print("未找到任何文件，批量处理终止！")
        return
    
    # 2. 创建输出文件夹（如果不存在）
    os.makedirs(output_folder,exist_ok=True)

    # 3. 遍历所有文件，逐个处理
    for file_path in file_list:
        # 3.1 识别文件格式
        file_format = get_file_format(file_path)
        file_name = os.path.basename(file_path)
        cleaned_file_name = f"cleaned_{file_name}"
        cleaned_file_path = os.path.join(output_folder,cleaned_file_name)

        # 3.2 调用对应清洗函数
        cleaned_df = None
        if file_format == "csv":
            cleaned_df = clean_csv(file_path)
            #保存
            if cleaned_df is not None:
                cleaned_df.to_csv(cleaned_file_path,index=False,encoding="utf-8-sig")
        elif file_format == "json":
            cleaned_df = clean_json(file_path)
            if cleaned_df is not None:
                cleaned_df.to_json(cleaned_file_path,orient="records",force_ascii=False,indent=2)    
        elif file_format == "excel":
            cleaned_df = clean_excel(file_path)
            # 统一保存为 .xlsx 格式
            if cleaned_df is not None:
                # 提取文件名（不含后缀）+ 拼接 .xlsx
                file_name_no_ext = os.path.splitext(cleaned_file_name)[0]
                final_excel_path = os.path.join(output_folder, f"{file_name_no_ext}.xlsx")
                cleaned_df.to_excel(final_excel_path, index=False, engine="openpyxl")        
        else:
            print(f"跳过不支持的文件格式： {file_path}")
            continue
        
        # 3.3 提示保存成功
        if cleaned_df is not None:
            print(f"清洗后文件已保存： {cleaned_file_path}\n")

#测试文件生成
def generate_test_files():
    """自动生成 10 个不同格式的测试数据集"""
    test_folder = "test_datasets"
    os.makedirs(test_folder, exist_ok=True)
    
    # 生成基础测试数据（包含重复行、空行）
    import numpy as np
    base_data = {
        "id": [1, 2, 3, 3, np.nan, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Charlie", "", "David", "Eve"],
        "value": [10.5, 20.3, 15.7, 15.7, np.nan, 25.1, 30.0]
    }
    df = pd.DataFrame(base_data)
    
    # 1. 生成 3 个 CSV 文件
    for i in range(1, 4):
        csv_path = os.path.join(test_folder, f"test_data_{i}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 2. 生成 3 个 JSON 文件
    for i in range(1, 4):
        json_path = os.path.join(test_folder, f"test_data_{i}.json")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    
    # 3. 生成 4 个 Excel 文件（4 个 .xlsx）
    for i in range(1, 5):
        xlsx_path = os.path.join(test_folder, f"test_data_{i}.xlsx")
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
       
    print("成功生成 10 个测试文件（3 CSV + 3 JSON + 4 Excel）！")


# 执行批量处理
if __name__ == "__main__":
    # 先生成测试文件
    generate_test_files()
    
    # 批量处理 test_datasets 文件夹下的所有文件
    batch_process_files("test_datasets")