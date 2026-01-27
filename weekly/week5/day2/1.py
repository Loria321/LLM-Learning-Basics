import os
import pandas as pd
import jsonlines
from typing import Dict, List, Tuple, Optional

# ====================== 1. 多格式数据读取模块 ======================
def read_structured_data(file_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    file_suffix = os.path.splitext(file_path)[-1].lower()
    
    try:
        if file_suffix == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        elif file_suffix == ".json":
            df = pd.read_json(file_path, encoding="utf-8")
        elif file_suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            print(f"错误：不支持的文件格式 {file_suffix}")
            return None
        print(f"成功读取 {file_path}，数据行数：{len(df)}")
        return df
    except Exception as e:
        print(f"读取文件失败：{str(e)}")
        return None

# ====================== 2. 数据清洗模块 ======================
def clean_ecommerce_data(df: pd.DataFrame) -> pd.DataFrame:
    field_mapping = {
        "商品名称": "product_name",
        "商品价格": "price",
        "商品库存": "stock",
        "商品销量": "sales",
        "商品分类": "category",
        "商品描述": "description"
    }
    df.rename(columns=field_mapping, inplace=True)
    
    df = df.drop_duplicates()
    required_fields = ["product_name", "price", "category"]
    df = df.dropna(subset=required_fields)
    
    df["stock"] = df["stock"].fillna(0)
    df["sales"] = df["sales"].fillna(0)
    df["description"] = df["description"].fillna("无描述")
    
    df = df[df["price"] >= 0]
    df = df[df["stock"] >= 0]
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce").fillna(0).astype(int)
    
    print(f"数据清洗完成，剩余数据行数：{len(df)}")
    return df

# ====================== 3. 数据校验模块 ======================
def validate_ecommerce_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    
    if not pd.api.types.is_string_dtype(df["product_name"]):
        errors.append("product_name 字段必须为字符串类型")
    if not pd.api.types.is_numeric_dtype(df["price"]):
        errors.append("price 字段必须为数值类型")
    if not pd.api.types.is_integer_dtype(df["stock"]):
        df["stock"] = df["stock"].astype(int)
    
    if df[df["price"] > 100000].shape[0] > 0:
        errors.append(f"发现 {df[df['price'] > 100000].shape[0]} 条价格异常（>10万）的数据")
    
    df["product_name"] = df["product_name"].str[:100]
    df["description"] = df["description"].str[:500]
    
    is_valid = len(errors) == 0
    if not is_valid:
        print("数据校验发现以下问题：")
        for err in errors:
            print(f"- {err}")
    else:
        print("数据校验通过")
    return is_valid, errors

# ====================== 4. JSONL输出模块（核心修复部分） ======================
def convert_to_finetune_jsonl(df: pd.DataFrame, output_path: str) -> bool:
    """
    修复：改用Python原生open指定编码，再传给jsonlines.Writer
    解决jsonlines.open不支持encoding参数的问题
    """
    finetune_data = []
    for _, row in df.iterrows():
        prompt = f"请介绍这款商品：{row['product_name']}（分类：{row['category']}）"
        completion = f"""
商品名称：{row['product_name']}
分类：{row['category']}
价格：{row['price']} 元
库存：{row['stock']} 件
销量：{row['sales']} 件
描述：{row['description']}
        """.strip()
        finetune_data.append({"prompt": prompt, "completion": completion})
    
    try:
        # 核心修改：先通过原生open打开文件（指定utf-8编码），再创建jsonlines.Writer
        with open(output_path, mode="w", encoding="utf-8") as f:
            writer = jsonlines.Writer(f)
            writer.write_all(finetune_data)
        print(f"成功输出JSONL文件：{output_path}，共 {len(finetune_data)} 条数据")
        return True
    except Exception as e:
        print(f"输出JSONL失败：{str(e)}")
        return False

# ====================== 5. 主流程函数 ======================
def ecommerce_data_clean_pipeline(input_file: str, output_file: str) -> bool:
    df = read_structured_data(input_file)
    if df is None:
        return False
    
    df_cleaned = clean_ecommerce_data(df)
    if df_cleaned.empty:
        print("错误：清洗后无数据")
        return False
    
    is_valid, _ = validate_ecommerce_data(df_cleaned)
    if not is_valid:
        print("警告：数据校验未通过，但仍尝试输出")
    
    return convert_to_finetune_jsonl(df_cleaned, output_file)

# ====================== 测试用例 ======================
if __name__ == "__main__":
    test_data = {
        "商品名称": ["华为Mate60 Pro", "苹果iPhone15", "小米14", None, "华为Mate60 Pro"],
        "商品价格": [6999, 5999, -3999, 4999, 6999],
        "商品库存": [1000, 2000, 1500, None, 1000],
        "商品销量": [5000, 8000, "6000", 7000, 5000],
        "商品分类": ["手机", "手机", "手机", "手机", "手机"],
        "商品描述": ["鸿蒙系统，卫星通话", None, "骁龙8Gen3，徕卡影像", "", "鸿蒙系统，卫星通话"]
    }
    test_csv_path = "test_ecommerce_data.csv"
    pd.DataFrame(test_data).to_csv(test_csv_path, index=False, encoding="utf-8")
    
    output_jsonl_path = "finetune_ecommerce_data.jsonl"
    result = ecommerce_data_clean_pipeline(test_csv_path, output_jsonl_path)
    
    if result:
        print("\n测试完成！输出文件：", output_jsonl_path)
        # 验证输出结果
        with open(output_jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 2:
                    print(f"\n第{i+1}条微调数据：{line.strip()}")
    else:
        print("测试失败")