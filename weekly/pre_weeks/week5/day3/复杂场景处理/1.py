import os
import pandas as pd
import jsonlines
from typing import Dict, List, Tuple, Optional

# ====================== 1. 字段配置模块（修正：补充全量映射关系） ======================
FIELD_CONFIG = {
    "duplicate_field_groups": {
        "product_name": ["商品名称", "产品名", "货品名称"],
        "product_price": ["商品价格", "产品定价", "售价"]
    },
    # 修正：key为【合并后/原始】所有需要映射的字段，value为最终标准化字段（白名单字段）
    "field_mapping": {
        "product_name": "product_name",  # 合并后的名称字段
        "product_price": "price",        # 合并后的价格字段→映射为price（白名单）
        "商品分类": "category",           # 原始分类字段→映射为category（白名单）
        "商品描述": "description"         # 原始描述字段→映射为description（白名单）
    },
    # 白名单：最终保留的标准化字段（无需修改）
    "keep_fields_whitelist": ["product_name", "price", "category", "description"]
}

# ====================== 2. 重复字段合并模块（无修改） ======================
def merge_duplicate_fields(df: pd.DataFrame, duplicate_config: Dict[str, List[str]]) -> pd.DataFrame:
    for target_field, duplicate_fields in duplicate_config.items():
        existing_duplicate_fields = [f for f in duplicate_fields if f in df.columns]
        if not existing_duplicate_fields:
            continue
        
        df[target_field] = None
        for field in existing_duplicate_fields:
            df[target_field] = df[target_field].fillna(df[field])
        
        df = df.drop(columns=existing_duplicate_fields)
    
    print(f"重复字段合并完成，当前字段列表：{list(df.columns)}")
    return df

# ====================== 3. 字段标准化映射模块（独立抽离，提前执行） ======================
def standardize_field_name(df: pd.DataFrame, field_mapping: Dict[str, str]) -> pd.DataFrame:
    """独立的字段映射函数，提前到冗余删除前执行，避免原始/合并字段被误删"""
    # 筛选数据中实际存在的、需要映射的字段
    existing_mappable_fields = [f for f in field_mapping.keys() if f in df.columns]
    if existing_mappable_fields:
        df.rename(columns={f: field_mapping[f] for f in existing_mappable_fields}, inplace=True)
        print(f"字段名标准化完成，映射后字段列表：{list(df.columns)}")
    else:
        print("无需要标准化的字段")
    return df

# ====================== 4. 冗余字段删除模块（无修改，基于映射后的白名单） ======================
def delete_redundant_fields(df: pd.DataFrame, keep_whitelist: List[str]) -> pd.DataFrame:
    redundant_fields = [col for col in df.columns if col not in keep_whitelist]
    if redundant_fields:
        df = df.drop(columns=redundant_fields)
        print(f"冗余字段删除完成，删除的字段：{redundant_fields}")
    else:
        print("无冗余字段需要删除")
    
    # 白名单字段缺失填充空值（仅兜底，正常流程不会触发）
    for keep_field in keep_whitelist:
        if keep_field not in df.columns:
            df[keep_field] = None
            print(f"警告：白名单字段 {keep_field} 不存在，已填充为空值")
    
    print(f"冗余字段处理完成，最终保留字段：{list(df.columns)}")
    return df

# ====================== 5. 多格式数据读取模块（无修改） ======================
def read_structured_data(file_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    file_suffix = os.path.splitext(file_path)[-1].lower()
    
    try:
        if file_suffix == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        elif file_suffix == ".json":
            df = df.read_json(file_path, encoding="utf-8")
        elif file_suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            print(f"错误：不支持的文件格式 {file_suffix}")
            return None
        print(f"成功读取 {file_path}，数据行数：{len(df)}，原始字段：{list(df.columns)}")
        return df
    except Exception as e:
        print(f"读取文件失败：{str(e)}")
        return None

# ====================== 6. 数据清洗模块（无修改，基于最终标准化字段） ======================
def clean_ecommerce_data(df: pd.DataFrame) -> pd.DataFrame:
    keep_fields = FIELD_CONFIG["keep_fields_whitelist"]
    
    # 步骤1：删除完全重复行
    df = df.drop_duplicates()
    
    # 步骤2：缺失值处理（仅处理存在的必填字段）
    required_fields = ["product_name", "price", "category"]
    existing_required_fields = [f for f in required_fields if f in df.columns]
    df = df.dropna(subset=existing_required_fields)
    
    # 非必填字段缺失填充（仅处理存在的字段）
    if "description" in df.columns:
        df["description"] = df["description"].fillna("无描述")
    
    # 步骤3：异常值过滤（仅处理存在的数值字段）
    if "price" in df.columns:
        df = df[df["price"] >= 0]
    
    # 步骤4：字段长度校验（仅处理存在的字段）
    if "product_name" in df.columns:
        df["product_name"] = df["product_name"].str[:100]
    if "description" in df.columns:
        df["description"] = df["description"].str[:500]
    
    print(f"数据清洗完成，剩余数据行数：{len(df)}")
    return df

# ====================== 7. 数据校验模块（无修改） ======================
def validate_ecommerce_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    keep_fields = FIELD_CONFIG["keep_fields_whitelist"]
    
    # 字段类型校验（先判断字段是否存在）
    if "product_name" in keep_fields and "product_name" in df.columns:
        if not pd.api.types.is_string_dtype(df["product_name"]):
            errors.append("product_name 字段必须为字符串类型")
    if "price" in keep_fields and "price" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["price"]):
            errors.append("price 字段必须为数值类型")
    
    # 业务规则校验（先判断字段是否存在）
    if "price" in df.columns and df[df["price"] > 100000].shape[0] > 0:
        errors.append(f"发现 {df[df['price'] > 100000].shape[0]} 条价格异常（>10万）的数据")
    
    is_valid = len(errors) == 0
    if not is_valid:
        print("数据校验发现以下问题：")
        for err in errors:
            print(f"- {err}")
    else:
        print("数据校验通过")
    return is_valid, errors

# ====================== 8. JSONL输出模块（无修改） ======================
def convert_to_finetune_jsonl(df: pd.DataFrame, output_path: str) -> bool:
    finetune_data = []
    for _, row in df.iterrows():
        prompt = f"请介绍这款商品：{row['product_name']}（分类：{row['category']}）"
        completion = f"""
商品名称：{row['product_name']}
分类：{row['category']}
价格：{row['price']} 元
描述：{row['description']}
        """.strip()
        
        finetune_data.append({"prompt": prompt, "completion": completion})
    
    try:
        with open(output_path, mode="w", encoding="utf-8") as f:
            writer = jsonlines.Writer(f)
            writer.write_all(finetune_data)
        print(f"成功输出JSONL文件：{output_path}，共 {len(finetune_data)} 条数据")
        return True
    except Exception as e:
        print(f"输出JSONL失败：{str(e)}")
        return False

# ====================== 9. 主流程函数（核心调整：重构执行顺序） ======================
def ecommerce_data_clean_pipeline(input_file: str, output_file: str) -> bool:
    # 步骤1：读取数据
    df = read_structured_data(input_file)
    if df is None:
        return False
    
    # 步骤2：合并重复字段（商品名称→product_name/商品价格→product_price）
    df = merge_duplicate_fields(df, FIELD_CONFIG["duplicate_field_groups"])
    if df.empty:
        print("错误：合并重复字段后无数据")
        return False
    
    # 【核心调整】步骤3：提前执行字段标准化映射（product_price→price/商品分类→category等）
    df = standardize_field_name(df, FIELD_CONFIG["field_mapping"])
    if df.empty:
        print("错误：字段标准化后无数据")
        return False
    
    # 步骤4：删除冗余字段（基于映射后的白名单，不会误删核心字段）
    df = delete_redundant_fields(df, FIELD_CONFIG["keep_fields_whitelist"])
    if df.empty:
        print("错误：删除冗余字段后无数据")
        return False
    
    # 步骤5：数据清洗（基于最终标准化的核心字段）
    df_cleaned = clean_ecommerce_data(df)
    if df_cleaned.empty:
        print("错误：清洗后无数据")
        return False
    
    # 步骤6：数据校验
    is_valid, _ = validate_ecommerce_data(df_cleaned)
    if not is_valid:
        print("警告：数据校验未通过，但仍尝试输出")
    
    # 步骤7：输出JSONL
    return convert_to_finetune_jsonl(df_cleaned, output_file)

# ====================== 测试用例（无修改，复用原有测试数据） ======================
if __name__ == "__main__":
    test_data = {
        "商品名称": ["华为Mate60 Pro", None, "小米14", None, "华为Mate60 Pro"],
        "产品名": [None, "苹果iPhone15", None, "vivo X100", None],
        "商品价格": [6999, 5999, -3999, 4999, 6999],
        "商品库存": [1000, 2000, 1500, None, 1000],
        "销量": [5000, 8000, "6000", 7000, 5000],
        "录入时间": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "商品分类": ["手机", "手机", "手机", "手机", "手机"],
        "商品描述": ["鸿蒙系统，卫星通话", None, "骁龙8Gen3，徕卡影像", "", "鸿蒙系统，卫星通话"]
    }
    
    test_csv_path = "test_ecommerce_data_v2.csv"
    pd.DataFrame(test_data).to_csv(test_csv_path, index=False, encoding="utf-8")
    
    output_jsonl_path = "finetune_ecommerce_data_v2.jsonl"
    result = ecommerce_data_clean_pipeline(test_csv_path, output_jsonl_path)
    
    if result:
        print("\n测试完成！输出文件：", output_jsonl_path)
        with open(output_jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 2:
                    print(f"\n第{i+1}条微调数据：{line.strip()}")
    else:
        print("测试失败")