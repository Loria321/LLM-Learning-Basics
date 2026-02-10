import pandas as pd
from pandas import json_normalize
import json

# ===================== 步骤1：读取JSON数据 =====================
def read_json_data(file_path):
    """读取JSON文件，处理编码和格式问题"""
    try:
        # 方式1：直接用pd.read_json（适合标准JSON）
        df = pd.read_json(file_path, orient="records", encoding="utf-8")
        print(f"✅ 读取JSON成功，原始数据量：{len(df)} 条")
        return df
    except Exception as e:
        # 方式2：手动读取JSON字符串（兼容非标准格式）
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"✅ 手动读取JSON成功，原始数据量：{len(df)} 条")
        return df

# ===================== 步骤2：解析嵌套的“规格”字段 =====================
def parse_nested_spec(df):
    """解析嵌套的规格字段（兼容字典/列表两种嵌套形式）"""
    # 拆分两种嵌套类型：规格是字典 / 规格是列表
    df_dict = df[df["规格"].apply(lambda x: isinstance(x, dict))].copy()
    df_list = df[df["规格"].apply(lambda x: isinstance(x, list))].copy()
    
    # 解析字典型规格（直接平铺）
    df_dict_normalize = json_normalize(df_dict.to_dict("records"), sep="_")
    
    # 解析列表型规格（指定record_path和meta）
    if not df_list.empty:
        df_list_normalize = json_normalize(
            df_list.to_dict("records"),
            record_path=["规格"],  # 嵌套列表的路径
            meta=["商品ID", "商品名称", "价格", "品牌", "上架时间", "销量"],  # 保留顶层字段
            sep="_"
        )
    else:
        df_list_normalize = pd.DataFrame()
    
    # 合并两种类型的解析结果
    df_parsed = pd.concat([df_dict_normalize, df_list_normalize], ignore_index=True)
    print(f"✅ 解析嵌套规格成功，解析后数据量：{len(df_parsed)} 条")
    return df_parsed

# ===================== 步骤3：数据清洗（去重/缺值/格式标准化） =====================
def clean_goods_data(df):
    """电商商品数据清洗：去重、补全缺值、格式校验"""
    # 1. 去重（根据商品ID+规格_颜色+规格_尺码去重）
    df = df.drop_duplicates(subset=["商品ID", "规格_颜色", "规格_尺码"], keep="first")
    print(f"🔍 去重后数据量：{len(df)} 条")
    
    # 2. 处理缺值（核心字段补全/过滤）
    # 过滤商品名称为空的无效数据
    df = df[df["商品名称"].notna() & (df["商品名称"].str.strip() != "")]
    # 价格缺值填充为0，转为浮点数
    df["价格"] = pd.to_numeric(df["价格"], errors="coerce").fillna(0.0)
    # 品牌缺值填充为“未知品牌”
    df["品牌"] = df["品牌"].fillna("未知品牌").replace("", "未知品牌")
    # 上架时间缺值填充为“未上架”
    df["上架时间"] = df["上架时间"].fillna("未上架").replace("", "未上架")
    
    # 3. 格式标准化
    # 库存校验：负数库存改为0
    df["规格_库存"] = pd.to_numeric(df["规格_库存"], errors="coerce").fillna(0)
    df.loc[df["规格_库存"] < 0, "规格_库存"] = 0
    # 销量标准化：转为整数
    df["销量"] = pd.to_numeric(df["销量"], errors="coerce").fillna(0).astype(int)
    
    # 4. 过滤无效数据（价格为0且销量为0的商品）
    df = df[~((df["价格"] == 0) & (df["销量"] == 0))]
    print(f"✅ 清洗完成，最终数据量：{len(df)} 条")
    return df

# ===================== 步骤4：保存为Excel/JSON（多格式输出） =====================
def save_cleaned_data(df, excel_path="goods_cleaned.xlsx", json_path="goods_cleaned.json"):
    """保存清洗后数据为Excel和JSON"""
    # 保存为Excel（支持多工作表）
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="清洗后商品数据", index=False)
    print(f"📁 清洗后数据已保存至Excel：{excel_path}")
    
    # 保存为JSON（records格式，便于后续读取）
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    print(f"📁 清洗后数据已保存至JSON：{json_path}")

# ===================== 主执行逻辑 =====================
if __name__ == "__main__":
    # 优化：使用原始字符串避免路径转义问题
    df_raw = read_json_data(r"weekly\week5\day1\goods_raw.json")
    
    # 2. 解析嵌套规格
    df_parsed = parse_nested_spec(df_raw)
    
    # 3. 数据清洗
    df_cleaned = clean_goods_data(df_parsed)
    
    # 4. 打印清洗结果预览
    print("\n📊 清洗后数据预览：")
    print(df_cleaned[["商品ID", "商品名称", "价格", "品牌", "规格_颜色", "规格_尺码", "规格_库存", "销量"]].head())
    
    # 5. 保存为Excel/JSON
    save_cleaned_data(df_cleaned)