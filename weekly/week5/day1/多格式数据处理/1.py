from pandas import json_normalize
import pandas as pd

# 原始嵌套数据（混合字典+列表）
nested_data = [
    {
        "商品ID": 1001,
        "商品名称": "纯棉T恤",
        "规格": {"颜色": "白色", "尺码": "M", "库存": 100}  # 嵌套字典
    },
    {
        "商品ID": 1002,
        "商品名称": "运动鞋",
        "规格": [{"颜色": "黑色", "尺码": "42", "库存": 50}, {"颜色": "白色", "尺码": "43", "库存": 30}]  # 嵌套列表
    }
]

# ===================== 步骤1：拆分数据为「字典型规格」和「列表型规格」 =====================
# 转换为DataFrame，方便按类型筛选
df = pd.DataFrame(nested_data)

# 筛选规格为字典的行
df_spec_dict = df[df["规格"].apply(lambda x: isinstance(x, dict))].copy()
# 筛选规格为列表的行
df_spec_list = df[df["规格"].apply(lambda x: isinstance(x, list))].copy()

# ===================== 步骤2：分别解析两种类型的规格 =====================
# 解析字典型规格：直接平铺（无需record_path）
df_dict_parsed = json_normalize(df_spec_dict.to_dict("records"), sep="_")
print("✅ 解析字典型规格结果：\n", df_dict_parsed)

# 解析列表型规格：指定record_path和meta（仅处理列表型数据）
df_list_parsed = json_normalize(
    df_spec_list.to_dict("records"),
    record_path=["规格"],  # 仅对列表型规格生效
    meta=["商品ID", "商品名称"],  # 保留顶层字段
    sep="_"
)
print("\n✅ 解析列表型规格结果：\n", df_list_parsed)

# ===================== 步骤3：合并两种解析结果（统一格式） =====================
# 为字典型结果补充列名（和列表型对齐），方便合并
df_dict_parsed = df_dict_parsed.rename(columns={
    "规格_颜色": "颜色",
    "规格_尺码": "尺码",
    "规格_库存": "库存"
})
# 选择和列表型一致的列
df_dict_parsed = df_dict_parsed[["商品ID", "商品名称", "颜色", "尺码", "库存"]]

# 合并结果
df_final = pd.concat([df_dict_parsed, df_list_parsed], ignore_index=True)
print("\n✅ 最终合并结果（统一格式）：\n", df_final)