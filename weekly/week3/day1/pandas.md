# Pandas 核心数据处理操作速查手册
## 概述
本手册整合了Pandas常用数据处理操作，涵盖环境搭建、核心数据结构、数据读写、基础操作、数据清洗、异常值处理及实战案例，方便快速查询和复用。

## 一、环境搭建与验证
### 1. Pandas 安装
```bash
# pip 安装（国内源加速）
pip install pandas numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

# conda 安装（Anaconda/Miniconda环境）
conda install pandas numpy
```

### 2. 安装验证
```python
import pandas as pd
import numpy as np

# 打印版本号，确认安装成功
print(f"Pandas 版本：{pd.__version__}")
print(f"NumPy 版本：{np.__version__}")
```

## 二、核心数据结构（Series & DataFrame）
### 1. Series（一维标签数组）
#### （1）创建 Series
```python
# 从列表创建（默认索引）
s1 = pd.Series([10, 20, 30, 40, 50])

# 自定义索引
s2 = pd.Series([10, 20, 30, 40, 50], index=["a", "b", "c", "d", "e"])

# 从字典创建
s3 = pd.Series({"北京": 100, "上海": 200, "广州": 300, "深圳": 400})
```

#### （2）Series 核心属性
```python
# 查看数据值、索引、数据类型、长度
print("数据值：", s2.values)
print("索引：", s2.index)
print("数据类型：", s2.dtype)
print("长度：", len(s2))
print("是否为空：", s2.empty)
```

### 2. DataFrame（二维标签表格）
#### （1）创建 DataFrame
```python
# 方式1：字典列表
data1 = [
    {"姓名": "张三", "年龄": 25, "城市": "北京"},
    {"姓名": "李四", "年龄": 30, "城市": "上海"},
    {"姓名": "王五", "年龄": 28, "城市": "广州"}
]
df1 = pd.DataFrame(data1)

# 方式2：嵌套列表+自定义列名/索引
data2 = [
    ["张三", 25, "北京"],
    ["李四", 30, "上海"],
    ["王五", 28, "广州"]
]
df2 = pd.DataFrame(data2, columns=["姓名", "年龄", "城市"], index=["a", "b", "c"])

# 方式3：字典（键为列名，值为列数据）
data3 = {
    "姓名": ["张三", "李四", "王五"],
    "年龄": [25, 30, 28],
    "城市": ["北京", "上海", "广州"]
}
df3 = pd.DataFrame(data3)
```

#### （2）DataFrame 核心属性
```python
# 列索引、行索引、数据形状
print("列索引：", df1.columns)
print("行索引：", df1.index)
print("数据形状（行, 列）：", df1.shape)

# 数据类型与基本信息
print("每列数据类型：")
print(df1.dtypes)
print("\n数据基本信息：")
df1.info()

# 数值型列统计描述
print("\n数值型列统计描述：")
print(df1.describe())
```

## 三、数据读取与写入
### 1. 数据读取
```python
# 读取 CSV 文件
df_csv = pd.read_csv(
    "data.csv",
    encoding="utf-8",  # 中文编码：utf-8/gbk
    skiprows=1,        # 跳过第1行（表头外的无效行）
    usecols=["姓名", "年龄", "城市"]  # 仅读取指定列
)

# 读取 Excel 文件（需安装 openpyxl）
# pip install openpyxl
df_excel = pd.read_excel(
    "data.xlsx",
    sheet_name=0,      # 第1个工作表
    engine="openpyxl"
)

# 读取 JSON 文件
df_json = pd.read_json(
    "data.json",
    orient="records"   # 按记录格式读取
)
```

### 2. 数据写入
```python
# 写入 CSV 文件
df1.to_csv(
    "output_data.csv",
    index=False,       # 不写入行索引
    encoding="utf-8"
)

# 写入 Excel 文件
df1.to_excel(
    "output_data.xlsx",
    index=False,
    engine="openpyxl",
    sheet_name="员工信息"
)

# 写入 JSON 文件
df1.to_json(
    "output_data.json",
    orient="records",
    force_ascii=False  # 保留中文
)
```

## 四、数据基础操作
### 1. 数据查看
```python
# 查看前n行（默认5行）
print(df1.head(2))

# 查看后n行（默认5行）
print(df1.tail(2))

# 随机查看n行
print(df1.sample(2))
```

### 2. 数据选择
```python
# 选择单列（两种方式）
name_col = df1["姓名"]
name_col = df1.姓名  # 列名无特殊字符时可用

# 选择多列
name_age_col = df1[["姓名", "年龄"]]

# 按标签选择行/列（loc）
row_0 = df1.loc[0]  # 第0行
row0_col_name = df1.loc[0, "姓名"]  # 第0行「姓名」列
row0_1_name_city = df1.loc[0:1, ["姓名", "城市"]]  # 0-1行，指定列

# 按位置选择行/列（iloc）
first_row = df1.iloc[0]  # 第1行（下标0）
first_col = df1.iloc[:, 0]  # 第1列（下标0）
row0_1_col0_1 = df1.iloc[0:2, 0:2]  # 0-1行，0-1列
```

### 3. 条件筛选
```python
# 单条件筛选（年龄>26）
df_filter1 = df1[df1["年龄"] > 26]

# 多条件筛选（年龄>26 且 城市=广州）
df_filter2 = df1[(df1["年龄"] > 26) & (df1["城市"] == "广州")]

# 多条件筛选（年龄>30 或 城市=北京）
df_filter3 = df1[(df1["年龄"] > 30) | (df1["城市"] == "北京")]
```

## 五、核心数据清洗操作
### 1. 去重（drop_duplicates）
```python
# 创建带重复值的测试数据
df_dup = pd.DataFrame({
    "姓名": ["张三", "张三", "李四", "王五", "李四"],
    "年龄": [25, 25, 30, 28, 30],
    "城市": ["北京", "北京", "上海", "广州", "上海"]
})

# 检测重复值
print("重复值标记：")
print(df_dup.duplicated())
print("重复行数量：", df_dup.duplicated().sum())

# 删除重复值
df_dup_drop1 = df_dup.drop_duplicates(keep="first")  # 保留第一条
df_dup_drop2 = df_dup.drop_duplicates(keep="last")   # 保留最后一条
df_dup_drop3 = df_dup.drop_duplicates(keep=False)    # 删除所有重复行
df_dup_drop4 = df_dup.drop_duplicates(subset=["姓名", "城市"], keep="first")  # 按指定列去重
```

### 2. 缺失值处理（fillna / dropna）
```python
# 创建带缺失值的测试数据
df_missing = pd.DataFrame({
    "姓名": ["张三", "李四", None, "王五"],
    "年龄": [25, None, 28, 30],
    "城市": ["北京", "上海", "广州", None]
})

# 检测缺失值
print("每列缺失值数量：")
print(df_missing.isnull().sum())

# 删除缺失值
df_drop1 = df_missing.dropna()  # 删除含任意缺失值的行
df_drop2 = df_missing.dropna(how="all")  # 仅删除全缺失的行

# 填充缺失值
df_fill1 = df_missing.fillna({
    "姓名": "未知",
    "年龄": 0,
    "城市": "未知"
})  # 固定值填充

df_fill2 = df_missing.copy()
df_fill2["年龄"] = df_fill2["年龄"].fillna(df_fill2["年龄"].mean())  # 均值填充
df_fill2["姓名"] = df_fill2["姓名"].fillna(df_fill2["姓名"].mode()[0])  # 众数填充
```

### 3. 数据类型转换（astype）
```python
# 创建测试数据
df_type = pd.DataFrame({
    "姓名": ["张三", "李四", "王五"],
    "年龄": ["25", "30", "28"],  # 字符串类型
    "薪资": [5000.0, 8000.0, 6000.0],  # 浮点型
    "入职日期": ["2020-01-01", "2021-03-15", "2022-05-20"]
})

# 查看原始数据类型
print("原始数据类型：")
print(df_type.dtypes)

# 类型转换
df_type["年龄"] = df_type["年龄"].astype(int)  # 字符串转整数
df_type["薪资"] = df_type["薪资"].astype(int)  # 浮点转整数
df_type["入职日期"] = pd.to_datetime(df_type["入职日期"])  # 字符串转日期类型

# 查看转换后数据类型
print("\n转换后数据类型：")
print(df_type.dtypes)
```

## 六、异常值检测与处理
### 1. 异常值检测
```python
# 创建带异常值的测试数据
df_outlier = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六", "孙七"],
    "年龄": [25, 30, 28, 100, -5],
    "薪资": [8000, 10000, 9000, 50000, 3000]
})

# 方法1：描述统计（describe）
print("数值型列描述统计：")
print(df_outlier[["年龄", "薪资"]].describe())

# 方法2：箱线图（boxplot）
import matplotlib.pyplot as plt

# 绘制年龄列箱线图
df_outlier.boxplot(column="年龄")
plt.title("年龄列异常值箱线图")
plt.show()

# 绘制薪资列箱线图
df_outlier.boxplot(column="薪资")
plt.title("薪资列异常值箱线图")
plt.show()

# 方法3：四分位数法（IQR）检测
def detect_outlier(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (series < lower_bound) | (series > upper_bound)

# 检测年龄和薪资异常值
age_outlier = detect_outlier(df_outlier["年龄"])
salary_outlier = detect_outlier(df_outlier["薪资"])
print("\n年龄异常值：")
print(df_outlier.loc[age_outlier, "年龄"])
print("\n薪资异常值：")
print(df_outlier.loc[salary_outlier, "薪资"])
```

### 2. 异常值处理策略
```python
# 策略1：删除异常值
df_outlier_drop = df_outlier[~age_outlier & ~salary_outlier]
print("\n删除异常值后：")
print(df_outlier_drop)

# 策略2：替换异常值（均值/中位数）
df_outlier_fill = df_outlier.copy()
# 年龄异常值替换为中位数
age_median = df_outlier_fill["年龄"].median()
df_outlier_fill.loc[age_outlier, "年龄"] = age_median
# 薪资异常值替换为中位数
salary_median = df_outlier_fill["薪资"].median()
df_outlier_fill.loc[salary_outlier, "薪资"] = salary_median
print("\n中位数替换异常值后：")
print(df_outlier_fill)

# 策略3：分箱处理（pd.cut / pd.qcut）
# 薪资分箱（等距分箱）
df_outlier["薪资分箱"] = pd.cut(
    df_outlier["薪资"],
    bins=3,
    labels=["低薪资", "中薪资", "高薪资"]
)

# 年龄分箱（分位数分箱）
df_outlier["年龄分箱"] = pd.qcut(
    df_outlier["年龄"],
    q=3,
    labels=["青年", "中年", "老年"]
)
print("\n分箱处理后：")
print(df_outlier[["姓名", "年龄", "年龄分箱", "薪资", "薪资分箱"]])
```

## 七、进阶数据操作
### 1. 数据排序
```python
# 按列值升序排序
df_sort1 = df1.sort_values(by="年龄", ascending=True)

# 按多列排序（城市升序，年龄降序）
df_sort2 = df1.sort_values(by=["城市", "年龄"], ascending=[True, False])

# 按行索引降序排序
df_sort3 = df2.sort_index(ascending=False)

# 按列索引降序排序
df_sort4 = df2.sort_index(axis=1, ascending=False)
```

### 2. 分组聚合（groupby）
```python
# 创建测试数据
df_group = pd.DataFrame({
    "部门": ["技术部", "技术部", "市场部", "市场部", "人事部", "技术部"],
    "姓名": ["张三", "李四", "王五", "赵六", "孙七", "周八"],
    "年龄": [25, 30, 28, 35, 26, 29],
    "薪资": [8000, 12000, 9000, 15000, 7000, 10000]
})

# 按部门分组，计算薪资均值
group_mean = df_group.groupby("部门")["薪资"].mean()

# 按部门分组，多指标统计
group_stats = df_group.groupby("部门")["薪资"].agg(["mean", "max", "min", "sum"])
group_stats.columns = ["薪资均值", "薪资最大值", "薪资最小值", "薪资总和"]

# 按部门分组，多列不同聚合
group_multi = df_group.groupby("部门").agg(
    年龄均值=("年龄", "mean"),
    薪资最大值=("薪资", "max"),
    人数=("姓名", "count")
)
```

### 3. 数据合并
```python
# （1）concat 拼接
df_concat1 = pd.DataFrame({"姓名": ["张三", "李四"], "年龄": [25, 30]})
df_concat2 = pd.DataFrame({"姓名": ["王五", "赵六"], "年龄": [28, 35]})
# 上下拼接
df_concat_row = pd.concat([df_concat1, df_concat2], ignore_index=True)
# 左右拼接
df_concat_col = pd.concat([df_concat1, pd.DataFrame({"城市": ["北京", "上海"]})], axis=1)

# （2）merge 关联（类似SQL JOIN）
df_merge1 = pd.DataFrame({"员工ID": [101, 102, 103], "姓名": ["张三", "李四", "王五"], "部门ID": [1, 2, 1]})
df_merge2 = pd.DataFrame({"部门ID": [1, 2, 3], "部门名称": ["技术部", "市场部", "人事部"]})
# 内连接
df_merge_inner = pd.merge(df_merge1, df_merge2, on="部门ID")
# 左连接
df_merge_left = pd.merge(df_merge1, df_merge2, on="部门ID", how="left")
# 右连接
df_merge_right = pd.merge(df_merge1, df_merge2, on="部门ID", how="right")
# 全连接
df_merge_outer = pd.merge(df_merge1, df_merge2, on="部门ID", how="outer")
```

## 八、实战案例
### 案例1：员工薪水翻倍
```python
import pandas as pd
from typing import List

class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        total_apple = sum(apple)
        if total_apple == 0:
            return 0
        capacity.sort(reverse=True)
        current_cap_sum = 0
        box_num = 0
        for cap in capacity:
            current_cap_sum += cap
            box_num += 1
            if current_cap_sum >= total_apple:
                return box_num
        return box_num

# 测试
if __name__ == "__main__":
    sol = Solution()
    apple = [1,2,3]
    capacity = [3,2,4]
    print(sol.minimumBoxes(apple, capacity))  # 输出2
```

### 案例2：完整数据清洗流程（读取→去重→缺失值填充→保存）
```python
import pandas as pd

def complete_data_cleaning(input_path, output_path):
    # 1. 读取数据
    df = pd.read_csv(input_path, encoding="utf-8")
    print("原始数据：")
    print(df.head())
    
    # 2. 去重
    df = df.drop_duplicates(subset=["姓名"], keep="first")
    print("\n去重后数据：")
    print(df.head())
    
    # 3. 缺失值填充
    df["年龄"] = df["年龄"].fillna(df["年龄"].median())
    df["城市"] = df["城市"].fillna(df["城市"].mode()[0])
    df["薪资"] = df["薪资"].fillna(0)
    print("\n缺失值填充后数据：")
    print(df.head())
    
    # 4. 数据类型转换
    df["年龄"] = df["年龄"].astype(int)
    df["薪资"] = df["薪资"].astype(int)
    
    # 5. 保存清洗后数据
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n清洗后数据已保存至 {output_path}")
    return df

# 执行清洗
if __name__ == "__main__":
    cleaned_df = complete_data_cleaning("raw_employees.csv", "cleaned_employees.csv")
```

## 九、快速查询索引
| 操作类型         | 关键词/函数                | 所在章节       |
|------------------|---------------------------|----------------|
| 环境搭建         | pip install pandas、pd.__version__ | 一             |
| Series 创建      | pd.Series()                | 二.1           |
| DataFrame 创建   | pd.DataFrame()             | 二.2           |
| CSV 读写         | pd.read_csv()、df.to_csv()  | 三             |
| Excel 读写       | pd.read_excel()、df.to_excel() | 三         |
| 数据查看         | head()、tail()、sample()    | 四.1           |
| 数据选择         | loc、iloc                  | 四.2           |
| 条件筛选         | &、\|                      | 四.3           |
| 去重             | drop_duplicates()、duplicated() | 五.1         |
| 缺失值处理       | fillna()、dropna()         | 五.2           |
| 类型转换         | astype()、pd.to_datetime()  | 五.3           |
| 异常值检测       | describe()、boxplot()、IQR  | 六.1           |
| 异常值处理       | 删除、替换、pd.cut()       | 六.2           |
| 数据排序         | sort_values()、sort_index() | 七.1           |
| 分组聚合         | groupby()、agg()           | 七.2           |
| 数据合并         | concat()、merge()          | 七.3           |
| 薪水翻倍         | df["salary"] * 2           | 八.1           |
| 完整清洗流程     | 读取→去重→填充→保存       | 八.2           |