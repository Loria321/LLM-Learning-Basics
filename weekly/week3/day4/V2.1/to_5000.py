import pandas as pd
# 读取50k条数据，取前5000条
df = pd.read_csv("imdb_50k.csv")
df = df.head(5000)  # 截取前5000条，符合需求
df.to_csv("imdb_5000.csv", index=False, encoding='utf-8')
print(f"数据集已保存：imdb_5000.csv，共{len(df)}条数据")