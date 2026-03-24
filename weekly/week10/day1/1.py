import chromadb

# 1. 初始化客户端（本地模式，自动创建文件存储）
client = chromadb.PersistentClient(path="./chroma_db")

# 2. 创建集合（相当于数据库的表，存储向量）
collection = client.get_or_create_collection(name="my_first_collection")

# 3. 添加数据（自动生成向量，无需手动处理）
collection.add(
    documents=["我喜欢吃火锅", "今天天气晴朗", "向量数据库很有用"],
    metadatas=[{"type": "food"}, {"type": "weather"}, {"type": "tech"}],
    ids=["id1", "id2", "id3"]
)

# 4. 相似性检索（核心功能！查最相似的2条数据）
results = collection.query(
    query_texts=["天气怎么样"],
    n_results=2
)

# 打印结果
print("检索结果：", results["documents"])