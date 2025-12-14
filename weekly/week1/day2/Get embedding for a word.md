# Get embedding for a word
    embedding_function = OpenAIEmbeddings()#创建OpenAIEmbeddings实例作为嵌入生成器
    vector = embedding_function.embed_query("apple")#调用embed_query方法生成"apple"的嵌入向量
    print(f"Vector for 'apple': {vector}")#打印"apple"的嵌入向量
    #例如:Vector for 'apple': [0.0123, -0.0456, 0.0789, ...]  # 实际输出是1536个浮点数
    print(f"Vector length: {len(vector)}")#打印嵌入向量的长度
    #例如:Vector length: 1536