def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    #删除已存在的数据库目录,确保每次运行都从干净的状态开始

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    #使用Chroma.from_documents()将文本块转换为向量并存储
    #使用OpenAIEmbeddings生成文本嵌入

    db.persist()#将数据库实例db持久化到指定的目录CHROMA_PATH
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")#打印保存的文本块数量和数据库路径
    #例如:Saved 20 chunks to data/chroma.
