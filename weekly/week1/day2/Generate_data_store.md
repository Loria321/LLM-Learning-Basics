CHROMA_PATH = "chroma"  
DATA_PATH = "data/books"  
#定义要存储的向量数据库路径和数据路径

def main():
    generate_data_store()

def generate_data_store():  
    documents = load_documents()#加载文档   
    chunks = split_text(documents)#将文档切分成文本块   
    save_to_chroma(chunks)#将文本块保存到向量数据库中
# 加载、切分、保存
## 加载文档
def load_documents():   
    loader = DirectoryLoader(DATA_PATH, glob="*.md")    
    documents = loader.load()   
    return documents    
## 切分文档
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,#每个文本块的最大字符数
        chunk_overlap=100,#每个文本块之间的重叠字符数,保持上下文连贯性
        length_function=len,
        add_start_index=True,#是否在元数据中添加文档的起始索引
    )#创建递归字符文本切分器实例

    chunks = text_splitter.split_documents(documents)#将文档切分成文本块
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")#打印切分后的文本块数量
    #例如:Split 1 documents into 20 chunks.

    document = chunks[10]#获取切分后的文本块列表中的第11个文本块
    print(document.page_content)#打印第11个文本块的内容
    print(document.metadata)#打印第11个文本块的元数据
    #例如:{'source': 'data/books/books.md', 'start_index': 1000}

    return chunks
