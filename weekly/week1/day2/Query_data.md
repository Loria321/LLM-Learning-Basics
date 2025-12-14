


          
我需要查看当前文件的完整内容，以便全面分析和总结代码。让我使用view_files工具查看整个query_data.py文件。
        
            
toolName: view_files
            
status: success
          
            
filePath: d:\Working\LLM-Learning-Basics\weekly\week1\day2\langchain-rag-tutorial-main\query_data.py
          
# query_data.py 文件分析总结

## 文件功能概述

这是一个基于LangChain和OpenAI构建的RAG（检索增强生成）系统的查询脚本。它的主要功能是接收用户查询，从之前创建的Chroma向量数据库中检索相关文档片段，然后使用OpenAI的ChatGPT模型生成基于这些文档的回答。

## 代码结构与关键组件

### 1. 导入依赖库

```python
import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
```

- **命令行参数**：使用`argparse`处理用户输入的查询文本
- **向量数据库**：使用`Chroma`加载和查询向量数据库
- **嵌入模型**：使用`OpenAIEmbeddings`生成查询的嵌入向量
- **LLM模型**：使用`ChatOpenAI`调用OpenAI的聊天模型
- **提示模板**：使用`ChatPromptTemplate`创建提示模板

### 2. 常量定义

```python
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
```

- `CHROMA_PATH`：向量数据库的存储路径（与create_database.py保持一致）
- `PROMPT_TEMPLATE`：用于生成回答的提示模板，包含上下文和问题占位符

### 3. 主流程函数

```python
def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
```

主函数实现了完整的查询流程，包括：
1. **命令行接口**：接收用户输入的查询文本
2. **数据库准备**：加载Chroma向量数据库
3. **相似性搜索**：检索与查询最相关的文档片段
4. **相关性过滤**：仅当最相关结果的得分大于0.7时继续处理
5. **提示构建**：将检索到的文档片段和查询组合成提示
6. **生成回答**：使用ChatGPT模型生成基于上下文的回答
7. **结果展示**：输出回答和使用的源文档

### 4. 程序入口点

```python
if __name__ == "__main__":
    main()
```

- 确保脚本在直接运行时执行`main()`函数

## 核心技术点解析

### 1. RAG查询流程

该脚本完整实现了RAG系统的查询部分：
- **检索**：通过向量相似性搜索从数据库中获取相关文档
- **增强**：将检索到的文档作为上下文添加到提示中
- **生成**：使用LLM基于增强的提示生成回答

### 2. 向量数据库查询

```python
results = db.similarity_search_with_relevance_scores(query_text, k=3)
```

- 使用`similarity_search_with_relevance_scores`进行相似性搜索
- 参数`k=3`表示返回最相关的3个结果
- 结果包含文档对象和相关性得分

### 3. 相关性过滤

```python
if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")
    return
```

- 设置相关性阈值0.7，确保只使用高度相关的文档
- 当没有结果或最相关结果得分低于阈值时，返回"无法找到匹配结果"

### 4. 提示工程

```python
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
```

- 使用清晰的提示模板引导模型仅基于提供的上下文回答
- 上下文和问题通过占位符动态插入
- "---"分隔符帮助模型区分上下文和问题

### 5. 命令行界面

```python
parser = argparse.ArgumentParser()
parser.add_argument("query_text", type=str, help="The query text.")
args = parser.parse_args()
query_text = args.query_text
```

- 使用argparse创建简单易用的命令行接口
- 用户可以直接通过命令行输入查询文本

## 与create_database.py的关系

这个脚本与之前分析的create_database.py一起构成了完整的RAG系统：

1. **create_database.py**：负责数据准备和向量数据库构建
   - 加载原始文档
   - 分割文本为块
   - 生成嵌入并存储到Chroma数据库

2. **query_data.py**：负责查询和生成回答
   - 接收用户查询
   - 从Chroma数据库检索相关文档
   - 使用ChatGPT生成基于上下文的回答

## 使用方法

1. 确保已经运行过create_database.py创建了向量数据库
2. 在命令行中运行：`python query_data.py "你的查询文本"`
3. 脚本将输出：
   - 完整的提示（上下文+问题）
   - 基于上下文生成的回答
   - 使用的源文档列表

## 学习建议

1. 理解RAG系统的查询流程：查询→检索→增强→生成
2. 学习向量数据库的查询方法和参数调优
3. 掌握提示工程的基本技巧，如上下文组织、指令清晰度等
4. 了解相关性阈值的设置对结果质量的影响
5. 学习如何使用argparse创建命令行工具
6. 结合create_database.py理解完整的RAG系统构建流程

这个脚本提供了一个完整的RAG系统查询流程示例，适合作为学习LangChain和向量数据库应用的进阶案例。通过将它与create_database.py结合使用，你可以掌握一个功能完整的RAG系统的构建和使用方法。
        