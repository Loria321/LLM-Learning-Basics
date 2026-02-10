import os
from openai import OpenAI
from numpy import dot
from numpy.linalg import norm
# from dotenv import load_dotenv  # 可选，方便加载本地.env文件

# 加载本地.env文件中的环境变量（若未配置系统环境变量则启用）
# load_dotenv()

# 定义余弦相似度计算函数（核心逻辑不变）
def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# 定义通义千问Embedding调用函数
def get_qwen_embedding(texts):
    """
    调用通义千问Embedding API生成文本向量
    :param texts: 文本列表（支持单/多文本）
    :return: 向量列表
    """
    try:
        # 初始化OpenAI客户端（兼容阿里云百炼）
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),  # 从环境变量读取API Key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云兼容模式地址
        )

        # 调用Embedding API（通义千问Embedding模型）
        response = client.embeddings.create(
            model="text-embedding-v2",  # 通义千问Embedding模型名称（固定）
            input=texts  # 传入待生成向量的文本
        )

        # 提取向量结果
        embeddings = [item.embedding for item in response.data]
        return embeddings

    except Exception as e:
        # 异常捕获并给出清晰提示
        print(f"Embedding调用失败，错误信息：{e}")
        print("排查建议：")
        print("1. 检查API Key是否正确（DASHSCOPE_API_KEY）")
        print("2. 确认API Key有Embedding调用权限")
        print("3. 参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")
        raise  # 可选：重新抛出异常，让上层处理；也可返回空列表[]

# 主函数测试
if __name__ == "__main__":
    # 测试文本
    test_texts = [
        "我想吃苹果",
        "我想吃香蕉",
        "今天天气很好"
    ]

    try:
        # 生成Embedding向量
        embeddings = get_qwen_embedding(test_texts)
        print(f"生成的向量数量：{len(embeddings)}，单个向量维度：{len(embeddings[0])}")

        # 计算并打印相似度
        sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])  # 苹果 vs 香蕉
        sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])  # 苹果 vs 天气

        print(f"苹果 vs 香蕉 相似度：{sim_1_2:.4f}")  # 保留4位小数，更易读
        print(f"苹果 vs 天气 相似度：{sim_1_3:.4f}")

    except Exception as e:
        print(f"程序执行失败：{e}")