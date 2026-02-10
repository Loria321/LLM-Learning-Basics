import os
from openai import OpenAI

try:
    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 定义两组对比输入数据（低质量数据 vs 高质量数据）
    comparison_inputs = {
        "低质量数据": "杭电 智棵 专业 大 2 课成 有 人工 智能 原理 还有 什么 忘了",
        "高质量数据": "杭州电子科技大学智能科学与技术专业大二已学课程有人工智能原理，还需要学习哪些核心课程？"
    }

    # 统一系统提示词（保证模型角色一致，排除其他变量干扰）
    system_prompt = "你是一个专业的教育信息查询助手，回答需准确、全面，优先分点罗列关键信息。"

    # 遍历两组输入，依次调用模型并输出对比结果
    print("===== 数据格式对模型输出的影响对比实验 =====")
    for input_type, user_content in comparison_inputs.items():
        print(f"\n【输入类型】：{input_type}")
        print(f"【输入内容】：{user_content}")
        print(f"【模型输出】：")
        
        # 发起API调用（保持模型参数一致，确保对比有效性）
        completion = client.chat.completions.create(
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_content}
            ],
            temperature=0.3,  # 固定温度值，避免输出随机性影响对比
            max_tokens=1000    # 限制输出长度，保证回复简洁完整
        )
        
        # 打印模型输出
        response_content = completion.choices[0].message.content
        print(response_content)
        print("-" * 80)  # 分隔线，清晰区分两组结果

except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")