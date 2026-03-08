import json
from typing import List, Dict

def standardize_instruction(question: str) -> str:
    """
    标准化指令表述：统一为「请回答以下问题：{question}」
    Args:
        question: 原始校园问答的问题文本
    Returns:
        标准化后的指令字符串
    """
    if not isinstance(question, str):
        raise ValueError(f"问题文本必须为字符串类型，当前类型：{type(question)}")
    # 去除首尾空格，避免空字符串
    question = question.strip()
    return f"请回答以下问题：{question}"

def adjust_text_length(text: str, max_len: int = 512, min_len: int = 10, pad_char: str = "。") -> str:
    """
    调整文本长度：截断过长文本，补齐过短文本
    Args:
        text: 待处理的文本（如回答内容）
        max_len: 最大长度（默认512字）
        min_len: 最小长度（默认10字）
        pad_char: 补齐用的字符（默认中文句号，避免无意义字符）
    Returns:
        长度调整后的文本
    """
    if not isinstance(text, str):
        raise ValueError(f"待处理文本必须为字符串类型，当前类型：{type(text)}")
    text = text.strip()
    text_len = len(text)
    
    # 1. 截断过长文本
    if text_len > max_len:
        adjusted_text = text[:max_len]
        print(f"文本过长（{text_len}字），已截断至{max_len}字")
    # 2. 补齐过短文本
    elif text_len < min_len:
        pad_num = min_len - text_len
        adjusted_text = text + pad_char * pad_num
        print(f"文本过短（{text_len}字），已补齐至{min_len}字")
    # 3. 长度符合要求，直接返回
    else:
        adjusted_text = text
    
    return adjusted_text

def process_campus_qa_data(
    input_file: str,
    output_file: str,
    max_len: int = 512,
    min_len: int = 10
) -> Dict[str, any]:
    """
    批量处理校园问答数据：标准化指令 + 统一文本长度
    Args:
        input_file: 原始数据文件路径（JSONL格式，每行{"question": "...", "answer": "..."}）
        output_file: 处理后数据保存路径（JSONL格式，SFT格式）
        max_len: 文本最大长度
        min_len: 文本最小长度
    Returns:
        处理结果统计：总样本数、成功处理数、失败数
    """
    # 初始化统计信息
    stats = {
        "total_samples": 0,
        "success_samples": 0,
        "failed_samples": 0,
        "failed_details": []
    }

    # 读取原始数据并处理
    with open(input_file, "r", encoding="utf-8") as in_f, open(output_file, "w", encoding="utf-8") as out_f:
        for line_num, line in enumerate(in_f, 1):
            stats["total_samples"] += 1
            line = line.strip()
            if not line:
                stats["failed_samples"] += 1
                stats["failed_details"].append({"line": line_num, "reason": "空行"})
                continue
            
            try:
                # 解析原始数据
                raw_data = json.loads(line)
                # 校验原始数据字段
                if "question" not in raw_data or "answer" not in raw_data:
                    raise ValueError("缺失question/answer字段")
                
                # 步骤1：标准化指令（生成instruction）
                instruction = standardize_instruction(raw_data["question"])
                # 步骤2：调整回答文本长度（生成output）
                output = adjust_text_length(raw_data["answer"], max_len, min_len)
                # 步骤3：构造SFT格式数据（input为空）
                sft_data = {
                    "instruction": instruction,
                    "input": "",  # 校园问答无额外输入，设为空
                    "output": output
                }
                
                # 保存处理后的数据
                out_f.write(json.dumps(sft_data, ensure_ascii=False) + "\n")
                stats["success_samples"] += 1

            except Exception as e:
                stats["failed_samples"] += 1
                stats["failed_details"].append({"line": line_num, "reason": str(e)})
    
    return stats

# ---------------------- 测试示例 ----------------------
if __name__ == "__main__":
    # 1. 创建测试用的原始校园问答数据文件
    test_input_file = "campus_qa_raw.jsonl"
    with open(test_input_file, "w", encoding="utf-8") as f:
        # 正常样本
        f.write('{"question": "图书馆开放时间", "answer": "图书馆周一至周五8:00-22:00，周末9:00-21:00"}\n')
        # 过短回答样本
        f.write('{"question": "食堂有几层", "answer": "3层"}\n')
        # 过长回答样本（模拟513字）
        long_answer = "校园一卡通可用于食堂消费、图书馆借书、门禁通行、热水使用、洗衣机使用、打印机使用、浴室洗澡、超市购物、校医院挂号、体育场馆预约、班车乘坐、宿舍电费缴纳、校园网缴费、自助售货机购物、复印店消费、健身房使用、游泳馆使用、篮球场使用、足球场使用、网球场使用、羽毛球场使用、乒乓球场使用、排球场使用、报告厅预约、会议室预约、实验室预约、机房使用、校车乘坐、快递柜取件、饮水机取水、自动贩卖机购物、打印店打印、复印店复印、扫描店扫描、书店购书、文具店购物、水果店购物、奶茶店消费、咖啡店消费、面包店消费、理发店理发、洗衣店洗衣、干洗店干洗、照相馆拍照、打印店装订、文印店排版、广告店制作、花店买花、礼品店购物、眼镜店配镜、药店买药、超市买零食、超市买饮料、超市买日用品、超市买水果、超市买蔬菜、超市买肉类、超市买蛋类、超市买奶类、超市买粮油、超市买调料、超市买零食、超市买饮料、超市买日用品、超市买水果、超市买蔬菜、超市买肉类、超市买蛋类、超市买奶类、超市买粮油、超市买调料" * 5  # 超过512字
        f.write(f'{{"question": "校园一卡通用途", "answer": "{long_answer}"}}\n')
        # 缺失字段样本（用于测试错误处理）
        f.write('{"question": "奖学金申请条件"}\n')

    # 2. 执行数据处理
    test_output_file = "campus_qa_processed.jsonl"
    process_stats = process_campus_qa_data(test_input_file, test_output_file)

    # 3. 打印处理结果统计
    print("=== 数据处理结果统计 ===")
    print(f"总样本数：{process_stats['total_samples']}")
    print(f"成功处理数：{process_stats['success_samples']}")
    print(f"处理失败数：{process_stats['failed_samples']}")
    if process_stats["failed_details"]:
        print("失败详情：")
        for fail in process_stats["failed_details"]:
            print(f"  行{fail['line']}：{fail['reason']}")

    # 4. 验证处理后的数据
    print("\n=== 处理后的数据示例 ===")
    with open(test_output_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            print(f"instruction: {data['instruction']}")
            print(f"output: {data['output']}")
            print(f"output长度：{len(data['output'])}字")
            print("-" * 50)