import json
import random
import re
from typing import List, Dict, Set

# ---------------------- 1. 自定义同义词表（适配校园问答场景） ----------------------
CAMPUS_SYNONYM_MAP = {
    "开放时间": ["营业时段", "开放时段", "运营时间", "使用时间"],
    "食堂": ["餐厅", "饭堂"],
    "几层": ["多少层", "楼层数"],
    "奖学金": ["助学金", "奖励金"],
    "申请": ["申报", "申领", "办理"],
    "校园一卡通": ["校园卡", "一卡通", "学生卡"],
    "办理": ["申请", "办理", "补办"],
    "宿舍": ["寝室", "住宿楼"],
    "限电": ["功率限制", "用电限制"],
    "补考": ["补测", "重考"],
    "选修课": ["选修课程", "可选课程"],
    "校医院": ["校园医院", "学校医务室"],
    "打印店": ["文印店", "复印店"],
    "在哪里": ["在何处", "位置在哪", "具体位置"],
    "多少瓦": ["功率多少", "多少功率"],
    "上班时间": ["工作时间", "接诊时间"],
    "篮球场": ["篮球场地", "蓝球场"],  # 兼容常见错别字
}

# ---------------------- 2. 标签处理函数（保持不变） ----------------------
def clean_output_label(output: str) -> str:
    """
    标签（output）处理：保证准确性、一致性
    """
    if not isinstance(output, str):
        return ""
    
    # 基础清洗：去空格、换行、语气词
    output = output.strip().replace("\n", "").replace("\r", "")
    tone_words = ["哦", "呢", "啊", "吧", "啦", "嘛"]
    for word in tone_words:
        output = output.replace(word, "")
    
    # 统一时间格式（简单匹配，可根据实际场景扩展）
    time_mapping = {
        "8点": "8:00", "9点": "9:00", "10点": "10:00",
        "11点": "11:00", "12点": "12:00", "13点": "13:00",
        "14点": "14:00", "15点": "15:00", "16点": "16:00",
        "17点": "17:00", "18点": "18:00", "19点": "19:00",
        "20点": "20:00", "21点": "21:00", "22点": "22:00",
        "23点": "23:00", "0点": "0:00", "凌晨0点": "0:00"
    }
    for old, new in time_mapping.items():
        output = output.replace(old, new)
    
    # 统一单位表述
    unit_mapping = {
        "层楼": "层", "个楼层": "层", "元钱": "元",
        "小时长": "小时", "天时间": "天"
    }
    for old, new in unit_mapping.items():
        output = output.replace(old, new)
    
    return output

# ---------------------- 3. 数据增强函数（替换synonyms，改用自定义同义词表） ----------------------
def synonym_replacement(text: str) -> str:
    """
    自定义同义词替换：基于校园场景同义词表，保持语义不变
    """
    if not text:
        return text
    
    # 随机选择1-2个词进行替换（避免过度替换）
    replace_times = random.choice([1, 2])
    replaced_text = text
    
    # 遍历同义词表，随机替换
    synonym_items = list(CAMPUS_SYNONYM_MAP.items())
    random.shuffle(synonym_items)  # 打乱顺序，保证随机性
    
    replace_count = 0
    for original_word, syn_list in synonym_items:
        if replace_count >= replace_times:
            break
        if original_word in replaced_text:
            # 随机选一个同义词替换
            syn_word = random.choice(syn_list)
            replaced_text = replaced_text.replace(original_word, syn_word, 1)  # 只替换一次
            replace_count += 1
    
    return replaced_text

def sentence_paraphrase(text: str) -> str:
    """
    句式变换：调整语序/改写句式，保持语义不变
    """
    if not text:
        return text
    
    # 定义句式变换规则（适配校园问答）
    paraphrase_rules = [
        # 原模式 → 新模式
        (r"^(.*)开放时间\？$", r"\1的开放时间是多少？"),
        (r"^(.*)有几层\？$", r"\1的楼层数量是多少？"),
        (r"^(.*)怎么申请\？$", r"如何申请\1？"),
        (r"^(.*)在哪里\？$", r"\1的具体位置在哪里？"),
        (r"^(.*)限电多少瓦\？$", r"\1的用电功率限制是多少瓦？"),
        (r"^(.*)补考时间\？$", r"\1的补考时间是什么时候？"),
        (r"^(.*)选修课有多少门\？$", r"\1的选修课数量有多少门？"),
    ]
    
    for pattern, repl in paraphrase_rules:
        if re.match(pattern, text):
            return re.sub(pattern, repl, text)
    
    # 无匹配规则时，微调语序（仅针对短文本）
    if len(text) > 5 and text.endswith("？"):
        # 简单拆分：如“图书馆开放时间？” → “开放时间 图书馆？”
        if "开放时间" in text:
            return text.replace("开放时间", "") + "的开放时间？"
        elif "在哪里" in text:
            return text.replace("在哪里", "") + "的位置在哪里？"
    
    return text

def augment_single_sample(sample: Dict[str, str]) -> List[Dict[str, str]]:
    """
    单样本增强：生成多个变体（自定义同义词替换+句式变换）
    """
    augmented_samples = []
    original_question = sample["question"]
    original_answer = clean_output_label(sample["answer"])  # 清洗标签
    
    # 生成原始样本（清洗标签后）
    augmented_samples.append({
        "question": original_question,
        "answer": original_answer
    })
    
    # 生成同义词替换变体
    syn_question = synonym_replacement(original_question)
    augmented_samples.append({
        "question": syn_question,
        "answer": original_answer
    })
    
    # 生成句式变换变体
    para_question = sentence_paraphrase(original_question)
    augmented_samples.append({
        "question": para_question,
        "answer": original_answer
    })
    
    # 生成“同义词+句式变换”组合变体
    syn_para_question = sentence_paraphrase(syn_question)
    augmented_samples.append({
        "question": syn_para_question,
        "answer": original_answer
    })
    
    return augmented_samples

# ---------------------- 4. 批量增强到指定数量 ----------------------
def batch_augment_data(
    raw_data: List[Dict[str, str]],
    target_num: int = 1000
) -> List[Dict[str, str]]:
    """
    批量增强数据到指定数量
    """
    augmented_data = []
    seen_questions = set()  # 去重集合
    
    # 循环增强直到达到目标数量
    while len(augmented_data) < target_num:
        for sample in raw_data:
            if len(augmented_data) >= target_num:
                break
            variants = augment_single_sample(sample)
            for var in variants:
                if len(augmented_data) >= target_num:
                    break
                if var["question"] not in seen_questions:
                    seen_questions.add(var["question"])
                    augmented_data.append(var)
    
    # 截断到目标数量
    augmented_data = augmented_data[:target_num]
    
    # 转换为SFT格式
    sft_data = []
    for item in augmented_data:
        sft_data.append({
            "instruction": f"请回答以下问题：{item['question']}",
            "input": "",
            "output": item["answer"]
        })
    
    return sft_data

# ---------------------- 5. 主函数：数据增强流程 ----------------------
if __name__ == "__main__":
    # 步骤1：原始校园问答数据
    raw_campus_qa = [
        {"question": "图书馆开放时间？", "answer": "周一至周五8:00-22:00，周末9:00-21:00"},
        {"question": "食堂有几层？", "answer": "3层"},
        {"question": "奖学金怎么申请？", "answer": "每年9月提交申请表至辅导员，需成绩前30%且无违纪"},
        {"question": "校园一卡通在哪里办理？", "answer": "行政楼1楼服务中心，工作日8:30-17:00"},
        {"question": "宿舍限电多少瓦？", "answer": "800瓦，超过会跳闸"},
        {"question": "补考时间是什么时候？", "answer": "每学期开学后第2周，具体看教务处通知"},
        {"question": "篮球场开放时间？", "answer": "6:00-22:00，节假日正常开放"},
        {"question": "选修课有多少门？", "answer": "每学期约50门，涵盖人文、科技、艺术等类别"},
        {"question": "校医院上班时间？", "answer": "8:00-18:00，急诊24小时值班"},
        {"question": "打印店在哪里？", "answer": "图书馆负1楼，营业时间8:00-22:00"}
    ]
    
    # 步骤2：批量增强到1000条
    target_sample_num = 100
    augmented_sft_data = batch_augment_data(raw_campus_qa, target_sample_num)
    
    # 步骤3：保存增强后的数据
    output_file = "campus_qa_augmented_1000.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in augmented_sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 步骤4：验证结果
    print(f"数据增强完成！共生成 {len(augmented_sft_data)} 条样本")
    print("\n=== 增强后数据示例 ===")
    for i in range(5):
        print(f"样本{i+1}：")
        print(f"instruction: {augmented_sft_data[i]['instruction']}")
        print(f"output: {augmented_sft_data[i]['output']}")
        print("-" * 60)
    
    # 标签处理效果示例
    print("\n=== 标签处理效果示例 ===")
    test_answer = "食堂有3个楼层哦！"
    cleaned_answer = clean_output_label(test_answer)
    print(f"原始标签：{test_answer}")
    print(f"清洗后标签：{cleaned_answer}")