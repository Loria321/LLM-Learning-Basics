import json
import csv
import random
import re
from typing import List, Dict, Any, Optional

# ======================== 1. 全局配置（可根据场景修改） ========================
class Config:
    # 特征对齐配置
    INSTRUCTION_TEMPLATE = "请回答以下金融风控问题：{question}"  # 指令模板
    MAX_TEXT_LENGTH = 512  # 文本最大长度（字）
    MIN_TEXT_LENGTH = 10   # 文本最小长度（字）
    PAD_CHAR = "。"        # 补齐字符
    
    # 数据增强配置
    TARGET_SAMPLE_NUM = 100  # 最终输出样本数量
    SYNONYM_REPLACE_TIMES = 2  # 单样本同义词替换次数（1-2次）
    
    # 金融风控场景同义词表（可扩展）
    FINANCE_SYNONYM_MAP = {
        "风控": ["风险控制", "风险管控"],
        "逾期": ["超期", "过期", "未按时还款"],
        "授信": ["信用授予", "额度授予"],
        "征信": ["信用报告", "征信报告"],
        "负债率": ["负债比例", "债务比率"],
        "催收": ["欠款催收", "账款回收"],
        "坏账": ["不良贷款", "呆账"],
        "额度": ["信用额度", "授信额度"],
        "审核": ["审查", "核审"],
        "还款": ["偿付", "归还欠款"],
        "违约金": ["滞纳金", "违约罚金"],
        "担保": ["保证", "担保抵押"],
        "流水": ["交易流水", "资金流水"],
        "风控模型": ["风险控制模型", "风控算法模型"]
    }
    
    # 标签清洗配置
    TONE_WORDS = ["哦", "呢", "啊", "吧", "啦", "嘛"]  # 需去除的语气词
    TIME_MAPPING = {  # 时间格式统一
        "1天": "1个自然日", "3天": "3个自然日", "7天": "1周",
        "15天": "半个月", "30天": "1个月", "90天": "3个月"
    }
    UNIT_MAPPING = {  # 单位格式统一
        "元": ["圆", "块"], "%": ["百分比"], "万": ["万元整"]
    }

# ======================== 2. 数据加载模块 ========================
def load_raw_data(file_path: str) -> List[Dict[str, str]]:
    """
    加载原始问答数据（支持JSONL/CSV格式）
    Args:
        file_path: 原始数据文件路径（.jsonl/.csv）
    Returns:
        原始数据列表 [{"question": "...", "answer": "..."}]
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
    """
    raw_data = []
    try:
        # 加载JSONL格式
        if file_path.endswith((".jsonl", ".jl")):
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # 校验必填字段
                        if "question" in data and "answer" in data:
                            raw_data.append({
                                "question": data["question"].strip(),
                                "answer": data["answer"].strip()
                            })
                        else:
                            print(f"警告：JSONL第{line_num}行缺失question/answer字段，跳过")
                    except json.JSONDecodeError:
                        print(f"警告：JSONL第{line_num}行JSON解析失败，跳过")
        
        # 加载CSV格式
        elif file_path.endswith(".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # 校验表头
                if "question" not in reader.fieldnames or "answer" not in reader.fieldnames:
                    raise ValueError("CSV文件表头必须包含question和answer字段")
                for line_num, row in enumerate(reader, 2):  # 行号从2开始（表头为1）
                    question = row.get("question", "").strip()
                    answer = row.get("answer", "").strip()
                    if question and answer:
                        raw_data.append({"question": question, "answer": answer})
                    else:
                        print(f"警告：CSV第{line_num}行question/answer为空，跳过")
        
        else:
            raise ValueError(f"不支持的文件格式：{file_path}，仅支持JSONL(.jsonl/.jl)和CSV(.csv)")
        
        print(f"原始数据加载完成，共加载有效样本 {len(raw_data)} 条")
        return raw_data
    
    except FileNotFoundError:
        raise FileNotFoundError(f"原始数据文件不存在：{file_path}")
    except Exception as e:
        raise Exception(f"数据加载失败：{str(e)}")

# ======================== 3. 特征对齐模块 ========================
def standardize_instruction(question: str) -> str:
    """标准化指令表述（适配金融风控场景）"""
    if not isinstance(question, str):
        raise ValueError(f"问题文本必须为字符串，当前类型：{type(question)}")
    return Config.INSTRUCTION_TEMPLATE.format(question=question.strip())

def adjust_text_length(text: str) -> str:
    """调整文本长度：截断过长文本，补齐过短文本"""
    if not isinstance(text, str):
        text = ""
    text = text.strip()
    text_len = len(text)
    
    # 截断过长文本
    if text_len > Config.MAX_TEXT_LENGTH:
        adjusted_text = text[:Config.MAX_TEXT_LENGTH]
        print(f"文本过长（{text_len}字），已截断至{Config.MAX_TEXT_LENGTH}字")
    # 补齐过短文本
    elif text_len < Config.MIN_TEXT_LENGTH:
        pad_num = Config.MIN_TEXT_LENGTH - text_len
        adjusted_text = text + Config.PAD_CHAR * pad_num
        print(f"文本过短（{text_len}字），已补齐至{Config.MIN_TEXT_LENGTH}字")
    else:
        adjusted_text = text
    
    return adjusted_text

# ======================== 4. 标签处理模块 ========================
def clean_output_label(answer: str) -> str:
    """清洗标签（保证准确性、一致性）"""
    if not isinstance(answer, str):
        return ""
    
    # 步骤1：基础清洗（去空格、换行、语气词）
    answer = answer.strip().replace("\n", "").replace("\r", "")
    for word in Config.TONE_WORDS:
        answer = answer.replace(word, "")
    
    # 步骤2：统一时间格式
    for old, new in Config.TIME_MAPPING.items():
        answer = answer.replace(old, new)
    
    # 步骤3：统一单位格式
    for target_unit, replace_units in Config.UNIT_MAPPING.items():
        for unit in replace_units:
            answer = answer.replace(unit, target_unit)
    
    # 步骤4：去除无意义字符
    answer = re.sub(r"\s+", "", answer)  # 去除所有空格
    return answer

# ======================== 5. 数据增强模块 ========================
def synonym_replacement(text: str) -> str:
    """金融风控场景同义词替换"""
    if not text:
        return text
    
    # 随机选择1-N个词替换
    replace_times = random.randint(1, Config.SYNONYM_REPLACE_TIMES)
    replaced_text = text
    synonym_items = list(Config.FINANCE_SYNONYM_MAP.items())
    random.shuffle(synonym_items)
    
    replace_count = 0
    for original_word, syn_list in synonym_items:
        if replace_count >= replace_times:
            break
        if original_word in replaced_text:
            syn_word = random.choice(syn_list)
            replaced_text = replaced_text.replace(original_word, syn_word, 1)
            replace_count += 1
    
    return replaced_text

def sentence_paraphrase(text: str) -> str:
    """金融风控场景句式变换"""
    if not text:
        return text
    
    # 句式变换规则（适配金融风控）
    paraphrase_rules = [
        (r"^(.*)是什么\？$", r"\1的定义是什么？"),
        (r"^(.*)怎么处理\？$", r"如何处理\1？"),
        (r"^(.*)的标准是多少\？$", r"\1的判定标准是多少？"),
        (r"^(.*)逾期怎么办\？$", r"\1超期后该如何处理？"),
        (r"^(.*)的风控流程\？$", r"\1的风险控制流程是什么？"),
        (r"^(.*)需要哪些材料\？$", r"办理\1需要准备哪些材料？")
    ]
    
    for pattern, repl in paraphrase_rules:
        if re.match(pattern, text):
            return re.sub(pattern, repl, text)
    
    # 无匹配规则时微调语序
    if len(text) > 5 and text.endswith("？"):
        if "风控" in text:
            return text.replace("风控", "") + "的风控规则是什么？"
        elif "逾期" in text:
            return text.replace("逾期", "") + "发生逾期该如何处理？"
    
    return text

def augment_data(raw_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """批量数据增强，扩充到目标样本数"""
    augmented_data = []
    seen_questions = set()  # 去重集合
    
    # 循环增强直到达到目标数量
    while len(augmented_data) < Config.TARGET_SAMPLE_NUM:
        for sample in raw_data:
            if len(augmented_data) >= Config.TARGET_SAMPLE_NUM:
                break
            
            # 单样本增强：生成4个变体（原始+同义词+句式+同义词+句式）
            original_question = sample["question"]
            cleaned_answer = clean_output_label(sample["answer"])
            
            variants = [
                original_question,  # 原始
                synonym_replacement(original_question),  # 同义词
                sentence_paraphrase(original_question),  # 句式变换
                sentence_paraphrase(synonym_replacement(original_question))  # 组合
            ]
            
            # 去重并添加
            for var in variants:
                if len(augmented_data) >= Config.TARGET_SAMPLE_NUM:
                    break
                if var not in seen_questions and var.strip():
                    seen_questions.add(var)
                    augmented_data.append({
                        "question": var,
                        "answer": cleaned_answer
                    })
    
    # 截断到目标数量
    return augmented_data[:Config.TARGET_SAMPLE_NUM]

# ======================== 6. 格式转换+保存模块 ========================
def convert_to_sft_format(augmented_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """转换为SFT微调格式：{"instruction": "...", "input": "", "output": "..."}"""
    sft_data = []
    for item in augmented_data:
        instruction = standardize_instruction(item["question"])
        output = adjust_text_length(item["answer"])  # 调整回答长度
        sft_data.append({
            "instruction": instruction,
            "input": "",  # 金融风控问答无额外输入
            "output": output
        })
    return sft_data

def save_final_data(sft_data: List[Dict[str, str]], output_file: str) -> None:
    """保存最终微调数据集（JSONL格式）"""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"最终数据集已保存至：{output_file}")
        print(f"数据集总条数：{len(sft_data)}")
    except Exception as e:
        raise Exception(f"数据保存失败：{str(e)}")

# ======================== 7. 主函数（整合全流程） ========================
def finetune_feature_engineering(
    raw_data_path: str,
    output_data_path: str = "finance_risk_finetune_data.jsonl"
) -> Dict[str, Any]:
    """
    微调数据集特征工程全流程
    Args:
        raw_data_path: 原始问答数据路径（JSONL/CSV）
        output_data_path: 最终微调数据集保存路径
    Returns:
        处理统计信息
    """
    try:
        # 步骤1：加载原始数据
        print("===== 1. 加载原始数据 =====")
        raw_data = load_raw_data(raw_data_path)
        raw_num = len(raw_data)
        
        # 步骤2：数据增强
        print("\n===== 2. 数据增强 =====")
        augmented_data = augment_data(raw_data)
        augment_num = len(augmented_data)
        
        # 步骤3：转换为SFT格式（含特征对齐）
        print("\n===== 3. 特征对齐+格式转换 =====")
        sft_data = convert_to_sft_format(augmented_data)
        
        # 步骤4：保存最终数据
        print("\n===== 4. 保存最终数据集 =====")
        save_final_data(sft_data, output_data_path)
        
        # 统计信息
        stats = {
            "原始样本数": raw_num,
            "增强后样本数": augment_num,
            "最终输出样本数": len(sft_data),
            "是否达到目标数量": len(sft_data) == Config.TARGET_SAMPLE_NUM,
            "输出路径": output_data_path
        }
        
        print("\n===== 处理完成！统计信息 =====")
        for k, v in stats.items():
            print(f"{k}: {v}")
        
        return stats
    
    except Exception as e:
        print(f"特征工程处理失败：{str(e)}")
        raise

# ======================== 8. 测试：金融风控数据处理 ========================
if __name__ == "__main__":
    # 步骤1：创建测试用金融风控原始数据（JSONL格式）
    test_raw_path = "finance_risk_raw.jsonl"
    with open(test_raw_path, "w", encoding="utf-8") as f:
        # 金融风控原始问答样本（10条）
        raw_samples = [
            {"question": "风控中的逾期定义是什么？", "answer": "逾期指借款人未按借款合同约定的还款日归还贷款本息，超过1天即视为逾期。"},
            {"question": "个人授信额度的审核标准是多少？", "answer": "个人授信额度主要根据征信报告、负债率（低于50%）、收入流水等审核，普通用户额度通常1万-50万。"},
            {"question": "坏账的催收流程怎么处理？", "answer": "坏账催收分3阶段：1. 逾期1-30天：短信提醒；2. 逾期31-90天：电话催收；3. 逾期90天以上：委托第三方或法律诉讼。"},
            {"question": "征信不良会影响授信吗？", "answer": "征信不良（如逾期超3次、负债率超70%）会直接降低授信额度，严重者会被拒绝授信。"},
            {"question": "违约金的计算标准是什么？", "answer": "违约金按未还金额的0.05%/天计算，最高不超过本金的30%，且需在还款时一次性缴纳。"},
            {"question": "风控模型包含哪些维度？", "answer": "风控模型涵盖5个核心维度：身份验证、信用历史、还款能力、负债情况、交易行为，每个维度权重占比20%左右。"},
            {"question": "担保贷款需要哪些材料？", "answer": "担保贷款需准备：身份证、征信报告、收入证明、担保人资质证明、贷款用途说明，材料需加盖公章。"},
            {"question": "流水不足怎么申请授信？", "answer": "流水不足可补充资产证明（房产、车辆）、担保人或提高首付比例，也可申请小额授信（≤5万）。"},
            {"question": "逾期3天会影响征信吗？", "answer": "多数金融机构有3天宽限期，逾期3天内还款不会上报征信，超过3天则会记录不良信用。"},
            {"question": "坏账率的警戒线是多少？", "answer": "金融机构风控要求坏账率不超过3%，超过5%需启动风险预警，调整授信策略。"}
        ]
        for sample in raw_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # 步骤2：执行特征工程全流程
    finetune_feature_engineering(
        raw_data_path=test_raw_path,
        output_data_path="finance_risk_finetune_data_1000.jsonl"
    )
    
    # 步骤3：验证最终数据
    print("\n===== 最终数据集示例 =====")
    with open("finance_risk_finetune_data_1000.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:  # 仅展示前5条
                break
            data = json.loads(line.strip())
            print(f"样本{i+1}：")
            print(f"instruction: {data['instruction']}")
            print(f"output: {data['output']}")
            print(f"output长度：{len(data['output'])}字")
            print("-" * 80)