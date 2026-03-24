import json
import csv
from typing import Dict, List, Tuple

def validate_finetune_dataset(
    file_path: str,
    finetune_type: str = "sft"  # 可选：sft / contrast
) -> Dict[str, any]:
    """
    校验大模型微调数据集格式
    
    Args:
        file_path: 数据集文件路径（JSONL/CSV）
        finetune_type: 微调类型，sft（监督微调）/contrast（对比学习）
    
    Returns:
        校验结果字典，包含：
        - is_valid: 是否通过校验（bool）
        - errors: 错误列表（每个元素包含行号、错误原因）
        - total_lines: 总行数
        - valid_lines: 有效行数
    """
    # 1. 定义字段规则
    field_rules = {
        "sft": {
            "required_fields": ["instruction", "input", "output"],
            "field_types": str
        },
        "contrast": {
            "required_fields": ["prompt", "chosen", "rejected"],
            "field_types": str
        }
    }
    if finetune_type not in field_rules:
        return {
            "is_valid": False,
            "errors": [{"line": 0, "error": f"微调类型错误，仅支持sft/contrast，输入为{finetune_type}"}],
            "total_lines": 0,
            "valid_lines": 0
        }
    required_fields = field_rules[finetune_type]["required_fields"]
    field_type = field_rules[finetune_type]["field_types"]

    # 2. 初始化结果
    result = {
        "is_valid": True,
        "errors": [],
        "total_lines": 0,
        "valid_lines": 0
    }

    # 3. 识别文件格式并校验
    try:
        # 判断文件格式（通过后缀）
        if file_path.endswith((".jsonl", ".jl")):
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    result["total_lines"] += 1
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    try:
                        # 解析JSON行
                        data = json.loads(line)
                        # 校验字段是否齐全
                        missing_fields = [f for f in required_fields if f not in data]
                        if missing_fields:
                            result["errors"].append({
                                "line": line_num,
                                "error": f"缺失字段：{missing_fields}"
                            })
                            continue
                        # 校验字段类型
                        type_errors = []
                        for field in required_fields:
                            value = data[field]
                            if not isinstance(value, field_type):
                                type_errors.append(f"{field}（值：{value}）类型应为{field_type}，实际为{type(value)}")
                        if type_errors:
                            result["errors"].append({
                                "line": line_num,
                                "error": f"字段类型错误：{'; '.join(type_errors)}"
                            })
                            continue
                        # 对比学习额外校验：chosen/rejected不可为空
                        if finetune_type == "contrast":
                            if not data["chosen"].strip() or not data["rejected"].strip():
                                result["errors"].append({
                                    "line": line_num,
                                    "error": "对比学习中chosen/rejected不可为空字符串"
                                })
                                continue
                        result["valid_lines"] += 1
                    except json.JSONDecodeError as e:
                        result["errors"].append({
                            "line": line_num,
                            "error": f"JSON解析错误：{str(e)}"
                        })
        elif file_path.endswith(".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # 校验表头
                header = reader.fieldnames
                missing_fields = [f for f in required_fields if f not in header]
                if missing_fields:
                    result["errors"].append({
                        "line": 1,
                        "error": f"CSV表头缺失字段：{missing_fields}"
                    })
                # 校验每行数据
                for line_num, row in enumerate(reader, 2):  # 行号从2开始（表头是1）
                    result["total_lines"] += 1
                    # 校验字段类型
                    type_errors = []
                    for field in required_fields:
                        value = row.get(field, "")
                        if not isinstance(value, field_type):
                            type_errors.append(f"{field}（值：{value}）类型应为{field_type}，实际为{type(value)}")
                    if type_errors:
                        result["errors"].append({
                            "line": line_num,
                            "error": f"字段类型错误：{'; '.join(type_errors)}"
                        })
                        continue
                    # 对比学习额外校验
                    if finetune_type == "contrast":
                        if not row["chosen"].strip() or not row["rejected"].strip():
                            result["errors"].append({
                                "line": line_num,
                                "error": "对比学习中chosen/rejected不可为空字符串"
                            })
                            continue
                    result["valid_lines"] += 1
        else:
            result["is_valid"] = False
            result["errors"].append({
                "line": 0,
                "error": "仅支持JSONL(.jsonl/.jl)和CSV(.csv)格式"
            })
    except FileNotFoundError:
        result["is_valid"] = False
        result["errors"].append({
            "line": 0,
            "error": f"文件不存在：{file_path}"
        })
    except UnicodeDecodeError:
        result["is_valid"] = False
        result["errors"].append({
            "line": 0,
            "error": "文件编码错误，推荐使用UTF-8编码"
        })
    except Exception as e:
        result["is_valid"] = False
        result["errors"].append({
            "line": 0,
            "error": f"未知错误：{str(e)}"
        })

    # 最终判断是否通过
    result["is_valid"] = len(result["errors"]) == 0
    return result

# ---------------------- 测试示例 ----------------------
if __name__ == "__main__":
    # 测试SFT格式的JSONL文件
    sft_jsonl_path = "sft_data.jsonl"
    # 先创建测试文件
    with open(sft_jsonl_path, "w", encoding="utf-8") as f:
        f.write('{"instruction": "计算1+2", "input": "", "output": "3"}\n')
        f.write('{"instruction": "解释特征工程", "input": "大模型微调", "output": 123}\n')  # 类型错误
        f.write('{"instruction": "缺失字段测试", "input": ""}\n')  # 缺失output

    # 执行校验
    sft_result = validate_finetune_dataset(sft_jsonl_path, finetune_type="sft")
    print("=== SFT JSONL 校验结果 ===")
    print(f"是否通过：{sft_result['is_valid']}")
    print(f"总行数：{sft_result['total_lines']}，有效行数：{sft_result['valid_lines']}")
    for error in sft_result["errors"]:
        print(f"行{error['line']}：{error['error']}")

    # 测试对比学习格式的CSV文件
    contrast_csv_path = "contrast_data.csv"
    with open(contrast_csv_path, "w", encoding="utf-8") as f:
        f.write("prompt,chosen,rejected\n")
        f.write("推荐数据工程书籍,《数据密集型应用系统设计》,随便找一本\n")
        f.write("写短文,,不好的回答\n")  # chosen为空
    contrast_result = validate_finetune_dataset(contrast_csv_path, finetune_type="contrast")
    print("\n=== 对比学习 CSV 校验结果 ===")
    print(f"是否通过：{contrast_result['is_valid']}")
    for error in contrast_result["errors"]:
        print(f"行{error['line']}：{error['error']}")