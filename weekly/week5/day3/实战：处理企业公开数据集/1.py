import os
import json
import pandas as pd
import jsonlines
from typing import Dict, List, Tuple, Optional

# ====================== 1. 全局配置（匹配实际字段名） ======================
FIELD_CONFIG = {
    "duplicate_field_groups": {
        "question_title": ["问题标题", "风控问题名称", "问题名称"],
        "risk_type": ["风控类型", "问题分类", "风险类别"]
    },
    "field_mapping": {
        "question_title": "question",
        "risk_type": "category",
        "问答详情": "question_detail",
        "answer": "answer"
    },
    "keep_fields_whitelist": ["question", "category", "answer"]
}

# 数据处理报告记录器
PROCESS_REPORT = {
    "数据集信息": {},
    "处理步骤记录": [],
    "最终结果汇总": {}
}

# ====================== 新增：辅助函数 - 去除重复列名（核心解决多列问题） ======================
def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    去除DataFrame中的重复列名，保留每列的第一次出现，删除后续重复列
    """
    # 保留不重复的列名（按首次出现顺序）
    unique_columns = []
    for col in df.columns:
        if col not in unique_columns:
            unique_columns.append(col)
    # 重新构建DataFrame，仅保留不重复列
    df_unique = df[unique_columns].copy()
    
    # 记录去重信息（可选）
    duplicate_count = len(df.columns) - len(unique_columns)
    if duplicate_count > 0:
        print(f"已自动去除 {duplicate_count} 个重复列，当前字段列表：{list(df_unique.columns)}")
    return df_unique

# ====================== 2. 嵌套字段解析模块（适配实际字段名"问答详情"，避免重复列） ======================
def parse_nested_json_field(df: pd.DataFrame, nested_field: str = "问答详情") -> pd.DataFrame:
    if nested_field not in df.columns:
        print(f"无嵌套字段 {nested_field}，跳过解析")
        PROCESS_REPORT["处理步骤记录"].append({
            "步骤": "嵌套字段解析",
            "状态": "跳过（无对应字段）",
            "行数变化": f"{len(df)} → {len(df)}",
            "字段数变化": f"{len(df.columns)} → {len(df.columns)}"
        })
        return df
    
    df["parsed_question"] = None
    df["parsed_answer"] = None
    
    error_count = 0
    for idx, row in df.iterrows():
        nested_content = row[nested_field]
        if pd.isna(nested_content):
            continue
        try:
            nested_json = json.loads(str(nested_content).replace("'", "\""))
            if "question" in nested_json:
                df.loc[idx, "parsed_question"] = nested_json["question"]
            if "answer" in nested_json:
                df.loc[idx, "parsed_answer"] = nested_json["answer"]
        except Exception as e:
            error_count += 1
            continue
    
    # 【优化】避免重复列：先判断目标列是否存在，再填充（不再手动新增question列）
    if "question_title" in df.columns:
        df["question_title"] = df["parsed_question"].fillna(df["question_title"])
    if "answer" in df.columns:
        df["answer"] = df["parsed_answer"].fillna(df["answer"])
    
    # 删除临时解析字段
    df = df.drop(columns=["parsed_question", "parsed_answer"])
    
    # 去除重复列（提前规避后续索引异常）
    df = remove_duplicate_columns(df)
    
    PROCESS_REPORT["处理步骤记录"].append({
        "步骤": "嵌套字段解析",
        "状态": f"完成（解析失败 {error_count} 条）",
        "行数变化": f"{len(df)} → {len(df)}",
        "字段数变化": f"{len(df.columns)-2} → {len(df.columns)}"
    })
    print(f"嵌套字段解析完成，解析失败 {error_count} 条数据，当前字段列表：{list(df.columns)}")
    return df

# ====================== 3. 数据变化记录辅助函数 ======================
def record_process_step(step_name: str, before_df: pd.DataFrame, after_df: pd.DataFrame, status: str = "完成"):
    step_record = {
        "步骤": step_name,
        "状态": status,
        "行数变化": f"{len(before_df)} → {len(after_df)}",
        "字段数变化": f"{len(before_df.columns)} → {len(after_df.columns)}"
    }
    PROCESS_REPORT["处理步骤记录"].append(step_record)

# ====================== 4. 重复字段合并 ======================
def merge_duplicate_fields(df: pd.DataFrame, duplicate_config: Dict[str, List[str]]) -> pd.DataFrame:
    before_df = df.copy()
    for target_field, duplicate_fields in duplicate_config.items():
        existing_duplicate_fields = [f for f in duplicate_fields if f in df.columns]
        if not existing_duplicate_fields:
            continue
        
        df[target_field] = None
        for field in existing_duplicate_fields:
            df[target_field] = df[target_field].fillna(df[field])
        
        df = df.drop(columns=existing_duplicate_fields)
    
    # 去除重复列
    df = remove_duplicate_columns(df)
    
    record_process_step("重复字段合并", before_df, df)
    print(f"重复字段合并完成，当前字段列表：{list(df.columns)}")
    return df

# ====================== 5. 字段标准化映射 ======================
def standardize_field_name(df: pd.DataFrame, field_mapping: Dict[str, str]) -> pd.DataFrame:
    before_df = df.copy()
    existing_mappable_fields = [f for f in field_mapping.keys() if f in df.columns]
    if existing_mappable_fields:
        df.rename(columns={f: field_mapping[f] for f in existing_mappable_fields}, inplace=True)
    
    # 去除重复列（核心：避免映射后出现重复的question/category列）
    df = remove_duplicate_columns(df)
    
    record_process_step("字段标准化映射", before_df, df)
    print(f"字段名标准化完成，映射后字段列表：{list(df.columns)}")
    return df

# ====================== 6. 冗余字段删除 ======================
def delete_redundant_fields(df: pd.DataFrame, keep_whitelist: List[str]) -> pd.DataFrame:
    before_df = df.copy()
    redundant_fields = [col for col in df.columns if col not in keep_whitelist]
    if redundant_fields:
        df = df.drop(columns=redundant_fields)
    
    for keep_field in keep_whitelist:
        if keep_field not in df.columns:
            df[keep_field] = None
    
    # 去除重复列
    df = remove_duplicate_columns(df)
    
    record_process_step("冗余字段删除", before_df, df, status=f"完成（删除 {len(redundant_fields)} 个冗余字段）")
    print(f"冗余字段处理完成，最终保留字段：{list(df.columns)}")
    return df

# ====================== 7. 数据读取 ======================
def read_structured_data(file_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    file_suffix = os.path.splitext(file_path)[-1].lower()
    try:
        if file_suffix == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8")
        elif file_suffix == ".json":
            df = pd.read_json(file_path, encoding="utf-8")
        elif file_suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            print(f"错误：不支持的文件格式 {file_suffix}")
            return None
        
        # 去除重复列（从源头规避问题）
        df = remove_duplicate_columns(df)
        
        PROCESS_REPORT["数据集信息"] = {
            "文件路径": file_path,
            "文件格式": file_suffix,
            "原始行数": len(df),
            "原始字段数": len(df.columns),
            "原始字段列表": list(df.columns)
        }
        print(f"成功读取 {file_path}，数据行数：{len(df)}，原始字段：{list(df.columns)}")
        return df
    except Exception as e:
        print(f"读取文件失败：{str(e)}")
        return None

# ====================== 8. 数据清洗（核心修复：强化.str操作容错，确保单列Series） ======================
def clean_risk_qa_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    金融风控问答数据清洗逻辑（核心修复：确保.str操作仅作用于单列Series，解决索引返回DataFrame问题）
    """
    before_df = df.copy()
    keep_fields = FIELD_CONFIG["keep_fields_whitelist"]
    
    # 第一步：先去除重复列（从根源确保每列唯一）
    df = remove_duplicate_columns(df)
    
    # 步骤1：删除完全重复行（保留第一条，避免数据丢失）
    if all(col in df.columns for col in ["question", "category"]):
        df = df.drop_duplicates(subset=["question", "category"], keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    
    # 步骤2：缺失值处理（必填字段：question/category/answer，优化填充逻辑）
    required_fields = ["question", "category", "answer"]
    existing_required_fields = [f for f in required_fields if f in df.columns]
    
    # 先填充必填字段空值，再删除完全缺失的行（确保后续转换字符串无异常）
    for field in existing_required_fields:
        # 第一步：用"无相关内容"填充NaN/None，避免转换字符串后出现"nan"
        df[field] = df[field].fillna("无相关内容")
    df = df.dropna(subset=existing_required_fields)
    
    # 步骤3：字段内容清洗（强化容错，确保单列Series）
    target_clean_fields = ["question", "answer", "category"]
    for field in target_clean_fields:
        # 多重容错：1.字段存在 2.是单列Series 3.非空
        if field not in df.columns:
            continue
        if not isinstance(df[field], pd.Series):
            continue
        if df[field].empty:
            continue
        
        # 分步处理字符串（避免链式调用导致的异常）
        df[field] = df[field].astype(str)
        df[field] = df[field].str.strip()
        df[field] = df[field].replace(["", "nan"], "无相关内容")
    
    # 步骤4：字段长度限制（核心修复：强化容错，确保.str操作合法）
    target_truncate_fields = [("question", 200), ("answer", 1000)]
    for field, max_length in target_truncate_fields:
        # 多重容错：确保字段存在、是Series、是字符串类型
        if field not in df.columns:
            continue
        if not isinstance(df[field], pd.Series):
            continue
        if not pd.api.types.is_string_dtype(df[field]):
            df[field] = df[field].astype(str)
        
        # 安全执行字符串截断（.str[:max_length]）
        df[field] = df[field].str[:max_length]
    
    # 最后再去重列，确保输出干净
    df = remove_duplicate_columns(df)
    
    # 记录报告
    record_process_step("数据清洗（问答场景）", before_df, df, status=f"完成（删除重复/缺失行 {len(before_df)-len(df)} 条）")
    print(f"数据清洗完成，剩余数据行数：{len(df)}")
    return df

# ====================== 9. 数据校验 ======================
def validate_risk_qa_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors = []
    keep_fields = FIELD_CONFIG["keep_fields_whitelist"]
    
    # 先去除重复列
    df = remove_duplicate_columns(df)
    
    # 1. 字段完整性校验
    missing_fields = [f for f in keep_fields if f not in df.columns or df[f].isna().sum() > 0]
    if missing_fields:
        errors.append(f"核心字段存在缺失：{missing_fields}")
    
    # 2. 字段类型校验
    if "question" in df.columns and not pd.api.types.is_string_dtype(df["question"]):
        errors.append("question 字段必须为字符串类型")
    if "answer" in df.columns and not pd.api.types.is_string_dtype(df["answer"]):
        errors.append("answer 字段必须为字符串类型")
    
    # 3. 业务规则校验（问答内容非空）
    empty_question = df[df["question"] == "无相关内容"].shape[0] if "question" in df.columns else 0
    empty_answer = df[df["answer"] == "无相关内容"].shape[0] if "answer" in df.columns else 0
    if empty_question > 0:
        errors.append(f"发现 {empty_question} 条问题内容为空的数据")
    if empty_answer > 0:
        errors.append(f"发现 {empty_answer} 条回答内容为空的数据")
    
    # 记录报告
    is_valid = len(errors) == 0
    PROCESS_REPORT["处理步骤记录"].append({
        "步骤": "数据校验",
        "状态": "通过" if is_valid else f"未通过（发现 {len(errors)} 个问题）",
        "行数变化": f"{len(df)} → {len(df)}",
        "字段数变化": f"{len(df.columns)} → {len(df.columns)}",
        "详细信息": "无异常" if is_valid else "; ".join(errors)
    })
    
    if not is_valid:
        print("数据校验发现以下问题：")
        for err in errors:
            print(f"- {err}")
    else:
        print("数据校验通过")
    return is_valid, errors

# ====================== 10. JSONL转换 ======================
def convert_to_risk_qa_jsonl(df: pd.DataFrame, output_path: str) -> bool:
    before_df = df.copy()
    # 先去除重复列
    df = remove_duplicate_columns(df)
    
    finetune_data = []
    for _, row in df.iterrows():
        # 容错：字段不存在时填充默认值
        category = row.get("category", "未知分类")
        question = row.get("question", "无相关问题")
        answer = row.get("answer", "无相关回答").strip()
        
        prompt = f"金融风控问题（{category}）：{question}"
        finetune_data.append({
            "prompt": prompt,
            "completion": answer
        })
    
    try:
        with open(output_path, mode="w", encoding="utf-8") as f:
            writer = jsonlines.Writer(f)
            writer.write_all(finetune_data)
        
        PROCESS_REPORT["最终结果汇总"] = {
            "输出JSONL路径": output_path,
            "最终有效数据行数": len(finetune_data),
            "最终保留字段": FIELD_CONFIG["keep_fields_whitelist"],
            "微调数据格式": "prompt-completion（金融风控问答）"
        }
        
        record_process_step("JSONL格式转换与输出", before_df, df, status=f"完成（生成 {len(finetune_data)} 条微调数据）")
        print(f"成功输出JSONL文件：{output_path}，共 {len(finetune_data)} 条数据")
        return True
    except Exception as e:
        print(f"输出JSONL失败：{str(e)}")
        return False

# ====================== 11. 生成数据处理报告 ======================
def generate_process_report(report_path: str = "金融风控问答数据处理报告.txt"):
    try:
        with open(report_path, mode="w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("          金融风控问答数据集处理报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1.  数据集基础信息\n")
            f.write("-" * 30 + "\n")
            for key, value in PROCESS_REPORT["数据集信息"].items():
                f.write(f"{key}：{value}\n")
            f.write("\n")
            
            f.write("2.  处理步骤详细记录\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'步骤名称':<20} {'状态':<30} {'行数变化':<15} {'字段数变化':<15}\n")
            f.write("-" * 90 + "\n")
            for step in PROCESS_REPORT["处理步骤记录"]:
                f.write(f"{step['步骤']:<20} {step['状态']:<30} {step['行数变化']:<15} {step['字段数变化']:<15}\n")
            f.write("\n")
            
            f.write("3.  最终结果汇总\n")
            f.write("-" * 30 + "\n")
            for key, value in PROCESS_REPORT["最终结果汇总"].items():
                f.write(f"{key}：{value}\n")
            f.write("\n")
            
            f.write("4.  总结与备注\n")
            f.write("-" * 30 + "\n")
            original_rows = PROCESS_REPORT["数据集信息"].get("原始行数", 0)
            final_rows = PROCESS_REPORT["最终结果汇总"].get("最终有效数据行数", 0)
            data_retention_rate = (final_rows / original_rows * 100) if original_rows > 0 else 0
            f.write(f"1.  数据留存率：{data_retention_rate:.2f}%（{original_rows} → {final_rows}）\n")
            f.write(f"2.  输出文件可直接用于大模型金融风控问答任务微调\n")
            f.write(f"3.  若需优化效果，可调整字段配置或清洗规则\n")
        
        print(f"\n数据处理报告已生成：{report_path}")
        return True
    except Exception as e:
        print(f"生成报告失败：{str(e)}")
        return False

# ====================== 12. 全流程主函数 ======================
def risk_qa_data_pipeline(input_file: str, output_jsonl: str, report_path: str = "金融风控问答数据处理报告.txt") -> bool:
    # 步骤1：读取Excel数据集
    df = read_structured_data(input_file)
    if df is None:
        return False
    
    # 步骤2：合并重复字段
    df = merge_duplicate_fields(df, FIELD_CONFIG["duplicate_field_groups"])
    if df.empty:
        print("错误：合并重复字段后无数据")
        return False
    
    # 步骤3：解析嵌套JSON字段
    df = parse_nested_json_field(df)
    if df.empty:
        print("错误：嵌套字段解析后无数据")
        return False
    
    # 步骤4：字段标准化映射
    df = standardize_field_name(df, FIELD_CONFIG["field_mapping"])
    if df.empty:
        print("错误：字段标准化后无数据")
        return False
    
    # 步骤5：删除冗余字段
    df = delete_redundant_fields(df, FIELD_CONFIG["keep_fields_whitelist"])
    if df.empty:
        print("错误：删除冗余字段后无数据")
        return False
    
    # 步骤6：金融场景数据清洗
    df_cleaned = clean_risk_qa_data(df)
    if df_cleaned.empty:
        print("错误：清洗后无数据")
        return False
    
    # 步骤7：数据校验
    is_valid, _ = validate_risk_qa_data(df_cleaned)
    if not is_valid:
        print("警告：数据校验未通过，但仍尝试输出")
    
    # 步骤8：转换为JSONL格式
    jsonl_result = convert_to_risk_qa_jsonl(df_cleaned, output_jsonl)
    if not jsonl_result:
        print("错误：JSONL输出失败")
        return False
    
    # 步骤9：生成数据处理报告
    report_result = generate_process_report(report_path)
    return jsonl_result and report_result

# ====================== 13. 实战测试 ======================
if __name__ == "__main__":
    # 生成模拟金融风控问答Excel数据集
    test_risk_qa_data = {
        "问题标题": ["信用卡逾期会影响征信吗？", None, "贷款审批需要哪些材料？", None, "信用卡逾期会影响征信吗？"],
        "风控问题名称": [None, "如何判断贷款欺诈？", None, "网贷逾期的后果是什么？", None],
        "风控类型": ["信用卡风控", "贷款反欺诈", "信贷审批", "网贷风控", "信用卡风控"],
        "问答详情": [
            '{"question": "信用卡逾期3天会影响个人征信吗？", "answer": "信用卡逾期3天通常不会影响征信，多数银行有3天宽限期，宽限期内还款视为正常还款。"}',
            '{"question": "如何识别贷款申请中的欺诈行为？", "answer": "可通过核验身份信息、收入证明真实性、交易流水异常、多头借贷记录等方式识别贷款欺诈。"}',
            '{"question": "个人办理银行贷款审批需要准备哪些材料？", "answer": "需要准备身份证、户口本、收入证明、工作证明、征信报告、贷款用途证明等材料。"}',
            None,
            '{"question": "信用卡逾期3天会影响个人征信吗？", "answer": "信用卡逾期3天通常不会影响征信，多数银行有3天宽限期，宽限期内还款视为正常还款。"}'
        ],
        "answer": [None, None, None, "网贷逾期会影响征信、产生高额罚息、面临催收，严重时可能被起诉。", None],
        "录入人员": ["张三", "李四", "王五", "赵六", "张三"],
        "录入日期": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    }
    
    # 生成Excel文件
    test_excel_path = "金融风控问答数据集_模拟.xlsx"
    pd.DataFrame(test_risk_qa_data).to_excel(test_excel_path, index=False, engine="openpyxl")
    print(f"模拟金融风控问答Excel已生成：{test_excel_path}\n")
    
    # 执行全流程数据处理
    output_jsonl_path = "金融风控问答微调数据.jsonl"
    report_path = "金融风控问答数据处理报告.txt"
    result = risk_qa_data_pipeline(test_excel_path, output_jsonl_path, report_path)
    
    if result:
        print("\n" + "=" * 60)
        print("实战全流程完成！已生成：")
        print(f"1.  大模型微调JSONL文件：{output_jsonl_path}")
        print(f"2.  数据处理报告：{report_path}")
    else:
        print("\n实战全流程失败！")