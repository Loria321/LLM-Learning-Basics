# 基础清洗
from .base_clean import (
    remove_duplicates,
    remove_null_values,
    remove_special_chars,
    convert_case,
    base_clean_pipeline
)

# 文本专项处理
from .text_special import (
    load_stopwords,
    split_text,
    remove_stopwords,
    filter_sensitive_words,
    filter_text_length,
    convert_traditional_simplified,
    text_special_pipeline
)

# 结构化处理
from .struct_process import (
    parse_json_str,
    extract_structured_data,
    struct_data_validate,
    text_to_struct,
    struct_process_pipeline
)

# 批量自动化
from .batch_auto import (
    process_single_file,
    batch_process_files,
    batch_summary_report
)

# 质量评估
from .quality_evaluate import (
    calculate_text_smoothness,
    calculate_duplicate_rate,
    calculate_keyword_coverage,
    calculate_data_completeness,
    generate_quality_report
)

# 通用工具
from .utils import read_file, save_file, get_file_list

# 版本号
__version__ = "1.0.0"

# 导出核心功能列表
__all__ = [
    # 基础清洗
    "remove_duplicates",
    "remove_null_values",
    "remove_special_chars",
    "convert_case",
    "base_clean_pipeline",
    # 文本专项
    "load_stopwords",
    "split_text",
    "remove_stopwords",
    "filter_sensitive_words",
    "filter_text_length",
    "convert_traditional_simplified",
    "text_special_pipeline",
    # 结构化处理
    "parse_json_str",
    "extract_structured_data",
    "struct_data_validate",
    "text_to_struct",
    "struct_process_pipeline",
    # 批量自动化
    "process_single_file",
    "batch_process_files",
    "batch_summary_report",
    # 质量评估
    "calculate_text_smoothness",
    "calculate_duplicate_rate",
    "calculate_keyword_coverage",
    "calculate_data_completeness",
    "generate_quality_report",
    # 通用工具
    "read_file",
    "save_file",
    "get_file_list"
]