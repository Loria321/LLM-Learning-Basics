# 基础清洗
from .base_clean import (
    remove_duplicates,
    remove_null_values,
    base_clean_pipeline
)

# 文本专项处理
from .text_clean import (
    load_stopwords,
    split_text,
    remove_stopwords,
    filter_sensitive_words,
    filter_text_length,
    convert_traditional_simplified,
    text_clean_pipeline,
    text_quality_evaluate
)

# 结构化处理
from .struct_clean import (
    handle_high_missing_cols,
    fill_missing_values,
    detect_and_handle_outliers,
    standardize_formats,
    struct_clean_pipeline
)

# 批量自动化
from .batch_auto import (
    process_single_file,
    batch_process_files,
    batch_summary_report
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
    "text_clean_pipeline",
    "text_quality_evaluate",
    # 结构化处理
    "handle_high_missing_cols",
    "fill_missing_values",
    "detect_and_handle_outliers",
    "standardize_formats",
    "struct_clean_pipeline",
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