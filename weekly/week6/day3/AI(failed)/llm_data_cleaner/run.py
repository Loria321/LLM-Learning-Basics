# from llm_data_cleaner import base_clean,struct_process,text_special,batch_auto,quality_evaluate,utils

'''基础清洗测试'''
from llm_data_cleaner.base_clean import base_clean_pipeline


# 待清洗的原始数据
raw_data = [
    "  这是一段测试文本  ",
    "",
    "这是一段测试文本",  # 与第一条重复（去重后仅保留1条）
    "这是一段长度超出限制的文本........................................................................................................................................"
]

# 执行清洗
cleaned_data = base_clean_pipeline(raw_data)
print("清洗后数据：", cleaned_data)
# 输出：['这是一段测试文本']

'''文本测试'''
from llm_data_cleaner.text_special import text_special_pipeline

# 待处理文本
text = ["Hello😜！这是一段包含英文、emoji和全角标点的文本：【测试】～"]

# 执行处理
processed_text = text_special_pipeline(text,filter_sensitive= True,do_split=True,remove_stopwords_flag= True)
print("处理后文本：", processed_text)
# 输出："！这是一段包含、和全角标点的文本：（测试）—"（标点标准化+过滤英文/emoji）

'''结构化数据测试'''
from llm_data_cleaner.struct_process import struct_process_pipeline
from llm_data_cleaner import utils


# 1. 加载原始结构化数据（JSONL文件，每行一个JSON对象）
raw_data = utils.read_file(r"raw_data\test_data_1.json")
# raw_data = struct_process_pipeline(raw_data)

# 2. 对执行清洗
cleaned_data = base_clean_pipeline(raw_data)
# 3. 导出清洗后的数据（标准化JSONL格式）
cleaned_data = struct_process_pipeline(cleaned_data)
utils.save_file(cleaned_data, r"cleaned_data\1.json")