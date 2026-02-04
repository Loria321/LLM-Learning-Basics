import jieba
import opencc
import re
import pandas as pd
import logging
from typing import List, Union, Optional, Callable, Literal
from pathlib import Path
from functools import lru_cache
from typing_extensions import TypeGuard
import os
from datetime import datetime

# ======================== 新增：日志配置（核心优化点1） ========================

# 1. 定义日志相关路径和文件名（xx采用日期命名，格式：text_clean_20260202.log）
log_dir = r".\logs\text_clean"  # 目标日志目录：logs\text_clean
current_date = datetime.now().strftime("%Y%m%d")  # 获取当前日期，作为xx的替代（更实用）
log_filename = f"text_clean_{current_date}.log"  # 日志文件名
log_full_path = os.path.join(log_dir, log_filename)  # 拼接完整日志路径，兼容跨平台

# 2. 自动创建多级目录（若不存在），避免FileHandler报错
os.makedirs(log_dir, exist_ok=True)

# 3. 配置日志：仅保存到本地文件，取消控制台输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_full_path, encoding="utf-8")]  # 使用拼接后的完整路径
)
logger = logging.getLogger(__name__)

# ======================== 常量定义 ========================
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SENSITIVE_PATH = PROJECT_ROOT / "config" / "sensitive_words.txt"
DEFAULT_STOPWORD_PATH = PROJECT_ROOT / "config" / "stopwords.txt"

DEFAULT_KEEP_CHARS = "，。！？；：\"''（）【】《》a-zA-Z0-9\u4e00-\u9fa5"
EMOJI_PATTERN = re.compile(
    r"""
    [\U0001F600-\U0001F64F]  # 表情符号
    |[\U0001F300-\U0001F5FF] # 符号&象形图
    |[\U0001F680-\U0001F6FF] # 交通&地图符号
    |[\U0001F1E0-\U0001F1FF] # 国旗符号
    """,
    flags=re.UNICODE | re.VERBOSE
)
SPECIAL_NOISE_PATTERN = re.compile(r'★|■|◆|●|△|▲|※|§|№|＃|＆|＄|％|＠|～|｀|＾|｜|＼|／')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

# ======================== 类型别名 ========================
TextInput = Union[str, List[str]]
SplitResult = Union[List[str], List[List[str]]]
DataInput = Union[pd.DataFrame, List[str]]
ConvertType = Literal["s2t", "t2s"]
CleanRule = Callable[[str], str]

# ======================== 通用工具函数 ========================
def is_list_of_str(v: object) -> TypeGuard[List[str]]:
    """类型守卫：判断是否为字符串列表"""
    return isinstance(v, list) and all(isinstance(item, str) for item in v)

def process_batch(
    data: TextInput,
    handler: Callable[[str], str]
) -> TextInput:
    """通用批量处理函数：统一处理单个文本/文本列表"""
    if isinstance(data, str):
        return handler(data)
    elif is_list_of_str(data):
        return [handler(item) for item in data]
    else:
        err_msg = f"输入类型必须为str或List[str]，当前类型：{type(data)}"
        logger.error(err_msg)  # 新增：日志记录
        raise TypeError(err_msg)

# ======================== 缓存装饰器（核心优化点3：优化缓存策略） ========================
@lru_cache(maxsize=2)
def get_opencc_converter(convert_type: ConvertType) -> opencc.OpenCC:
    """缓存opencc实例，避免重复初始化"""
    logger.info(f"初始化opencc转换器，类型：{convert_type}")
    return opencc.OpenCC(convert_type)

# 优化点：1. maxsize改为None（无上限缓存） 2. 新增version参数支持热更新 3. 统一路径为str作为key
@lru_cache(maxsize=None)
def load_stopwords(stopword_path: Optional[str] = None, version: int = 1) -> set:
    """
    加载停用词表（带缓存+异常处理+日志）
    :param stopword_path: 停用词文件路径（自动转换为字符串，兼容Path）
    :param version: 缓存版本号（修改版本号可触发缓存刷新，支持热更新）
    """
    # 统一路径为字符串（兼容Path对象）
    stopword_path_str = str(stopword_path) if stopword_path is not None else None
    
    if stopword_path_str is None:
        default_stopwords = {"的", "了", "是", "我", "你", "他", "她", "它", "们", "在", "有", "就", "不", "和", "也", "都"}
        logger.info(f"加载默认停用词表，共{len(default_stopwords)}个停用词")
        return default_stopwords
    
    try:
        with open(stopword_path_str, "r", encoding="utf-8") as f:
            stopwords = set([line.strip() for line in f if line.strip()])
        # 新增：空文件日志提示
        if not stopwords:
            logger.warning(f"停用词文件为空：{stopword_path_str}")
        else:
            logger.info(f"加载自定义停用词表成功，路径：{stopword_path_str}，共{len(stopwords)}个停用词")
        return stopwords
    except FileNotFoundError:
        err_msg = f"停用词文件不存在：{stopword_path_str}"
        logger.error(err_msg)
        raise
    except PermissionError:
        err_msg = f"无权限读取停用词文件：{stopword_path_str}"
        logger.error(err_msg)
        raise
    except UnicodeDecodeError:  # 新增：编码错误处理
        err_msg = f"停用词文件编码错误（请使用UTF-8）：{stopword_path_str}"
        logger.error(err_msg)
        raise
    except Exception as e:  # 新增：兜底异常捕获
        err_msg = f"加载停用词文件失败，路径：{stopword_path_str}，异常：{str(e)}"
        logger.error(err_msg, exc_info=True)
        raise

# 同load_stopwords，优化缓存策略+补充异常+日志
@lru_cache(maxsize=None)
def load_sensitive_words(sensitive_path: Optional[str] = None, version: int = 1) -> set:
    """
    加载敏感词表（带缓存+异常处理+日志）
    :param sensitive_path: 敏感词文件路径（自动转换为字符串，兼容Path）
    :param version: 缓存版本号（修改版本号可触发缓存刷新）
    """
    sensitive_path_str = str(sensitive_path) if sensitive_path is not None else None
    
    if sensitive_path_str is None:
        logger.info("未指定敏感词路径，返回空敏感词集合")
        return set()
    
    try:
        with open(sensitive_path_str, "r", encoding="utf-8") as f:
            sensitive_words = set([line.strip() for line in f if line.strip()])
        if not sensitive_words:
            logger.warning(f"敏感词文件为空：{sensitive_path_str}")
        else:
            logger.info(f"加载自定义敏感词表成功，路径：{sensitive_path_str}，共{len(sensitive_words)}个敏感词")
        return sensitive_words
    except FileNotFoundError:
        err_msg = f"敏感词文件不存在：{sensitive_path_str}"
        logger.error(err_msg)
        raise
    except PermissionError:
        err_msg = f"无权限读取敏感词文件：{sensitive_path_str}"
        logger.error(err_msg)
        raise
    except UnicodeDecodeError:
        err_msg = f"敏感词文件编码错误（请使用UTF-8）：{sensitive_path_str}"
        logger.error(err_msg)
        raise
    except Exception as e:
        err_msg = f"加载敏感词文件失败，路径：{sensitive_path_str}，异常：{str(e)}"
        logger.error(err_msg, exc_info=True)
        raise

# ======================== 文本清理核心函数 ========================
def clean_text_black_white(
    text: TextInput,
    keep_chars: str = DEFAULT_KEEP_CHARS,
    replace_char: str = "",
    optimize_format: bool = True,
    custom_black_rules: Optional[List[CleanRule]] = None
) -> TextInput:
    """黑白名单结合的文本去噪方案"""
    @lru_cache(maxsize=10)
    def _get_whitelist_pattern(keep_chars: str) -> re.Pattern:
        # 仅转义字符集中的特殊字符：^ (开头)、] (结束)、\ (反斜杠)
        # 保留 - 作为字符范围符，避免破坏 a-zA-Z、\u4e00-\u9fa5 等范围
        escaped_keep_chars = keep_chars
        # 转义反斜杠
        escaped_keep_chars = escaped_keep_chars.replace("\\", "\\\\")
        # 转义闭合方括号（字符集的结束符）
        escaped_keep_chars = escaped_keep_chars.replace("]", "\\]")
        # 转义开头的^（字符集的取反符，若存在）
        if escaped_keep_chars.startswith("^"):
            escaped_keep_chars = "\\" + escaped_keep_chars
        # 生成白名单正则（仅过滤不在keep_chars中的字符）
        return re.compile(f"[^{escaped_keep_chars}]")
    
    def _blacklist_clean_single(txt: str) -> str:
        if pd.isna(txt) or txt is None or txt.strip() == "":
            logger.debug("输入为空/NaN，返回空字符串")
            return ""
        
        try:
            txt = HTML_TAG_PATTERN.sub('', txt)
            txt = EMOJI_PATTERN.sub('', txt)
            txt = SPECIAL_NOISE_PATTERN.sub('', txt)
            
            if custom_black_rules:
                for rule in custom_black_rules:
                    txt = rule(txt)
            
            if optimize_format:
                txt = re.sub(r'\s+', ' ', txt).strip()
                txt = re.sub(r'(\d+) +([年月日])', r'\1\2', txt)
            return txt
        except Exception as e:  # 新增：单文本处理异常捕获
            logger.error(f"文本黑名单清理失败，文本：{txt[:50]}...，异常：{str(e)}", exc_info=True)
            return ""
    
    def _whitelist_clean_single(txt: str) -> str:
        try:
            pattern = _get_whitelist_pattern(keep_chars)
            return pattern.sub(replace_char, txt)
        except Exception as e:
            logger.error(f"文本白名单清理失败，文本：{txt[:50]}...，异常：{str(e)}", exc_info=True)
            return ""
    
    def _clean_single(txt: str) -> str:
        black_cleaned = _blacklist_clean_single(txt)
        white_cleaned = _whitelist_clean_single(black_cleaned)
        return white_cleaned
    
    return process_batch(text, _clean_single)

# ======================== 分词/去停用词 ========================
def split_text(
    text: TextInput,
    cut_all: bool = False,
    tokenizer: Optional[Callable[[str], List[str]]] = None
) -> SplitResult:
    """文本分词（支持自定义分词器）"""
    def _cut(t: str) -> List[str]:
        if tokenizer:
            return tokenizer(t)
        return jieba.lcut(t, cut_all=cut_all)
    
    def _split_single(txt: str) -> List[str]:
        try:
            return _cut(txt.strip()) if txt.strip() else []
        except Exception as e:
            logger.error(f"文本分词失败，文本：{txt[:50]}...，异常：{str(e)}", exc_info=True)
            return []
    
    if isinstance(text, str):
        return _split_single(text)
    elif is_list_of_str(text):
        return [_split_single(item) for item in text]
    else:
        err_msg = f"输入类型必须为str或List[str]，当前类型：{type(text)}"
        logger.error(err_msg)
        raise TypeError(err_msg)

def remove_stopwords(
    words: SplitResult,
    stopwords: Optional[set] = None,
    stopword_path: Optional[str] = None
) -> SplitResult:
    """去除停用词（增强类型校验+日志）"""
    try:
        stopwords = stopwords or load_stopwords(stopword_path)
    except Exception as e:
        logger.error(f"加载停用词失败，路径：{stopword_path}，异常：{str(e)}", exc_info=True)
        raise
    
    def _remove_single(ws: List[str]) -> List[str]:
        return [w for w in ws if w not in stopwords and w.strip()]
    
    if isinstance(words, list) and all(isinstance(w, str) for w in words):
        return _remove_single(words)
    elif isinstance(words, list) and all(isinstance(w, list) for w in words):
        return [_remove_single(w_list) for w_list in words]
    else:
        err_msg = "输入必须为List[str]或List[List[str]]"
        logger.error(err_msg)
        raise TypeError(err_msg)

def filter_sensitive_words(
    text: TextInput,
    sensitive_words: Optional[set] = None,
    sensitive_path: Optional[str] = None,
    replace_char: str = "*",
    full_word_match: bool = False  # 新增：是否全词匹配，默认子串匹配
) -> TextInput:
    """
    敏感词过滤（支持全词匹配/子串匹配+缓存+异常处理）
    :param text: 输入文本/文本列表
    :param sensitive_words: 敏感词集合
    :param sensitive_path: 敏感词文件路径
    :param replace_char: 替换字符
    :param full_word_match: 是否全词匹配（避免子串误匹配）
    :return: 过滤后的文本/文本列表
    """
    try:
        sensitive_words = sensitive_words or load_sensitive_words(sensitive_path)
    except Exception as e:
        logger.error(f"加载敏感词失败，路径：{sensitive_path}，异常：{str(e)}", exc_info=True)
        raise
    
    # 预编译敏感词正则（缓存避免重复编译）
    @lru_cache(maxsize=100)
    def _get_sensitive_pattern(sensitive_word: str) -> re.Pattern:
        escaped_word = re.escape(sensitive_word)
        if full_word_match:
            # 修正：全词匹配 - 敏感词前后不是中文/字母/数字，或在字符串开头/结尾
            pattern = re.compile(
                rf'(?<![\u4e00-\u9fa5a-zA-Z0-9]){escaped_word}(?![\u4e00-\u9fa5a-zA-Z0-9])',
                flags=re.UNICODE
            )
        else:
            # 子串匹配（默认）：只要包含敏感词就替换
            pattern = re.compile(escaped_word, flags=re.UNICODE)
        return pattern   

    def _filter_single(txt: str) -> str:
        if pd.isna(txt) or txt is None or not txt.strip():
            return ""
        try:
            for word in sensitive_words:
                pattern = _get_sensitive_pattern(word)
                # 替换敏感词（无需分组，直接替换）
                txt = pattern.sub(replace_char * len(word), txt)
            return txt
        except Exception as e:
            logger.error(f"敏感词过滤失败，文本：{txt[:50]}...，异常：{str(e)}", exc_info=True)
            return txt
    
    return process_batch(text, _filter_single)

def filter_text_length(
    text: TextInput,
    min_len: int = 1,
    max_len: Optional[int] = None
) -> TextInput:
    """文本长度过滤（添加参数合法性校验+日志）"""
    if min_len < 0:
        err_msg = f"min_len不能为负数，当前值：{min_len}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    if max_len is not None and max_len < min_len:
        err_msg = f"max_len({max_len})不能小于min_len({min_len})"
        logger.error(err_msg)
        raise ValueError(err_msg)
    
    def _filter_single(txt: str) -> str:
        stripped_txt = txt.strip()
        length = len(stripped_txt)
        if length < min_len:
            logger.debug(f"文本长度不足{min_len}，过滤：{stripped_txt[:50]}...")
            return ""
        if max_len and length > max_len:
            logger.debug(f"文本长度超过{max_len}，截断：{stripped_txt[:50]}...")
            return stripped_txt[:max_len]
        return stripped_txt
    
    return process_batch(text, _filter_single)

def convert_traditional_simplified(
    text: TextInput,
    convert_type: ConvertType = "s2t"
) -> TextInput:
    """繁简转换（缓存实例+严格类型约束+日志）"""
    try:
        converter = get_opencc_converter(convert_type)
    except Exception as e:
        logger.error(f"初始化繁简转换器失败，类型：{convert_type}，异常：{str(e)}", exc_info=True)
        raise
    
    def _convert_single(txt: str) -> str:
        try:
            return converter.convert(txt) if txt.strip() else txt
        except Exception as e:
            logger.error(f"繁简转换失败，文本：{txt[:50]}...，异常：{str(e)}", exc_info=True)
            return txt
    
    return process_batch(text, _convert_single)

def text_clean_pipeline(
    data: DataInput,
    col: Optional[str] = None,
    do_split: bool = False,
    remove_stopwords_flag: bool = False,
    filter_sensitive: bool = False,
    filter_length: bool = True,
    min_len: int = 5,
    max_len: int = 1000,
    convert_t2s: bool = False,
    sensitive_path: Optional[str] = str(DEFAULT_SENSITIVE_PATH),
    sensitive_words: Optional[str] = None,
    stopword_path: Optional[str] = str(DEFAULT_STOPWORD_PATH),
    custom_black_rules: Optional[List[CleanRule]] = None,
    full_word_match: bool = False  # 新增：传递给敏感词过滤函数
) -> Union[pd.DataFrame, List[str], List[List[str]]]:
    """文本专项处理流水线（优化扩展性+可配置+日志）"""
    logger.info(f"开始文本清理流水线，输入类型：{type(data)}，分词：{do_split}，敏感词过滤：{filter_sensitive}")
    
    # 1. 输入数据格式统一（原有逻辑不变）
    try:
        if isinstance(data, pd.DataFrame):
            if not col or col not in data.columns:
                err_msg = f"DataFrame必须指定有效列名，当前列：{data.columns}"
                logger.error(err_msg)
                raise ValueError(err_msg)
            text_list = data[col].apply(
                lambda x: "" if pd.isna(x) or str(x).strip() in ["None", "<NA>"] else str(x)
            ).tolist()
        elif is_list_of_str(data):
            text_list = data
        else:
            err_msg = f"仅支持pd.DataFrame或List[str]类型，当前类型：{type(data)}"
            logger.error(err_msg)
            raise TypeError(err_msg)
    except Exception as e:
        logger.error(f"输入数据格式转换失败，异常：{str(e)}", exc_info=True)
        raise
    
    # 2. 核心处理步骤（修改敏感词过滤调用逻辑）
    try:
        if convert_t2s:
            text_list = convert_traditional_simplified(text_list, "t2s")
        text_list = clean_text_black_white(
            text_list,
            optimize_format=True,
            custom_black_rules=custom_black_rules
        )
        if filter_sensitive:
            # 传递full_word_match参数
            text_list = filter_sensitive_words(
                text_list, 
                sensitive_words=sensitive_words,
                sensitive_path=sensitive_path,
                full_word_match=full_word_match
            )
        if filter_length:
            text_list = filter_text_length(text_list, min_len=min_len, max_len=max_len)
        if do_split:
            text_list = split_text(text_list)
            if remove_stopwords_flag:
                text_list = remove_stopwords(text_list, stopword_path=stopword_path)
    except Exception as e:
        logger.error(f"文本清理流水线核心步骤失败，异常：{str(e)}", exc_info=True)
        raise
    
    # 3. 结果回填（原有逻辑不变）
    try:
        if isinstance(data, pd.DataFrame) and col:
            data = data.copy()
            data[col] = text_list
            logger.info(f"文本清理流水线完成，处理后DataFrame行数：{len(data)}")
            return data
        logger.info(f"文本清理流水线完成，处理后列表长度：{len(text_list)}")
        return text_list
    except Exception as e:
        logger.error(f"结果回填失败，异常：{str(e)}", exc_info=True)
        raise

# ======================== 文本质量评估函数（优化版） ========================
def text_quality_evaluate(
    text_series: pd.Series,
    is_cleaned: bool = False,
    stopword_path: Optional[str] = None,
    stopwords: Optional[set] = None
) -> dict:
    """
    文本质量评估函数（优化版：兼容原有代码、复用工具函数、补充日志与异常处理）
    :param text_series: pd.Series，待评估的文本列（清洗前/后）
    :param is_cleaned: bool，是否为清洗后文本（清洗后已分词，用空格分隔）
    :param stopword_path: 停用词文件路径（优先使用传入的stopwords，无则加载该路径停用词）
    :param stopwords: 停用词集合（直接传入，避免重复加载，提升效率）
    :return: dict，结构化评估结果
    """
    # ------------- 常量定义（与原有代码风格统一，便于维护） -------------
    LENGTH_BINS = [0, 10, 20, 50, float('inf')]
    LENGTH_LABELS = ["0-10字", "10-20字", "20-50字", "50字以上"]
    PURE_NUM_PATTERN = re.compile(r'^\d+(\.\d+)?$')  # 纯数字/小数正则
    
    # ------------- 初始化评估结果 -------------
    eval_result = {
        "样本总数": 0,
        "有效样本数": 0,
        "平均字符长度": 0.0,
        "平均词汇数": 0.0,
        "有效词汇占比(%)": 0.0,
        "长度分布": {label: 0 for label in LENGTH_LABELS}
    }
    
    # ------------- 步骤1：输入参数校验 -------------
    try:
        if not isinstance(text_series, pd.Series):
            err_msg = f"输入必须为pd.Series类型，当前类型：{type(text_series)}"
            logger.error(err_msg)
            raise TypeError(err_msg)
        
        eval_result["样本总数"] = len(text_series)
        if eval_result["样本总数"] == 0:
            logger.warning("输入的文本Series为空，返回默认评估结果")
            return eval_result
    except Exception as e:
        logger.error(f"文本质量评估 - 输入参数校验失败，异常：{str(e)}", exc_info=True)
        return eval_result
    
    # ------------- 步骤2：加载停用词（复用原有加载逻辑，避免硬编码） -------------
    try:
        used_stopwords = stopwords or load_stopwords(stopword_path=stopword_path)
    except Exception as e:
        logger.error(f"文本质量评估 - 加载停用词失败，异常：{str(e)}", exc_info=True)
        used_stopwords = set()  # 加载失败时使用空集合，避免后续流程中断
    
    # ------------- 步骤3：过滤空文本（兼容NaN、空字符串、全空格） -------------
    try:
        # 过滤条件：非NaN + 去除前后空格后非空
        valid_filter = (text_series.notna()) & (text_series.astype(str).str.strip() != "")
        valid_text_series = text_series[valid_filter].astype(str)
        valid_count = len(valid_text_series)
        eval_result["有效样本数"] = valid_count
        
        if valid_count == 0:
            logger.warning("过滤后无有效文本样本，返回默认评估结果")
            return eval_result
    except Exception as e:
        logger.error(f"文本质量评估 - 过滤空文本失败，异常：{str(e)}", exc_info=True)
        return eval_result
    
    # ------------- 步骤4：指标1：文本长度分布 + 平均字符长度 -------------
    try:
        # 计算每个有效文本的字符长度
        char_lengths = valid_text_series.apply(lambda x: len(x.strip()))
        eval_result["平均字符长度"] = round(char_lengths.mean(), 2)
        
        # 统计长度分布（兼容无数据的区间，保证返回结果结构完整）
        length_cut = pd.cut(
            char_lengths,
            bins=LENGTH_BINS,
            labels=LENGTH_LABELS,
            right=False,
            include_lowest=True
        )
        length_dist = length_cut.value_counts().sort_index()
        for label in LENGTH_LABELS:
            eval_result["长度分布"][label] = int(length_dist.get(label, 0))
    except Exception as e:
        logger.error(f"文本质量评估 - 计算长度指标失败，异常：{str(e)}", exc_info=True)
        # 保留已有结果，后续指标继续执行（不中断整体流程）
    
    # ------------- 步骤5：指标2：平均词汇数 + 有效词汇占比 -------------
    total_words = 0  # 总词汇数
    valid_words = 0  # 有效词汇数（非停用词、非空、非纯数字/符号）
    
    try:
        for text in valid_text_series:
            text_stripped = text.strip()
            if not text_stripped:
                continue
            
            # 分词逻辑（复用原有工具，清洗前文本复用已有去噪函数，避免重复写正则）
            if is_cleaned:
                # 清洗后已分词，按空格切分
                words = text_stripped.split()
            else:
                # 清洗前：复用已有去噪函数做基础处理，再分词（与原有清洗逻辑一致）
                basic_cleaned_text = clean_text_black_white(text_stripped, optimize_format=True)
                words = jieba.lcut(basic_cleaned_text)
            
            # 统计总词汇数
            current_word_count = len(words)
            total_words += current_word_count
            
            # 统计有效词汇数
            for word in words:
                word_stripped = word.strip()
                is_valid = (
                    word_stripped not in used_stopwords
                    and len(word_stripped) > 0
                    and not PURE_NUM_PATTERN.match(word_stripped)
                )
                if is_valid:
                    valid_words += 1
        
        # 计算最终指标（避免除零错误，边界值保护）
        eval_result["平均词汇数"] = round(total_words / valid_count, 2) if valid_count > 0 else 0.0
        eval_result["有效词汇占比(%)"] = round((valid_words / total_words) * 100, 2) if total_words > 0 else 0.0
        
        logger.info(f"文本质量评估完成：有效样本{valid_count}个，总词汇{total_words}个，有效词汇{valid_words}个")
    except Exception as e:
        logger.error(f"文本质量评估 - 计算词汇指标失败，异常：{str(e)}", exc_info=True)
    
    return eval_result

# ======================== 初始化 ========================
try:
    jieba.initialize()
    logger.info("jieba分词器初始化成功")
except Exception as e:
    logger.error(f"jieba分词器初始化失败，异常：{str(e)}", exc_info=True)
    raise

def _run_builtin_tests():
    """运行内置测试，输出简单测试报告"""
    print("=" * 60)
    print("开始执行内置简单测试...")
    print("=" * 60)
    
    # 测试数据准备
    TEST_TEXT = "Hello😀<p>2025 年</p>！敏感词测试"
    TEST_TEXT_LIST = ["Hello😀2025 年！", "测试敏感词123", "", None, pd.NA]
    TEST_DF = pd.DataFrame({"text": TEST_TEXT_LIST, "other_col": [1, 2, 3, 4, 5]})
    TEST_STOPWORDS = {"的", "是", "测试"}
    TEST_SENSITIVE_WORDS = {"敏感词"}
    test_pass_count = 0
    test_total_count = 0

    # 1. 测试 clean_text_black_white（文本去噪）
    test_total_count += 1
    try:
        cleaned_text = clean_text_black_white(TEST_TEXT)
        # 新增：控制台输出清洗前、清洗后文本，直观展示结果
        print(f"\n📝 测试1 - 文本去噪详情：")
        print(f"   清洗前：{TEST_TEXT}")
        print(f"   清洗后：{cleaned_text}")
        assert cleaned_text == "Hello2025年！敏感词测试", "文本去噪结果不符合预期"
        print("✅ 测试1（文本去噪）：通过")
        test_pass_count += 1
    except AssertionError as e:
        print(f"❌ 测试1（文本去噪）：失败 - {str(e)}")
    except Exception as e:
        print(f"❌ 测试1（文本去噪）：异常 - {str(e)}")

    # 2. 测试 filter_sensitive_words（敏感词全词匹配，修复误匹配）
    test_total_count += 1
    try:
        test_text = "测试123 敏感词 敏感词123"
        filtered_text = filter_sensitive_words(test_text, sensitive_words=TEST_SENSITIVE_WORDS,full_word_match=True)
        print(f"匹配前：{str(test_text)}")
        print(f"匹配后：{str(filtered_text)}")
        assert filtered_text == "测试123 *** 敏感词123", "敏感词误匹配修复未生效"
        print("✅ 测试2（敏感词全词匹配）：通过")
        test_pass_count += 1
    except AssertionError as e:
        print(f"❌ 测试2（敏感词全词匹配）：失败 - {str(e)}")
    except Exception as e:
        print(f"❌ 测试2（敏感词全词匹配）：异常 - {str(e)}")

    # # 3. 测试 load_stopwords（自定义停用词加载）
    # test_total_count += 1
    # try:
    #     # 创建临时停用词文件
    #     stopword_content = "测试\n的\n是"
    #     tmp_stopword_path = _create_temp_file(stopword_content)
    #     stopwords = load_stopwords(tmp_stopword_path)
    #     os.unlink(tmp_stopword_path)  # 删除临时文件
    #     assert stopwords == {"测试", "的", "是"}, "自定义停用词加载失败"
    #     print("✅ 测试3（自定义停用词加载）：通过")
    #     test_pass_count += 1
    # except AssertionError as e:
    #     print(f"❌ 测试3（自定义停用词加载）：失败 - {str(e)}")
    # except Exception as e:
    #     print(f"❌ 测试3（自定义停用词加载）：异常 - {str(e)}")

    # 4. 测试 filter_text_length（文本长度过滤）
    test_total_count += 1
    try:
        short_text = "123"
        long_text = "1234567890"
        assert filter_text_length(short_text, min_len=5) == "", "短文本过滤未生效"
        assert filter_text_length(long_text, max_len=5) == "12345", "长文本截断未生效"
        print("✅ 测试4（文本长度过滤）：通过")
        test_pass_count += 1
    except AssertionError as e:
        print(f"❌ 测试4（文本长度过滤）：失败 - {str(e)}")
    except Exception as e:
        print(f"❌ 测试4（文本长度过滤）：异常 - {str(e)}")

    # 5. 测试 text_clean_pipeline（流水线处理DataFrame）
    test_total_count += 1
    try:
        # ===== 新增：输出清洗前的文本 =====
        print("\n--- 文本清洗前 ---")
        # 提取清洗前的文本列表并打印（保持和清洗后对应）
        original_text_list = TEST_DF["text"].tolist()
        for idx, text in enumerate(original_text_list):
            print(f"索引 {idx}：{text}")
    
        # 执行文本清洗流水线
        processed_df = text_clean_pipeline(
            TEST_DF, col="text",
            filter_sensitive=True,
            sensitive_words=TEST_SENSITIVE_WORDS,
            min_len=3
        )

        # ===== 新增：输出清洗后的文本 =====
        print("\n--- 文本清洗后 ---")
        # 提取清洗后的文本列表并打印
        cleaned_text_list = processed_df["text"].tolist()
        for idx, text in enumerate(cleaned_text_list):
            print(f"索引 {idx}：{text}")
    
        # 原有断言逻辑
        expected_result = ["Hello2025年！", "测试***123", "", "", ""]
        assert processed_df["text"].tolist() == expected_result, "流水线处理结果不符合预期"
        print("\n✅ 测试5（流水线处理DataFrame）：通过")
        test_pass_count += 1
    except AssertionError as e:
        print(f"\n❌ 测试5（流水线处理DataFrame）：失败 - {str(e)}")
    except Exception as e:
        print(f"\n❌ 测试5（流水线处理DataFrame）：异常 - {str(e)}")

    # 6. 测试 text_quality_evaluate（文本质量评估）
    test_total_count += 1
    try:
        # 准备测试Series
        test_series = pd.Series(["Hello😀2025 年！", "测试敏感词123", "我是一个测试文本，长度超过20字。"])
        # 评估清洗前文本
        eval_result = text_quality_evaluate(test_series, is_cleaned=False)
        assert eval_result["有效样本数"] == 3, "有效样本数统计错误"
        assert eval_result["平均字符长度"] > 0, "平均字符长度计算错误"
        print("✅ 测试6（文本质量评估）：通过")
        print(f"\n📊 文本质量评估结果预览：")
        for key, value in eval_result.items():
            if key != "长度分布":
                print(f"   {key}：{value}")
            else:
                print(f"   {key}：{value}")
        test_pass_count += 1
    except AssertionError as e:
        print(f"❌ 测试6（文本质量评估）：失败 - {str(e)}")
    except Exception as e:
        print(f"❌ 测试6（文本质量评估）：异常 - {str(e)}")

    # 测试总结
    print("\n" + "=" * 60)
    print(f"测试完成：共{test_total_count}个测试，通过{test_pass_count}个，失败{test_total_count - test_pass_count}个")
    print("=" * 60)

# _run_builtin_tests()