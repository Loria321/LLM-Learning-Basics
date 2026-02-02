import jieba
import opencc
import re
import pandas as pd
from typing import List, Union, Optional, Callable, Literal, TypeAlias
from pathlib import Path
from functools import lru_cache
from typing_extensions import TypeGuard

# ======================== 常量定义（抽离硬编码，统一维护） ========================
# 跨平台默认路径（项目根目录下的config文件夹）
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SENSITIVE_PATH = PROJECT_ROOT / "config" / "sensitive_words.txt"
DEFAULT_STOPWORD_PATH = PROJECT_ROOT / "config" / "stopwords.txt"

# 文本清理常量
DEFAULT_KEEP_CHARS = "，。！？；：\"''（）【】《》a-zA-Z0-9\u4e00-\u9fa5"
# Emoji正则（拆分+注释，提升可读性）
EMOJI_PATTERN = re.compile(
    r"""
    [\U0001F600-\U0001F64F]  # 表情符号
    |[\U0001F300-\U0001F5FF] # 符号&象形图
    |[\U0001F680-\U0001F6FF] # 交通&地图符号
    |[\U0001F1E0-\U0001F1FF] # 国旗符号
    """,
    flags=re.UNICODE | re.VERBOSE
)
# 特殊排版噪声（可扩展）
SPECIAL_NOISE_PATTERN = re.compile(r'★|■|◆|●|△|▲|※|§|№|＃|＆|＄|％|＠|～|｀|＾|｜|＼|／')
# HTML标签正则（预编译）
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

# ======================== 类型别名（简化复杂类型注解） ========================
TextInput = Union[str, List[str]]
SplitResult = Union[List[str], List[List[str]]]
DataInput = Union[pd.DataFrame, List[str]]
ConvertType = Literal["s2t", "t2s"]  # 严格约束繁简转换类型
CleanRule = Callable[[str], str]     # 自定义清理规则的函数类型

# ======================== 通用工具函数（减少重复代码） ========================
def is_list_of_str(v: object) -> TypeGuard[List[str]]:
    """类型守卫：判断是否为字符串列表"""
    return isinstance(v, list) and all(isinstance(item, str) for item in v)

def process_batch(
    data: TextInput,
    handler: Callable[[str], str]
) -> TextInput:
    """
    通用批量处理函数：统一处理单个文本/文本列表
    :param data: 输入文本/文本列表
    :param handler: 单个文本的处理函数
    :return: 处理后结果
    """
    if isinstance(data, str):
        return handler(data)
    elif is_list_of_str(data):
        return [handler(item) for item in data]
    else:
        raise TypeError(f"输入类型必须为str或List[str]，当前类型：{type(data)}")

# ======================== 缓存装饰器（减少重复IO/实例化） ========================
@lru_cache(maxsize=2)  # 仅缓存s2t/t2s两个实例
def get_opencc_converter(convert_type: ConvertType) -> opencc.OpenCC:
    """缓存opencc实例，避免重复初始化"""
    return opencc.OpenCC(convert_type)

@lru_cache(maxsize=1)  # 缓存默认/指定路径的停用词
def load_stopwords(stopword_path: Optional[str] = None) -> set:
    """
    加载停用词表（带缓存）
    :param stopword_path: 停用词文件路径（每行一个停用词）
    :return: 停用词集合
    :example:
        >>> load_stopwords()
        {'的', '了', '是', '我', '你', '他'}
    """
    if stopword_path is None:
        return {"的", "了", "是", "我", "你", "他", "她", "它", "们", "在", "有", "就", "不", "和", "也", "都"}
    
    try:
        with open(stopword_path, "r", encoding="utf-8") as f:
            return set([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        raise FileNotFoundError(f"停用词文件不存在：{stopword_path}")
    except PermissionError:
        raise PermissionError(f"无权限读取停用词文件：{stopword_path}")

@lru_cache(maxsize=1)
def load_sensitive_words(sensitive_path: Optional[str] = None) -> set:
    """
    加载敏感词表（带缓存）
    :param sensitive_path: 敏感词文件路径
    :return: 敏感词集合
    """
    if sensitive_path is None:
        return set()
    
    try:
        with open(sensitive_path, "r", encoding="utf-8") as f:
            return set([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        raise FileNotFoundError(f"敏感词文件不存在：{sensitive_path}")
    except PermissionError:
        raise PermissionError(f"无权限读取敏感词文件：{sensitive_path}")

# ======================== 文本清理核心函数（优化性能+扩展性） ========================
def clean_text_black_white(
    text: TextInput,
    keep_chars: str = DEFAULT_KEEP_CHARS,
    replace_char: str = "",
    optimize_format: bool = True,
    custom_black_rules: Optional[List[CleanRule]] = None  # 自定义黑名单规则
) -> TextInput:
    """
    黑白名单结合的文本去噪方案：先靶向黑名单清理，再白名单兜底提纯
    :param text: 待处理文本/文本列表
    :param keep_chars: 白名单：保留的字符集
    :param replace_char: 白名单过滤时，替换非保留字符的内容（默认空字符串删除）
    :param optimize_format: 是否优化文本格式（合并多余空格、数字+中文去空格）
    :param custom_black_rules: 自定义黑名单清理规则（列表，每个元素为单文本处理函数）
    :return: 清洗后文本/文本列表
    :example:
        >>> clean_text_black_white("Hello😀<p>2025 年</p>！")
        'Hello2025年！'
    """
    # 预编译白名单正则（按keep_chars缓存）
    @lru_cache(maxsize=10)
    def _get_whitelist_pattern(keep_chars: str) -> re.Pattern:
        return re.compile(f"[^{re.escape(keep_chars)}]")
    
    def _blacklist_clean_single(txt: str) -> str:
        # 处理无效输入（NaN/None/空字符串）
        if pd.isna(txt) or txt is None or txt.strip() == "":
            return ""
        
        # 1. 内置黑名单规则
        txt = HTML_TAG_PATTERN.sub('', txt)          # 删除HTML标签
        txt = EMOJI_PATTERN.sub('', txt)             # 删除Emoji
        txt = SPECIAL_NOISE_PATTERN.sub('', txt)     # 删除特殊排版符号
        
        # 2. 自定义黑名单规则（扩展点）
        if custom_black_rules:
            for rule in custom_black_rules:
                txt = rule(txt)
        
        # 3. 格式优化
        if optimize_format:
            txt = re.sub(r'\s+', ' ', txt).strip()  # 合并多余空格
            txt = re.sub(r'(\d+) +([年月日])', r'\1\2', txt)  # 数字+年月日去空格
        
        return txt
    
    def _whitelist_clean_single(txt: str) -> str:
        pattern = _get_whitelist_pattern(keep_chars)
        return pattern.sub(replace_char, txt)
    
    # 组合黑白名单处理
    def _clean_single(txt: str) -> str:
        black_cleaned = _blacklist_clean_single(txt)
        white_cleaned = _whitelist_clean_single(black_cleaned)
        return white_cleaned
    
    # 通用批量处理
    return process_batch(text, _clean_single)

# ======================== 分词/去停用词（优化扩展性+类型注解） ========================
def split_text(
    text: TextInput,
    cut_all: bool = False,
    tokenizer: Optional[Callable[[str], List[str]]] = None  # 自定义分词器
) -> SplitResult:
    """
    文本分词（支持自定义分词器）
    :param text: 待分词文本/文本列表
    :param cut_all: jieba全模式（仅当使用默认jieba时生效）
    :param tokenizer: 自定义分词器（优先级高于jieba）
    :return: 分词结果/分词结果列表
    :example:
        >>> split_text("我是中国人")
        ['我', '是', '中国人']
    """
    def _cut(t: str) -> List[str]:
        if tokenizer:
            return tokenizer(t)
        return jieba.lcut(t, cut_all=cut_all)
    
    def _split_single(txt: str) -> List[str]:
        return _cut(txt.strip()) if txt.strip() else []
    
    if isinstance(text, str):
        return _split_single(text)
    elif is_list_of_str(text):
        return [_split_single(item) for item in text]
    else:
        raise TypeError(f"输入类型必须为str或List[str]，当前类型：{type(text)}")

def remove_stopwords(
    words: SplitResult,
    stopwords: Optional[set] = None,
    stopword_path: Optional[str] = None
) -> SplitResult:
    """
    去除停用词（增强类型校验）
    :param words: 分词结果/分词结果列表
    :param stopwords: 停用词集合（优先级高于stopword_path）
    :param stopword_path: 停用词文件路径
    :return: 去停用词后结果
    :example:
        >>> remove_stopwords(['我', '是', '中国人'], stopwords={'是'})
        ['我', '中国人']
    """
    stopwords = stopwords or load_stopwords(stopword_path)
    
    def _remove_single(ws: List[str]) -> List[str]:
        return [w for w in ws if w not in stopwords and w.strip()]
    
    if isinstance(words, list) and all(isinstance(w, str) for w in words):
        return _remove_single(words)
    elif isinstance(words, list) and all(isinstance(w, list) for w in words):
        return [_remove_single(w_list) for w_list in words]
    else:
        raise TypeError("输入必须为List[str]或List[List[str]]")

# ======================== 敏感词/长度过滤（优化鲁棒性） ========================
def filter_sensitive_words(
    text: TextInput,
    sensitive_words: Optional[set] = None,
    sensitive_path: Optional[str] = None,
    replace_char: str = "*"
) -> TextInput:
    """
    敏感词过滤（带缓存+异常处理）
    :param text: 待处理文本/文本列表
    :param sensitive_words: 敏感词集合（优先级高于sensitive_path）
    :param sensitive_path: 敏感词文件路径
    :param replace_char: 替换字符
    :return: 过滤后文本/文本列表
    """
    sensitive_words = sensitive_words or load_sensitive_words(sensitive_path)
    
    def _filter_single(txt: str) -> str:
        for word in sensitive_words:
            if word in txt:
                txt = txt.replace(word, replace_char * len(word))
        return txt
    
    return process_batch(text, _filter_single)

def filter_text_length(
    text: TextInput,
    min_len: int = 1,
    max_len: Optional[int] = None
) -> TextInput:
    """
    文本长度过滤（添加参数合法性校验）
    :param text: 待处理文本/文本列表
    :param min_len: 最小长度（≥0）
    :param max_len: 最大长度（None则不限制）
    :return: 过滤后文本/文本列表
    :raises ValueError: 当min_len > max_len时触发
    """
    # 参数合法性校验
    if min_len < 0:
        raise ValueError(f"min_len不能为负数，当前值：{min_len}")
    if max_len is not None and max_len < min_len:
        raise ValueError(f"max_len({max_len})不能小于min_len({min_len})")
    
    def _filter_single(txt: str) -> str:
        stripped_txt = txt.strip()
        length = len(stripped_txt)
        if length < min_len:
            return ""
        if max_len and length > max_len:
            return stripped_txt[:max_len]
        return stripped_txt
    
    return process_batch(text, _filter_single)

def convert_traditional_simplified(
    text: TextInput,
    convert_type: ConvertType = "s2t"
) -> TextInput:
    """
    繁简转换（缓存实例+严格类型约束）
    :param text: 待处理文本/文本列表
    :param convert_type: 转换类型（s2t:简转繁, t2s:繁转简）
    :return: 转换后文本/文本列表
    """
    converter = get_opencc_converter(convert_type)
    
    def _convert_single(txt: str) -> str:
        return converter.convert(txt) if txt.strip() else txt
    
    return process_batch(text, _convert_single)

# ======================== 文本处理流水线（优化扩展性+可配置） ========================
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
    stopword_path: Optional[str] = str(DEFAULT_STOPWORD_PATH),
    custom_black_rules: Optional[List[CleanRule]] = None
) -> Union[pd.DataFrame, List[str], List[List[str]]]:
    """
    文本专项处理流水线（支持自定义清理规则，优化步骤可配置性）
    :param data: 待处理数据（DataFrame/List[str]）
    :param col: DataFrame处理列名（仅当data为DataFrame时生效）
    :param do_split: 是否分词
    :param remove_stopwords_flag: 是否去停用词（仅当do_split=True时生效）
    :param filter_sensitive: 是否过滤敏感词
    :param filter_length: 是否过滤长度
    :param min_len: 最小长度（filter_length=True时生效）
    :param max_len: 最大长度（filter_length=True时生效）
    :param convert_t2s: 是否繁转简
    :param sensitive_path: 敏感词路径
    :param stopword_path: 停用词路径
    :param custom_black_rules: 自定义黑名单清理规则
    :return: 处理后数据
    :example:
        >>> df = pd.DataFrame({"text": ["Hello😀2025 年！", "敏感词测试"]})
        >>> text_clean_pipeline(df, col="text", convert_t2s=True, filter_sensitive=True)
               text
        0  Hello2025年！
        1        ***测试
    """
    # 1. 输入数据格式统一
    if isinstance(data, pd.DataFrame):
        if not col or col not in data.columns:
            raise ValueError(f"DataFrame必须指定有效列名，当前列：{data.columns}")
        text_list = data[col].astype(str).tolist()  # 统一转为字符串，避免NaN
    elif is_list_of_str(data):
        text_list = data
    else:
        raise TypeError(f"仅支持pd.DataFrame或List[str]类型，当前类型：{type(data)}")
    
    # 2. 核心处理步骤（可按需调整顺序）
    steps = []
    # 繁简转换
    if convert_t2s:
        text_list = convert_traditional_simplified(text_list, "t2s")
    # 黑白名单清理（核心）
    text_list = clean_text_black_white(
        text_list,
        optimize_format=True,
        custom_black_rules=custom_black_rules
    )
    # 敏感词过滤
    if filter_sensitive:
        text_list = filter_sensitive_words(text_list, sensitive_path=sensitive_path)
    # 长度过滤
    if filter_length:
        text_list = filter_text_length(text_list, min_len=min_len, max_len=max_len)
    # 分词+去停用词
    if do_split:
        text_list = split_text(text_list)
        if remove_stopwords_flag:
            text_list = remove_stopwords(text_list, stopword_path=stopword_path)
    
    # 3. 结果回填
    if isinstance(data, pd.DataFrame) and col:
        data = data.copy()  # 避免修改原DataFrame
        data[col] = text_list
        return data
    return text_list

# ======================== 初始化（提前加载jieba，提升首次调用性能） ========================
# 提前初始化jieba，避免首次分词时的加载耗时
jieba.initialize()