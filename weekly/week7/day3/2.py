import jieba
import numpy as np
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from openai.types.chat import ChatCompletion

# ====================== 1. 配置通义千问OpenAI兼容接口 ======================
# 替换为你的阿里云百炼API-KEY（优先环境变量，也可直接赋值）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or "你的通义千问API-KEY"

# 初始化OpenAI客户端（适配阿里云百炼兼容模式）
def init_openai_client():
    try:
        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云兼容接口地址
        )
        return client
    except Exception as e:
        print(f"客户端初始化失败：{str(e)}")
        return None

# 初始化客户端（全局复用）
client = init_openai_client()

# ====================== 2. 数据准备（复用60篇校园问答，无修改） ======================
# 0=选课（20篇）
course_qa = [
    "我想选一门计算机相关的课程，请问有哪些推荐",
    "如何申请本学期的选修课，截止日期是什么时候",
    "本学期的公选课有哪些热门专业可以选择",
    "选修课的报名流程是什么，需要在教务系统填报吗",
    "错过了选课时间，还能补选或者退选课程吗",
    "大一新生可以选高年级的专业选修课吗",
    "计算机专业的核心课程有哪些，难度如何",
    "选修课的学分要求是多少，不够能毕业吗",
    "如何查询自己的选课结果，在哪里查看课表",
    "选了一门网课，需要按时完成线上作业吗",
    "可选课程里有没有人工智能相关的入门课",
    "申请跨专业选课需要满足什么条件，找谁审批",
    "选修课挂科了会影响绩点吗，需要重修吗",
    "本学期的选课系统什么时候开放，持续几天",
    "有没有容易拿学分的公选课，求推荐",
    "双学位的选课和本专业选课冲突了怎么办",
    "体育课可以选两门吗，有没有特殊要求",
    "选课的时候显示名额已满，还有机会候补吗",
    "教务系统选课失败是什么原因，该怎么解决",
    "研究生可以选本科生的选修课来修学分吗"
]
# 1=图书馆（20篇）
library_qa = [
    "图书馆的图书可以借阅多久，逾期会有什么处罚",
    "图书馆的自习室需要提前预约吗，预约方式是什么",
    "如何查询图书馆有没有我需要的专业书籍",
    "借阅的图书快到期了，能不能在线续借，续借多久",
    "图书馆的电子资源怎么访问，需要校园网吗",
    "不小心把借阅的图书弄丢了，该怎么赔偿",
    "图书馆的工具书可以外借吗，还是只能在馆内阅读",
    "自习室预约成功后，迟到多久会取消资格",
    "图书馆的复印打印服务在哪里，怎么收费",
    "校外人员可以进入学校图书馆吗，需要什么证件",
    "借阅的图书有破损，还书的时候会被罚款吗",
    "图书馆的数据库资源可以下载论文吗，有没有版权限制",
    "自习室里可以使用笔记本电脑吗，有没有电源插座",
    "如何办理图书馆的读者证，需要准备什么材料",
    "图书馆的新书多久更新一次，怎么关注新书通知",
    "借取的图书想转借给同学，可以吗，需要办理手续吗",
    "图书馆的闭馆时间是什么时候，周末开放吗",
    "忘记带校园卡，能不能用身份证进入图书馆",
    "图书馆的专题书架在哪里，怎么快速找到",
    "电子图书可以下载到本地吗，支持什么格式"
]
# 2=请假（20篇）
leave_qa = [
    "我需要请假三天，请假流程是什么样的",
    "请假需要提交什么材料，多久能审批通过",
    "病假需要提供医院的诊断证明吗，复印件可以吗",
    "事假最多可以请多少天，会不会影响考勤成绩",
    "学生请假需要找辅导员还是班主任审批",
    "请假期间的课程落下了，该怎么补回来",
    "异地就医无法及时提交病假材料，能事后补报吗",
    "毕业班学生请假外出找工作，需要额外提交什么证明",
    "请假审批通过后，需要告知任课老师吗",
    "节假日前后请假，会不会有特殊限制",
    "休学和长期请假有什么区别，该怎么办理",
    "请假条填写错误，可以重新填写提交吗",
    "体育课请假需要单独找体育老师审批吗",
    "因为家里有事紧急请假，能不能走绿色通道快速审批",
    "请假期间的作业和考试，能不能申请缓交或缓考",
    "研究生请假需要导师和辅导员双重审批吗",
    "请假记录会记入学生档案吗，对评优有影响吗",
    "忘记办理请假手续，事后补假需要什么材料",
    "实习期间需要请假，应该找学校还是实习单位审批",
    "病假痊愈后返校，需要向辅导员销假吗"
]
# 整合数据、标签
texts = course_qa + library_qa + leave_qa
labels = [0]*20 + [1]*20 + [2]*20  # 0=选课，1=图书馆，2=请假
label_map = {0:"选课", 1:"图书馆", 2:"请假"}  # 标签与文本映射
reverse_label_map = {"选课":0, "图书馆":1, "请假":2}

# ====================== 3. 提取TF-IDF核心关键词（保留优化版逻辑） ======================
# 停用词列表（扩充，适配通义千问的语义理解）
stop_words = [
    "的", "地", "得", "我", "你", "他", "吗", "呢", "这", "那", "是", "有", "在", "了", "就", "都",
    "如何", "什么", "请问", "可以", "需要", "能", "会", "该", "要", "从", "到", "为", "与", "和"
]

# 预处理函数：分词+去停用词+过滤短词
def preprocess(text):
    words = jieba.lcut(text.strip())
    return " ".join([w for w in words if w not in stop_words and len(w)>=2])  # 仅保留≥2字的词

processed_texts = [preprocess(t) for t in texts]

# 提取TF-IDF特征（优化参数）
tfidf = TfidfVectorizer(
    ngram_range=(1,2),    # 单字+双词
    max_df=0.8,           # 过滤80%以上样本出现的通用词
    min_df=2,             # 过滤仅出现1次的低频词
    max_features=200      # 限制特征总数
)
tfidf_matrix = tfidf.fit_transform(processed_texts).toarray()
vocab_list = tfidf.get_feature_names_out()  # 词汇表

# 提取每个样本的Top3 TF-IDF核心关键词
def get_top_tfidf_words(matrix, vocab, top_n=3):
    top_words_list = []
    for row in matrix:
        # 按TF-IDF权重排序，过滤权重为0的词
        top_indices = [i for i in row.argsort()[-top_n:][::-1] if row[i]>0.01]
        top_words = [vocab[i] for i in top_indices]
        # 不足补空，保证格式统一
        top_words += [""]*(top_n - len(top_words))
        top_words_str = "、".join([w for w in top_words if w])
        top_words_list.append(top_words_str)
    return top_words_list

top3_tfidf_words = get_top_tfidf_words(tfidf_matrix, vocab_list, top_n=3)

# ====================== 4. Prompt模板构造（保留优化版） ======================
# 纯文本Prompt模板（强化指令）
PROMPT_PURE = """### 任务说明
请严格按照以下要求完成校园问答主题分类：
1. 可选主题：仅允许输出「选课」「图书馆」「请假」三者之一；
2. 输出规则：仅输出主题名称，不添加任何解释、标点、换行或额外文字；
3. 判断依据：基于校园问答的核心语义判断主题。

### 校园问答
{text}

### 主题输出
"""

# 文本+TF-IDF特征Prompt模板
PROMPT_FEATURE = """### 任务说明
请严格按照以下要求完成校园问答主题分类：
1. 可选主题：仅允许输出「选课」「图书馆」「请假」三者之一；
2. 输出规则：仅输出主题名称，不添加任何解释、标点、换行或额外文字；
3. 判断依据：优先结合该问题的核心关键词，再基于语义判断主题。

### 核心关键词
{keywords}

### 校园问答
{text}

### 主题输出
"""

# 构造两种Prompt列表
pure_prompts = [PROMPT_PURE.format(text=t) for t in texts]
feature_prompts = [PROMPT_FEATURE.format(keywords=kw, text=t) for kw,t in zip(top3_tfidf_words, texts)]

# ====================== 5. 通义千问API调用（OpenAI兼容模式，核心优化） ======================
def qwen_infer_openai(prompt, model="qwen-turbo", retry_times=3, delay=1):
    """
    通义千问API推理（OpenAI兼容模式）
    Args:
        prompt: 输入的Prompt
        model: 模型版本（qwen-turbo/qwen-plus/qwen-max）
        retry_times: 失败重试次数
        delay: 重试间隔（秒）
    Returns:
        清洗后的主题结果（选课/图书馆/请假/None）
    """
    if client is None:
        print("客户端未初始化，无法调用API")
        return None
    
    # 构造OpenAI格式的messages
    messages = [
        {"role": "system", "content": "你是一个精准的文本分类助手，严格遵循用户指令输出结果。"},
        {"role": "user", "content": prompt}
    ]
    
    # 重试机制
    for i in range(retry_times):
        try:
            # 调用OpenAI兼容接口
            completion: ChatCompletion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.01,  # 极低温度，保证输出稳定
                top_p=0.01,
                max_tokens=5,       # 仅输出主题（最多5字符）
                stream=False
            )
            # 提取并清洗结果
            res = completion.choices[0].message.content.strip()
            # 精准匹配可选主题
            if res in ["选课", "图书馆", "请假"]:
                return res
            # 处理冗余输出（如“主题：选课”“选课。”）
            elif "选课" in res:
                return "选课"
            elif "图书馆" in res:
                return "图书馆"
            elif "请假" in res:
                return "请假"
            else:
                print(f"无效输出（第{i+1}次重试）：{res}")
                time.sleep(delay)
        except Exception as e:
            print(f"API调用异常（第{i+1}次重试）：{str(e)}")
            time.sleep(delay)
    
    # 多次重试失败返回None
    return None

# ====================== 6. 对比实验（保留原逻辑） ======================
def run_experiment(prompts, experiment_name):
    """运行实验，返回预测结果"""
    preds = []
    total = len(prompts)
    print(f"\n===== 开始{experiment_name}推理（共{total}个样本） =====")
    for i, prompt in enumerate(prompts):
        res = qwen_infer_openai(prompt, model="qwen-plus")  # 推荐用qwen-plus提升效果
        preds.append(res)
        # 每10个样本输出进度
        if (i+1) % 10 == 0:
            print(f"已完成{i+1}/{total}个样本，当前有效预测数：{len([p for p in preds if p is not None])}")
    return preds

# 6.1 纯文本输入推理
pure_preds = run_experiment(pure_prompts, "纯文本输入")

# 6.2 文本+TF-IDF特征输入推理
feature_preds = run_experiment(feature_prompts, "文本+TF-IDF特征输入")

# ====================== 7. 准确率计算与结果分析（保留原逻辑） ======================
def calculate_accuracy(preds, true_labels):
    """计算准确率：无效预测记为错误"""
    correct = 0
    total = len(true_labels)
    for p, t in zip(preds, true_labels):
        true_t = label_map[t]
        if p == true_t:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

# 计算两种方式的准确率
pure_acc, pure_correct, pure_total = calculate_accuracy(pure_preds, labels)
feature_acc, feature_correct, feature_total = calculate_accuracy(feature_preds, labels)

# 输出核心对比结果
print("\n" + "="*80)
print("【通义千问API（OpenAI兼容模式） - 纯文本 VS 文本+TF-IDF特征 准确率对比】")
print(f"纯文本输入：正确数={pure_correct}/{pure_total}，准确率={pure_acc:.4f}（{pure_acc*100:.2f}%）")
print(f"文本+TF-IDF特征：正确数={feature_correct}/{feature_total}，准确率={feature_acc:.4f}（{feature_acc*100:.2f}%）")
print(f"准确率提升：{(feature_acc - pure_acc)*100:.2f}个百分点")
print("="*80)

# 输出错误案例
def print_error_cases(preds, experiment_name):
    """输出错误案例"""
    print(f"\n===== {experiment_name}错误案例（前5个） =====")
    error_count = 0
    for i, (p, t) in enumerate(zip(preds, labels)):
        true_t = label_map[t]
        if p != true_t and p is not None:
            print(f"样本{i+1}：{texts[i]}")
            print(f"模型预测：{p} | 真实主题：{true_t}\n")
            error_count += 1
            if error_count >= 5:
                break
    if error_count == 0:
        print("无错误案例！")

# 输出错误案例
print_error_cases(pure_preds, "纯文本输入")
print_error_cases(feature_preds, "文本+TF-IDF特征输入")