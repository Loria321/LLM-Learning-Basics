# 导入所需库
import jieba
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import silhouette_score

# ====================== 1. 数据准备（复用扩充的校园问答样本） ======================
# 0=选课主题（20篇）
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

# 1=图书馆主题（20篇）
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

# 2=请假主题（20篇）
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

# 整合数据和标签
texts = course_qa + library_qa + leave_qa
labels = [0]*20 + [1]*20 + [2]*20  # 0=选课，1=图书馆，2=请假
label_names = ["选课", "图书馆", "请假"]
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]  # 三类主题的配色

# ====================== 2. 中文预处理函数（深度优化版） ======================
def preprocess_text(text, stop_words=None, core_topic_words=None):
    """
    深度优化预处理：
    1. 分词 + 过滤停用词
    2. 过滤单字词汇
    3. 强制保留核心主题词，剔除无关通用词
    """
    # 分词（精确模式）
    words = jieba.lcut(text.strip())
    
    if stop_words and core_topic_words:
        # 三层过滤：停用词 + 单字 + 非核心主题词
        words = [
            w for w in words 
            if w not in stop_words          # 过滤停用词
            and len(w) > 1                 # 过滤单字
            and (
                w in core_topic_words      # 直接是核心主题词
                or any(ctw in w for ctw in core_topic_words)  # 包含核心主题词（如“选修课”包含“选课”）
            )
        ]
    return " ".join(words)

# 核心配置：停用词+核心主题词（从源头把控特征质量）
# 1. 超全停用词列表（覆盖所有通用无区分词）
stop_words = [
    # 基础停用词
    "的", "地", "得", "我", "你", "他", "她", "它", "什么", "如何", "请问", "吗", "呢",
    "这", "那", "是", "有", "在", "了", "就", "都", "可以", "需要", "能", "会", "要",
    "该", "从", "为", "与", "和", "及", "或", "若", "如", "也", "还", "只", "仅", "又",
    # 新增通用词（之前筛选出的无区分词）
    "专业", "事后", "多少", "审批", "开放", "怎么", "提交", "时间", "期间", "流程", "通过",
    "手续", "材料", "证明", "影响", "原因", "解决", "找到", "使用", "办理", "限制", "资格"
]

# 2. 核心主题词（仅保留三类主题的核心标识词）
core_topic_words = [
    "选课", "选修课", "公选课",  # 选课类核心
    "图书馆", "借阅", "自习室", "续借",  # 图书馆类核心
    "请假", "病假", "事假", "销假"  # 请假类核心
]

# 执行预处理
processed_texts = [
    preprocess_text(t, stop_words=stop_words, core_topic_words=core_topic_words) 
    for t in texts
]

# ====================== 3. 基础版TF-IDF特征（保留原逻辑，用于对比） ======================
tfidf_basic = TfidfVectorizer()
features_basic = tfidf_basic.fit_transform(processed_texts).toarray()
print(f"【基础版】TF-IDF特征维度：{features_basic.shape[1]}")

# ====================== 4. 优化版TF-IDF特征（精准过滤+核心筛选） ======================
"""
优化点：
1. max_df=0.6：过滤60%以上样本出现的通用词（比0.8更严格）
2. min_df=3：过滤仅出现2次及以下的低频词
3. max_features=100：大幅缩减原始特征数，聚焦核心
4. ngram_range=(1,1)：先回归单字，确保核心词权重集中
"""
tfidf_optimized = TfidfVectorizer(
    ngram_range=(1, 1),        # 单字特征（核心词权重集中）
    max_df=0.6,                # 严格过滤通用词（文档频率>60%的词）
    min_df=3,                  # 过滤低频词（出现<3次的词）
    max_features=100           # 限制原始特征总数，避免冗余
)

# 第一步：提取优化版原始特征
features_optimized_raw = tfidf_optimized.fit_transform(processed_texts).toarray()
print(f"【优化版】原始TF-IDF特征维度（过滤后）：{features_optimized_raw.shape[1]}")

# 第二步：互信息法筛选（k=8，仅保留最核心的8个特征）
skb = SelectKBest(score_func=mutual_info_classif, k=8)
features_optimized = skb.fit_transform(features_optimized_raw, labels)
print(f"【优化版】互信息筛选后特征维度：{features_optimized.shape[1]}")

# ====================== 5. t-SNE可视化函数（消除警告+优化参数） ======================
def tsne_visualize(features, labels, n_components=2, perplexity=15, title=""):
    """
    优化版t-SNE可视化：
    1. 替换n_iter为max_iter，消除版本警告
    2. 适配8维特征的perplexity=15
    3. 增强可视化可读性
    """
    # t-SNE降维（版本兼容+参数优化）
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,      # 适配8维特征+60样本
        learning_rate=100,          # 保守学习率，避免过度分散
        random_state=42,            # 固定种子，结果可复现
        init="pca",                 # PCA初始化，提升稳定性
        max_iter=1000               # 替换n_iter为max_iter，消除警告
    )
    features_tsne = tsne.fit_transform(features)
    
    # 绘图配置（支持中文）
    plt.figure(figsize=(12, 8))
    plt.rcParams["font.sans-serif"] = ["SimHei"]   # 显示中文
    plt.rcParams["axes.unicode_minus"] = False     # 显示负号
    
    # 绘制2D散点图
    for label in [0, 1, 2]:
        mask = (np.array(labels) == label)
        plt.scatter(
            features_tsne[mask, 0],
            features_tsne[mask, 1],
            c=colors[label],
            label=label_names[label],
            alpha=0.8,       # 透明度提升
            s=80,            # 点大小增大
            edgecolors="white",
            linewidth=1,     # 白色描边，区分重叠点
            marker="o"       # 圆形标记
        )
    
    # 图表标注
    plt.xlabel("t-SNE维度1", fontsize=12, fontweight="bold")
    plt.ylabel("t-SNE维度2", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=15, fontweight="bold", pad=20)
    plt.legend(fontsize=12, loc="best", frameon=True, shadow=True)
    plt.grid(alpha=0.2, linestyle="--")  # 浅灰色虚线网格
    plt.tight_layout()
    plt.show()
    
    return features_tsne

# ====================== 6. 可视化对比（基础版 vs 优化版） ======================
# 6.1 基础版TF-IDF的2D可视化
print("\n=== 基础版TF-IDF特征2D可视化 ===")
tsne_2d_basic = tsne_visualize(
    features_basic,
    labels,
    n_components=2,
    perplexity=15,
    title="基础版TF-IDF特征 t-SNE 2D可视化（校园问答三类主题）"
)

# 6.2 优化版TF-IDF的2D可视化
print("\n=== 优化版TF-IDF特征2D可视化 ===")
tsne_2d_optimized = tsne_visualize(
    features_optimized,
    labels,
    n_components=2,
    perplexity=15,
    title="优化版TF-IDF特征（深度过滤+核心筛选） t-SNE 2D可视化"
)

# ====================== 7. 聚类效果量化分析+核心特征词校验 ======================
# 7.1 轮廓系数计算（量化聚类效果）
sil_basic = silhouette_score(tsne_2d_basic, labels)
sil_optimized = silhouette_score(tsne_2d_optimized, labels)

print("\n=== 聚类效果量化分析（最终优化版） ===")
print(f"基础版TF-IDF特征轮廓系数：{sil_basic:.4f}")
print(f"优化版TF-IDF特征轮廓系数：{sil_optimized:.4f}")
print(f"优化版相对基础版提升：{(sil_optimized - sil_basic)/sil_basic*100:.2f}%")

# 7.2 核心特征词二次校验（确保无通用词）
print("\n=== 优化版核心特征词（最终筛选结果） ===")
# 获取筛选后的特征索引
selected_indices = skb.get_support(indices=True)
# 获取TF-IDF词汇表
vocab = tfidf_optimized.get_feature_names_out()
# 输出核心特征词（已过滤通用词）
core_features = [vocab[idx] for idx in selected_indices]
# 最终校验：确保核心特征词都在核心主题词范围内
core_features_validated = [
    w for w in core_features 
    if w in core_topic_words or any(ctw in w for ctw in core_topic_words)
]
print(f"最终有效核心特征词：{core_features_validated}")
print(f"无效通用词（已自动过滤）：{list(set(core_features) - set(core_features_validated))}")