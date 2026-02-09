# 导入所需库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
import jieba
import numpy as np

# 步骤1：构造充足的校园问答样本集（3类，每类20篇，共60篇）
# 0=选课主题
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

# 1=图书馆主题
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

# 2=请假主题
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

# 整合所有样本和标签
campus_qa = course_qa + library_qa + leave_qa
labels = [0]*20 + [1]*20 + [2]*20  # 对应3类主题，每类20个标签

# 中文分词预处理（仅TF-IDF需要）
def chinese_word_cut(text):
    return " ".join(jieba.lcut(text))

processed_qa = [chinese_word_cut(qa) for qa in campus_qa]

print(f"样本集构造完成，共{len(campus_qa)}篇校园问答，{len(set(labels))}类主题")
print(f"选课主题：{len(course_qa)}篇 | 图书馆主题：{len(library_qa)}篇 | 请假主题：{len(leave_qa)}篇")

# 步骤2：提取两种特征
# 2.1 提取TF-IDF特征
tfidf_vec = TfidfVectorizer()
tfidf_features = tfidf_vec.fit_transform(processed_qa).toarray()

# 2.2 提取大模型Embedding特征（复用之前的轻量多语言模型）
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embedding_features = embedding_model.encode(campus_qa)

# 输出两种特征的维度信息
print("\n" + "="*60)
print("【两种特征维度对比】")
print(f"TF-IDF特征维度：{tfidf_features.shape[1]} 维（对应词汇表大小）")
print(f"大模型Embedding特征维度：{embedding_features.shape[1]} 维（模型固定维度）")
print(f"TF-IDF特征矩阵形状：{tfidf_features.shape}")
print(f"大模型Embedding特征矩阵形状：{embedding_features.shape}")

# 步骤3：构建逻辑回归模型（样本充足，可正常学习）
lr_model = LogisticRegression(random_state=42, max_iter=2000)

# 3.1 方式1：留一法交叉验证（结果最稳定，适合充足样本，稍慢）
print("\n" + "="*60)
print("【方式1：留一法交叉验证】")
loo = LeaveOneOut()

# TF-IDF特征评估
tfidf_loo_scores = cross_val_score(lr_model, tfidf_features, labels, cv=loo, scoring='accuracy')
tfidf_loo_avg = np.mean(tfidf_loo_scores)

# 大模型Embedding特征评估
embedding_loo_scores = cross_val_score(lr_model, embedding_features, labels, cv=loo, scoring='accuracy')
embedding_loo_avg = np.mean(embedding_loo_scores)

print(f"TF-IDF特征 平均准确率：{tfidf_loo_avg:.4f}（{tfidf_loo_avg*100:.2f}%）")
print(f"大模型Embedding特征 平均准确率：{embedding_loo_avg:.4f}（{embedding_loo_avg*100:.2f}%）")

# 3.2 方式2：普通训练测试拆分（8:2，计算更快，直观易懂）
print("\n" + "="*60)
print("【方式2：训练测试拆分（8:2）】")
# 拆分数据（随机种子固定，结果可复现）
X_tfidf_train, X_tfidf_test, y_tfidf_train, y_tfidf_test = train_test_split(
    tfidf_features, labels, test_size=0.2, random_state=42
)
X_emb_train, X_emb_test, y_emb_train, y_emb_test = train_test_split(
    embedding_features, labels, test_size=0.2, random_state=42
)

# TF-IDF特征训练与预测
lr_model.fit(X_tfidf_train, y_tfidf_train)
tfidf_test_pred = lr_model.predict(X_tfidf_test)
tfidf_test_acc = accuracy_score(y_tfidf_test, tfidf_test_pred)

# 大模型Embedding特征训练与预测
lr_model.fit(X_emb_train, y_emb_train)
emb_test_pred = lr_model.predict(X_emb_test)
emb_test_acc = accuracy_score(y_emb_test, emb_test_pred)

print(f"TF-IDF特征 测试集准确率：{tfidf_test_acc:.4f}（{tfidf_test_acc*100:.2f}%）")
print(f"大模型Embedding特征 测试集准确率：{emb_test_acc:.4f}（{emb_test_acc*100:.2f}%）")

# 3.3 输出训练集上的整体预测结果（直观查看拟合效果）
print("\n" + "="*60)
print("【训练集整体拟合结果】")
lr_model.fit(tfidf_features, labels)
tfidf_full_pred = lr_model.predict(tfidf_features)
tfidf_full_acc = accuracy_score(labels, tfidf_full_pred)

lr_model.fit(embedding_features, labels)
emb_full_pred = lr_model.predict(embedding_features)
emb_full_acc = accuracy_score(labels, emb_full_pred)

print(f"TF-IDF特征 训练集全量准确率：{tfidf_full_acc:.4f}（{tfidf_full_acc*100:.2f}%）")
print(f"大模型Embedding特征 训练集全量准确率：{emb_full_acc:.4f}（{emb_full_acc*100:.2f}%）")