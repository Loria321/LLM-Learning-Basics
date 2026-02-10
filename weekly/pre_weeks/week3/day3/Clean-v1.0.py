import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

#1.日志配置
def setup_logger(log_path):
    '''配置日志：同时输出到控制台和日志文件，记录清洗全流程'''
    #日志文件名包含时间戳，避免覆盖
    log_file = f"{log_path}_清洗日志_{datetime.now().strftime('%Y%M%d_%H%M%S')}.log"
    
    #配置日志格式
    logger = logging.getLogger('date_cleaner')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d%H:%M:%S')
    
    #控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    #文件处理器
    file_handler = logging.FileHandler(log_file,encoding='utf-8')
    file_handler.setFormatter(formatter)

    #避免重复添加处理器
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger,log_file

#2.核心清洗函数（可配置参数）
def clean_csv_data(
        input_path,         #输入CSV文件路径
        output_path,        #输出清洗后CSV路径
        log_path="清洗日志",    #日志文件基础路径
        duplicate_threshold=5.0,    #重复行占比阈值（%），超过则终止
        missing_fill_strategy="auto",   #缺失值填充策略：auto/mean/median/mode/drop
        missing_col_threshold=30.0,     #列缺失率阈值（%），超过则删除列
        outlier_method="IQR",       #异常值判定方法:IQR/3σ
        outlier_threshold=5.0       #异常值占比阈值（%），超过则提示
):
    """
    通用CSV数据清洗函数（支持参数配置）
    :param input_path: 输入CSV文件路径
    :param output_path: 输出清洗后CSV路径
    :param log_path: 日志文件保存基础路径
    :param duplicate_threshold: 重复行占比阈值（%），>该值则终止清洗
    :param missing_fill_strategy: 缺失值填充策略
                                  - auto：数值列用median，类别列用mode
                                  - mean：数值列用均值
                                  - median：数值列用中位数
                                  - mode：类别列用众数
                                  - drop：删除所有缺失行
    :param missing_col_threshold: 列缺失率阈值（%），>该值删除列
    :param outlier_method: 异常值判定方法（IQR/3σ）
    :param outlier_threshold: 异常值占比阈值（%），>该值仅提示不处理
    :return: 清洗后DataFrame、日志文件路径
    """
    #初始化日志
    logger,log_file = setup_logger(log_path)
    logger.info("="*50)
    logger.info("开始执行数据清洗流程")
    logger.info(f"输入文件：{input_path}")
    logger.info(f"配置参数：重复行阈值={duplicate_threshold}% | 缺失列阈值={missing_col_threshold}% | 缺失填充策略={missing_fill_strategy} | 异常值方法={outlier_method}")
    logger.info('='*50)

    try:
        #1)数据加载
        logger.info("1)数据加载")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在：{input_path}")
        df = pd.read_csv(input_path,encoding='utf-8')
        if df.empty:
            raise ValueError("加载的CSV文件为空")
        original_shape = df.shape
        logger.info(f"原始数据维度：{original_shape[0]}行 * {original_shape[1]}列")

        #2)探索性分析（日志记录）
        logger.info("\n2)探索性分析(日志记录)")
        #数据类型
        logger.info(f"数据类型分布: \n{df.dtypes.to_string()}")
        #缺失值统计
        missing_sum = df.isnull().sum()
        missing_rate = (missing_sum / len(df) * 100).round(2)
        logger.info(f"缺失值分布（列）：\n{missing_rate[missing_rate > 0].to_string()}")
        #描述性统计
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns
        if len(numeric_cols) > 0:
            logger.info(f"数值列描述性统计：\n{df[numeric_cols].describe().round(2).to_string()}")
        
        #3)去重
        logger.info("\n3)去重处理")
        duplicate_count = df.duplicated().sum()
        duplicate_rate = (duplicate_count / len(df) * 100).round(2)
        logger.info(f"重复行数量：{duplicate_count} | 重复行占比：{duplicate_rate}%")

        if duplicate_rate > duplicate_threshold:
            raise ValueError(f"重复行占比（{duplicate_rate}%）超过阈值（{duplicate_threshold}%）,终止清洗")
        elif duplicate_count > 0:
            df = df.drop_duplicates(keep='first')
            logger.info(f"已删除重复行，当前数据维度：{df.shape[0]}行 * {df.shape[1]}列")
        
        #4)缺失值处理
        logger.info("\n4)缺失值处理")
        for col in df.columns:
            #跳过没有缺失列
            if df[col].isnull().sum() == 0:
                continue
            #先处理缺失率超过阈值的列
            col_missing_rate = (df[col].isnull().sum() / len(df) *100).round(2)
            if col_missing_rate > missing_col_threshold:
                logger.info(f"列[{col}]缺失率{col_missing_rate}% > 阈值{missing_col_threshold}%,删除该列")
                df = df.drop(columns=[col])
                continue

            #处理列内缺失值
            if missing_fill_strategy == "drop":
                df = df.dropna(subset=[col])
                logger.info(f"列[{col}]:删除缺失行，当前行数：{len(df)}")
            else:
                #根据策略选择填充值
                if df[col].dtype in ['int64','float64']:
                    if missing_fill_strategy == "mean":
                        fill_val = df[col].mean().round(2)
                    elif missing_fill_strategy == "median":
                        fill_val = df[col].median()
                    else:   #auto/默认
                        fill_val = df[col].median()
                else:
                    fill_val = df[col].mode()[0] #类别列用众数

                df[col] = df[col].fillna(fill_val)
                logger.info(f"列[{col}]：填充缺失值（策略={missing_fill_strategy} | 填充值={fill_val}）")

                #5)异常值处理（仅数值列）
                logger.info("\n5)异常值处理")
                for col in numeric_cols:
                    if col not in df.columns:   #避免列已被删除
                        continue

                    #异常值判定
                    if outlier_method == "3σ":
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        lower_bound = mean_val - 3*std_val
                        upper_bound = mean_val + 3*std_val
                    else:   #IQR(默认)
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5*IQR
                        upper_bound = Q3 + 1.5*IQR

                    #筛选异常值
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outlier_count = len(outliers)
                    outlier_rate = (outlier_count / len(df) * 100).round(2)
                    logger.info(f"列[{col}]：异常值数量={outlier_count} | 占比={outlier_rate}% | 判定范围=[{lower_bound:.2f},{upper_bound:.2f}]")

                    #异常值处理：占比<=阈值则删除，超过则仅提示
                    if outlier_rate > 0:
                        if outlier_rate <= outlier_threshold:
                            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                            logger.info(f"列[{col}]：已删除异常行，当前行数：{len(df)}")
                        else:
                            logger.warning(f"列[{col}]：异常值占比超过阈值（{outlier_threshold}%）,请排查数据采集问题，暂不处理")

                #6)格式标准化
                logger.info("6)格式标准化")
                #字符串列：去空格、同一大写
                str_cols = df.select_dtypes(include=['object']).columns
                for col in str_cols:
                    df[col] = df[col].astype(str).str.strip().str.upper()
                    logger.info(f"列[{col}]：完成字符串标准化（去空格+大写）")

                #时间列：自动识别并标准化
                time_cols = [col for col in df.columns if any(key in col.lower() for key in ['time','date','dt'])]
                for col in time_cols:
                    df[col] = pd.to_datetime(df[col],errors='coerce')   #coerce=“强制”，转换失败的单个值会被设为NaT（Not a Time，时间类型的缺失值），而不是终止程序；
                    logger.info(f"列[{col}]：标准化为datetime格式")
                
                #7)数据保存
                logger.info("\n7)数据保存")
                df.to_csv(output_path,index=False,encoding='utf-8')
                final_shape = df.shape
                logger.info(f"清洗完成！输出文件：{output_path}")
                logger.info(f"最终数据维度：{final_shape[0]}行 * {final_shape[1]}列")
                logger.info(f"数据清洗总览：删除重复行{original_shape[0]-df.shape[0]}行 | 保留列{final_shape[1]}列")
                logger.info("="*50)
                
                return df,log_file
        
    except Exception as e:
        logger.error(f"清洗过程出错：{str(e)}",exc_info=True)
        raise       

# ===================== 3. 测试：学生成绩数据 =====================
def generate_test_student_data(test_path="学生成绩_原始数据.csv"):
    """生成模拟的学生成绩测试数据（包含重复、缺失、异常、格式问题）"""
    # 构造测试数据
    data = {
        "学号": ["2024001", "2024002", "2024003", "2024004", "2024005", "2024002", "2024006", "2024007", "2024008", "2024009", "2024010"],
        "姓名": [" 张三 ", "李四", "王五", "赵六", "钱七", "李四", "孙八", "周九", "吴十", "郑十一", "冯十二"],
        "语文": [85, 92, np.nan, 78, 88, 92, 95, 80, 75, 89, 82],  # 1个缺失值（10%）
        "数学": [90, np.nan, 85, np.nan, 78, np.nan, 200, np.nan, 88, np.nan, 95],  # 5个缺失值（45%）+ 异常值200
        "英语": [78, 85, 90, 82, 88, 85, 92, 75, 80, 89, np.nan],  # 1个缺失值（10%）
        "考试日期": ["2024-06-20", "2024/06/20", "2024.06.20", "2024-06-20", "2024-06-20", "2024/06/20", "2024-06-20", "2024-06-20", "2024-06-20", "2024-06-20", "2024-06-20"],
        "班级": [" 一班 ", "二班", "三班", "一班", "二班", "二班", "三班", "一班", "二班", "三班", "一班"]
    }
    df = pd.DataFrame(data)
    df.to_csv(test_path, index=False, encoding='utf-8')
    print(f"测试数据已生成：{test_path}")
    return test_path

# 测试主函数
if __name__ == "__main__":
    # 1. 生成测试数据
    test_input = generate_test_student_data()
    # 2. 配置清洗参数
    test_output = "学生成绩_清洗后数据.csv"
    # 3. 执行清洗
    try:
        cleaned_df, log_file = clean_csv_data(
            input_path=test_input,
            output_path=test_output,
            duplicate_threshold=5.0,
            missing_fill_strategy="auto",
            missing_col_threshold=30.0,
            outlier_method="IQR",
            outlier_threshold=5.0
        )
        print(f"\n测试完成！")
        print(f"清洗后数据预览：\n{cleaned_df.head()}")
        print(f"日志文件路径：{log_file}")
    except Exception as e:
        print(f"测试失败：{str(e)}")