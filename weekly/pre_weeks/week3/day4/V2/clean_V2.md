### 1. 需求确认
你需要对通用清洗脚本V1.0做三大优化：① 强化异常处理（覆盖文件不存在、格式错误等场景）；② 增加`argparse`支持终端命令行运行；③ 适配不同异常场景测试（如传入Excel文件）。我会提供**完整优化后代码**，并详细说明关键优化点和异常场景测试方法。

### 2. 完整优化后脚本（V2.0）
```python
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import argparse  # 新增：命令行参数支持
import traceback  # 新增：详细异常栈追踪

# ===================== 1. 日志配置（保留并优化） =====================
def setup_logger(log_path):
    """配置日志：同时输出到控制台和日志文件，记录清洗全流程"""
    # 日志文件名包含时间戳，避免覆盖
    log_file = f"{log_path}_清洗日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志格式
    logger = logging.getLogger("data_cleaner")
    logger.setLevel(logging.INFO)
    # 清空已有处理器（避免重复打印）
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger, log_file

# ===================== 2. 核心清洗函数（强化异常处理） =====================
def clean_csv_data(
    input_path,          # 输入CSV文件路径
    output_path,         # 输出清洗后CSV路径
    log_path="清洗日志",  # 日志文件基础路径
    duplicate_threshold=5.0,  # 重复行占比阈值（%），超过则终止
    missing_fill_strategy="auto",  # 缺失值填充策略：auto/mean/median/mode/drop
    missing_col_threshold=30.0,    # 列缺失率阈值（%），超过则删除列
    outlier_method="IQR",          # 异常值判定方法：IQR/3σ
    outlier_threshold=5.0          # 异常值占比阈值（%），超过则提示
):
    """
    通用CSV数据清洗函数（支持参数配置+强化异常处理）
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
    :return: 清洗后DataFrame、日志文件路径 | 异常时返回None, 日志路径
    """
    # 初始化日志
    logger, log_file = setup_logger(log_path)
    logger.info("="*50)
    logger.info("开始执行数据清洗流程")
    logger.info(f"输入文件：{input_path}")
    logger.info(f"配置参数：重复行阈值={duplicate_threshold}% | 缺失列阈值={missing_col_threshold}% | 缺失填充策略={missing_fill_strategy} | 异常值方法={outlier_method}")
    logger.info("="*50)

    try:
        # -------------------- 新增：前置格式校验 --------------------
        logger.info("【前置校验】检查文件格式和存在性")
        # 1. 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在：{input_path}")
        
        # 2. 检查输入文件格式（仅支持CSV）
        if not input_path.lower().endswith('.csv'):
            raise ValueError(f"输入文件格式错误！仅支持CSV文件，当前文件：{input_path}")
        
        # 3. 检查输出文件格式（仅支持CSV）
        if not output_path.lower().endswith('.csv'):
            raise ValueError(f"输出文件格式错误！仅支持CSV文件，当前文件：{output_path}")

        # -------------------- 步骤1：数据加载 --------------------
        logger.info("【步骤1：数据加载】")
        df = pd.read_csv(input_path, encoding='utf-8')
        original_shape = df.shape
        logger.info(f"原始数据维度：{original_shape[0]}行 × {original_shape[1]}列")
        
        if df.empty:
            raise ValueError("加载的CSV文件为空，无数据可清洗")

        # -------------------- 步骤2：探索性分析（日志记录，修复重复打印） --------------------
        logger.info("\n【步骤2：探索性分析】")
        # 数据类型
        logger.info(f"数据类型分布：\n{df.dtypes.to_string()}")
        # 缺失值统计
        missing_sum = df.isnull().sum()
        missing_rate = (missing_sum / len(df) * 100).round(2)
        missing_info = missing_rate[missing_rate > 0].to_string() if any(missing_rate > 0) else "无缺失值"
        logger.info(f"缺失值分布（列）：\n{missing_info}")
        # 描述性统计（仅打印一次）
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            desc_stats = df[numeric_cols].describe().round(2).to_string()
            logger.info(f"数值列描述性统计：\n{desc_stats}")

        # -------------------- 步骤3：去重 --------------------
        logger.info("\n【步骤3：去重处理】")
        duplicate_count = df.duplicated().sum()
        duplicate_rate = (duplicate_count / len(df) * 100).round(2)
        logger.info(f"重复行数量：{duplicate_count} | 重复行占比：{duplicate_rate}%")
        
        if duplicate_rate > duplicate_threshold:
            raise ValueError(f"重复行占比（{duplicate_rate}%）超过阈值（{duplicate_threshold}%），终止清洗")
        elif duplicate_count > 0:
            df = df.drop_duplicates(keep='first')
            logger.info(f"已删除重复行，当前数据维度：{df.shape[0]}行 × {df.shape[1]}列")

        # -------------------- 步骤4：缺失值处理 --------------------
        logger.info("\n【步骤4：缺失值处理】")
        # 先处理列缺失率超过阈值的列
        for col in df.columns:
            col_missing_rate = (df[col].isnull().sum() / len(df) * 100).round(2)
            if col_missing_rate > missing_col_threshold:
                logger.info(f"列[{col}]缺失率{col_missing_rate}% > 阈值{missing_col_threshold}%，删除该列")
                df = df.drop(columns=[col])
                continue
            
            # 处理列内缺失值
            if df[col].isnull().sum() == 0:
                continue
            
            if missing_fill_strategy == "drop":
                df = df.dropna(subset=[col])
                logger.info(f"列[{col}]：删除缺失行，当前行数：{len(df)}")
            else:
                # 根据策略选择填充值
                if df[col].dtype in ['int64', 'float64']:
                    if missing_fill_strategy == "mean":
                        fill_val = df[col].mean().round(2)
                    elif missing_fill_strategy == "median":
                        fill_val = df[col].median()
                    else:  # auto/默认
                        fill_val = df[col].median()
                else:
                    fill_val = df[col].mode()[0]  # 类别列用众数
                
                df[col] = df[col].fillna(fill_val)
                logger.info(f"列[{col}]：填充缺失值（策略={missing_fill_strategy} | 填充值={fill_val}）")

        # -------------------- 步骤5：异常值处理（仅数值列） --------------------
        logger.info("\n【步骤5：异常值处理】")
        for col in numeric_cols:
            if col not in df.columns:  # 避免列已被删除
                continue
            
            # 异常值判定
            if outlier_method == "3σ":
                mean_val = df[col].mean()
                std_val = df[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
            else:  # IQR（默认）
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            
            # 筛选异常值
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_rate = (outlier_count / len(df) * 100).round(2)
            logger.info(f"列[{col}]：异常值数量={outlier_count} | 占比={outlier_rate}% | 判定范围=[{lower_bound:.2f}, {upper_bound:.2f}]")
            
            # 异常值处理：占比≤阈值则删除，超过则仅提示
            if outlier_rate > 0:
                if outlier_rate <= outlier_threshold:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    logger.info(f"列[{col}]：已删除异常行，当前行数：{len(df)}")
                else:
                    logger.warning(f"列[{col}]：异常值占比超过阈值（{outlier_threshold}%），请排查数据采集问题，暂不处理")

        # -------------------- 步骤6：格式标准化 --------------------
        logger.info("\n【步骤6：格式标准化】")
        # 字符串列：去空格、统一大写
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().str.upper()
            logger.info(f"列[{col}]：完成字符串标准化（去空格+大写）")
        
        # 时间列：自动识别并标准化
        time_cols = [col for col in df.columns if any(key in col.lower() for key in ['time', 'date', 'dt'])]
        for col in time_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logger.info(f"列[{col}]：标准化为datetime格式")

        # -------------------- 步骤7：数据保存 --------------------
        logger.info("\n【步骤7：数据保存】")
        # 新增：检查输出目录是否存在，不存在则创建
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"输出目录不存在，已创建：{output_dir}")
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        final_shape = df.shape
        logger.info(f"清洗完成！输出文件：{output_path}")
        logger.info(f"最终数据维度：{final_shape[0]}行 × {final_shape[1]}列")
        logger.info(f"数据清洗总览：删除重复行{original_shape[0]-df.shape[0]}行 | 保留列{final_shape[1]}列")
        logger.info("="*50)

        return df, log_file

    except Exception as e:
        # 新增：详细记录异常栈，便于排查
        logger.error(f"清洗过程出错：{str(e)}", exc_info=True)
        logger.error(f"异常详细栈信息：\n{traceback.format_exc()}")
        return None, log_file

# ===================== 3. 命令行参数配置（新增核心） =====================
def parse_args():
    """配置命令行参数，支持终端直接运行"""
    parser = argparse.ArgumentParser(description="通用CSV数据清洗脚本 V2.0（支持命令行参数+强化异常处理）")
    
    # 必选参数
    parser.add_argument('-i', '--input', required=True, help="输入CSV文件路径（必填），示例：./原始数据.csv")
    parser.add_argument('-o', '--output', required=True, help="输出清洗后CSV文件路径（必填），示例：./清洗后数据.csv")
    
    # 可选参数（均有默认值）
    parser.add_argument('-l', '--log', default="清洗日志", help="日志文件基础路径，默认：清洗日志")
    parser.add_argument('-dt', '--duplicate_threshold', type=float, default=5.0, help="重复行占比阈值（%），默认：5.0")
    parser.add_argument('-mfs', '--missing_fill_strategy', choices=['auto', 'mean', 'median', 'mode', 'drop'], 
                        default='auto', help="缺失值填充策略，可选：auto/mean/median/mode/drop，默认：auto")
    parser.add_argument('-mct', '--missing_col_threshold', type=float, default=30.0, help="列缺失率阈值（%），默认：30.0")
    parser.add_argument('-om', '--outlier_method', choices=['IQR', '3σ'], default='IQR', help="异常值判定方法，可选：IQR/3σ，默认：IQR")
    parser.add_argument('-ot', '--outlier_threshold', type=float, default=5.0, help="异常值占比阈值（%），默认：5.0")
    
    return parser.parse_args()

# ===================== 4. 测试数据生成（保留） =====================
def generate_test_student_data(test_path="学生成绩_原始数据.csv"):
    """生成模拟的学生成绩测试数据（包含重复、缺失、异常、格式问题）"""
    data = {
        "学号": ["2024001", "2024002", "2024003", "2024004", "2024005", "2024002", "2024006", "2024007", "2024008", "2024009", "2024010"],
        "姓名": [" 张三 ", "李四", "王五", "赵六", "钱七", "李四", "孙八", "周九", "吴十", "郑十一", "冯十二"],
        "语文": [85, 92, np.nan, 78, 88, 92, 95, 80, 75, 89, 82],
        "数学": [90, np.nan, 85, np.nan, 78, np.nan, 200, np.nan, 88, np.nan, 95],
        "英语": [78, 85, 90, 82, 88, 85, 92, 75, 80, 89, np.nan],
        "考试日期": ["2024-06-20", "2024/06/20", "2024.06.20", "2024-06-20", "2024-06-20", "2024/06/20", "2024-06-20", "2024-06-20", "2024-06-20", "2024-06-20", "2024-06-20"],
        "班级": [" 一班 ", "二班", "三班", "一班", "二班", "二班", "三班", "一班", "二班", "三班", "一班"]
    }
    df = pd.DataFrame(data)
    df.to_csv(test_path, index=False, encoding='utf-8')
    print(f"测试数据已生成：{test_path}")
    return test_path

# ===================== 5. 主函数（整合命令行+清洗逻辑） =====================
def main():
    """主函数：解析命令行参数 → 执行清洗 → 输出结果"""
    # 解析命令行参数
    args = parse_args()
    
    # 执行清洗
    print("="*60)
    print("开始执行通用CSV数据清洗脚本 V2.0")
    print(f"输入文件：{args.input}")
    print(f"输出文件：{args.output}")
    print("="*60)
    
    cleaned_df, log_file = clean_csv_data(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        duplicate_threshold=args.duplicate_threshold,
        missing_fill_strategy=args.missing_fill_strategy,
        missing_col_threshold=args.missing_col_threshold,
        outlier_method=args.outlier_method,
        outlier_threshold=args.outlier_threshold
    )
    
    # 输出最终结果
    if cleaned_df is not None:
        print(f"\n✅ 清洗成功！")
        print(f"📄 清洗后数据预览：\n{cleaned_df.head()}")
        print(f"📝 日志文件路径：{log_file}")
    else:
        print(f"\n❌ 清洗失败！")
        print(f"📝 错误日志已保存至：{log_file}，请查看日志排查问题")

# ===================== 6. 入口函数（支持命令行+脚本运行） =====================
if __name__ == "__main__":
    # 方式1：命令行运行（优先）
    try:
        main()
    # 方式2：脚本直接运行（测试用，保留原测试逻辑）
    except SystemExit:
        # 若未传入命令行参数，自动生成测试数据并运行
        print("\n⚠️  未传入命令行参数，自动执行测试模式...")
        # 1. 生成测试数据
        test_input = generate_test_student_data()
        # 2. 配置清洗参数（调高重复行阈值，避免终止）
        test_output = "学生成绩_清洗后数据.csv"
        # 3. 执行清洗
        cleaned_df, log_file = clean_csv_data(
            input_path=test_input,
            output_path=test_output,
            duplicate_threshold=10.0,  # 调高阈值，适配测试数据的9.09%重复率
            missing_fill_strategy="auto",
            missing_col_threshold=30.0,
            outlier_method="IQR",
            outlier_threshold=5.0
        )
        # 输出测试结果
        if cleaned_df is not None:
            print(f"\n✅ 测试模式 - 清洗成功！")
            print(f"📄 清洗后数据预览：\n{cleaned_df.head()}")
            print(f"📝 日志文件路径：{log_file}")
        else:
            print(f"\n❌ 测试模式 - 清洗失败！")
            print(f"📝 错误日志已保存至：{log_file}")
    except Exception as e:
        print(f"\n❌ 脚本运行出错：{str(e)}")
        print(f"📝 异常详情：\n{traceback.format_exc()}")
```

### 3. 核心优化点详解
#### （1）强化异常处理（覆盖关键场景）
| 异常场景                | 处理方式                                                                 | 优化效果                                                                 |
|-------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| 文件不存在              | 捕获`FileNotFoundError`，日志记录并友好提示                             | 避免脚本崩溃，明确告知文件路径错误                                       |
| 格式错误（传入Excel）| 前置校验`input_path`是否以`.csv`结尾，抛出`ValueError`并记录            | 提前拦截非CSV文件，避免`pd.read_csv`读取失败的模糊错误                   |
| 输出路径目录不存在      | 自动创建输出目录（`os.makedirs`）                                       | 避免因目录不存在导致保存失败                                             |
| 空数据                  | 检测`df.empty`，抛出`ValueError`                                         | 避免后续清洗逻辑处理空数据                                             |
| 通用运行时异常          | 捕获所有`Exception`，记录完整异常栈（`traceback.format_exc()`）         | 便于定位具体错误行，快速排查问题                                         |
| 日志重复打印            | 清空日志器已有处理器（`logger.handlers.clear()`）                       | 修复原脚本中描述性统计重复打印的问题                                     |

#### （2）命令行参数支持（argparse）
脚本新增`parse_args()`函数，支持终端直接运行，核心参数说明：
```bash
# 终端运行示例（基础用法）
python 清洗脚本V2.0.py -i 学生成绩_原始数据.csv -o 学生成绩_清洗后.csv

# 完整参数示例（自定义阈值）
python 清洗脚本V2.0.py \
  -i ./data/原始数据.csv \
  -o ./data/清洗后数据.csv \
  -l ./logs/学生成绩 \
  -dt 10.0 \
  -mfs median \
  -mct 20.0 \
  -om 3σ \
  -ot 2.0
```
参数说明：
| 参数                | 简写 | 作用                                  | 默认值  |
|---------------------|------|---------------------------------------|---------|
| `--input`           | `-i` | 输入CSV路径（必填）                   | 无      |
| `--output`          | `-o` | 输出CSV路径（必填）                   | 无      |
| `--log`             | `-l` | 日志基础路径                          | 清洗日志 |
| `--duplicate_threshold` | `-dt` | 重复行占比阈值（%） | 5.0     |
| `--missing_fill_strategy` | `-mfs` | 缺失值填充策略 | auto    |
| `--missing_col_threshold` | `-mct` | 列缺失率阈值（%） | 30.0    |
| `--outlier_method`  | `-om` | 异常值判定方法（IQR/3σ）              | IQR     |
| `--outlier_threshold` | `-ot` | 异常值占比阈值（%） | 5.0     |

#### （3）适配测试模式
脚本入口函数做了兼容：
- 若传入命令行参数 → 按命令行模式运行；
- 若未传参数 → 自动生成测试数据，并**调高重复行阈值至10.0%**（适配测试数据的9.09%重复率），避免测试终止。

### 4. 异常场景测试（验证优化效果）
#### 场景1：传入Excel文件（格式错误）
```bash
# 1. 先生成一个Excel测试文件（模拟错误输入）
python -c "import pandas as pd; pd.DataFrame({'a':[1,2]}).to_excel('错误数据.xlsx', index=False)"

# 2. 用脚本清洗该Excel文件
python 清洗脚本V2.0.py -i 错误数据.xlsx -o 清洗后.csv
```
**预期结果**：
- 控制台提示`❌ 清洗失败！`；
- 日志中记录`输入文件格式错误！仅支持CSV文件，当前文件：错误数据.xlsx`；
- 无崩溃，错误信息清晰。

#### 场景2：输入文件不存在
```bash
python 清洗脚本V2.0.py -i 不存在的文件.csv -o 清洗后.csv
```
**预期结果**：
- 控制台提示`❌ 清洗失败！`；
- 日志中记录`输入文件不存在：不存在的文件.csv`；
- 完整异常栈记录，便于排查。

#### 场景3：重复行占比超阈值
```bash
# 用默认阈值5.0%清洗测试数据（重复率9.09%）
python 清洗脚本V2.0.py -i 学生成绩_原始数据.csv -o 清洗后.csv -dt 5.0
```
**预期结果**：
- 控制台提示`❌ 清洗失败！`；
- 日志中记录`重复行占比（9.09%）超过阈值（5.0%），终止清洗`；
- 错误原因明确，可通过调高`-dt`参数解决。

#### 场景4：输出目录不存在
```bash
# 输出路径指定不存在的目录
python 清洗脚本V2.0.py -i 学生成绩_原始数据.csv -o ./不存在的目录/清洗后.csv -dt 10.0
```
**预期结果**：
- 脚本自动创建`不存在的目录`；
- 清洗成功，数据保存至该目录；
- 日志中记录`输出目录不存在，已创建：./不存在的目录`。

### 5. 总结
1. **核心功能**：保留原清洗流程的所有核心逻辑，强化了异常处理的覆盖范围和友好性；
2. **命令行支持**：新增`argparse`，可通过终端灵活配置参数，适配生产环境运行；
3. **异常测试**：覆盖格式错误、文件不存在、目录不存在、重复行超阈值等场景，错误信息清晰、日志可追溯；
4. **兼容性**：同时支持命令行运行（生产）和脚本直接运行（测试），兼顾不同使用场景。

关键优化记忆点：
- 异常处理：前置校验+完整栈记录，避免崩溃且易排查；
- 命令行：`-i/-o`必填，其他参数可选，终端运行更灵活；
- 测试兼容：未传参数自动执行测试模式，调高重复行阈值避免终止。