import os
import csv
import random
import time

# 生成模拟数据集的保存目录
DATA_DIR = "./simulated_datasets"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 生成 10 个 CSV 数据集（包含脏数据：缺失值、异常值、重复值）
def generate_simulated_datasets(num_datasets=10, rows_per_dataset=10000):
    for dataset_id in range(1, num_datasets + 1):
        file_path = os.path.join(DATA_DIR, f"dataset_{dataset_id}.csv")
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(["id", "name", "age", "salary"])
            # 写入数据（包含脏数据）
            for row_id in range(1, rows_per_dataset + 1):
                name = f"User_{row_id}_{random.randint(1000, 9999)}"
                # 模拟年龄缺失值（10% 概率）和异常值（大于 150 或小于 0）
                age = random.choice([None, random.randint(-10, 200)]) if random.random() < 0.1 else random.randint(18, 60)
                # 模拟薪资缺失值和异常值
                salary = None if random.random() < 0.05 else random.randint(3000, 50000)
                writer.writerow([row_id, name, age, salary])
    print(f"已生成 {num_datasets} 个模拟数据集，保存在 {DATA_DIR} 目录下")

# 执行生成（每个数据集 1 万行，10 个共 10 万行，适合做速度对比）
generate_simulated_datasets(num_datasets=10, rows_per_dataset=10000)

# 定义数据清洗函数（核心业务逻辑）
def clean_single_dataset(file_path, output_dir="./cleaned_datasets"):
    """
    清洗单个数据集（单文件处理逻辑）
    :param file_path: 原始数据集文件路径
    :param output_dir: 清洗结果保存目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 提取文件名，用于保存清洗结果
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, f"cleaned_{file_name}")
    
    try:
        cleaned_data = []
        seen_ids = set()  # 用于去重（记录已出现的 id）
        
        # 读取原始数据并清洗
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 1. 跳过缺失值
                if not row["age"] or not row["salary"]:
                    continue
                
                # 2. 类型转换（避免后续数值判断报错）
                try:
                    row["age"] = int(row["age"])
                    row["salary"] = int(row["salary"])
                except ValueError:
                    continue
                
                # 3. 修正异常值
                if not (18 <= row["age"] <= 60):
                    continue
                if not (3000 <= row["salary"] <= 50000):
                    continue
                
                # 4. 去重（基于 id）
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    cleaned_data.append(row)
        
        # 5. 保存清洗后的数据
        with open(output_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "age", "salary"])
            writer.writeheader()
            writer.writerows(cleaned_data)
        
        # 模拟轻微的处理耗时（更贴近真实场景，方便观察速度差异）
        time.sleep(0.1)
        return f"✅ 清洗完成：{file_name} -> 输出到 {output_file_path}"
    
    except Exception as e:
        return f"❌ 清洗失败：{file_name}，错误信息：{str(e)}"
    
# 单线程
def single_thread_cleaning(dataset_dir=DATA_DIR):
    """
    单线程清洗所有数据集（逐个处理）
    """
    # 获取所有 CSV 数据集文件路径
    dataset_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".csv")
    ]
    if not dataset_files:
        print("未找到需要清洗的数据集")
        return
    
    # 记录开始时间
    start_time = time.time()
    
    # 单线程逐个处理
    for file_path in dataset_files:
        result = clean_single_dataset(file_path)
        print(result)
    
    # 记录结束时间 & 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n📊 单线程处理完成！总耗时：{total_time:.2f} 秒")
    return total_time

# 执行单线程清洗（先运行这个，记录基准耗时）
# single_thread_time = single_thread_cleaning()

# 多线程
from concurrent.futures import ThreadPoolExecutor, as_completed

def multi_thread_cleaning_advanced(dataset_dir=DATA_DIR, thread_num=5):
    """
    高级多线程清洗（使用 ThreadPoolExecutor，推荐）
    :param thread_num: 线程数（核心优化参数）
    """
    dataset_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".csv")
    ]
    if not dataset_files:
        print("未找到需要清洗的数据集")
        return
    
    start_time = time.time()
    
    # 使用线程池管理线程
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        # 提交所有任务到线程池，返回任务对象与文件路径的映射
        future_to_file = {
            executor.submit(clean_single_dataset, file_path): file_path
            for file_path in dataset_files
        }
        
        # 遍历完成的任务，输出结果
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"❌ 处理 {os.path.basename(file_path)} 时发生异常：{str(e)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n📊 多线程（{thread_num} 线程池）处理完成！总耗时：{total_time:.2f} 秒")
    return total_time

# 执行高级多线程清洗（先试用 5 个线程）
# multi_thread_advanced_time = multi_thread_cleaning_advanced(thread_num=10)

# 对比
# 1. 先获取单线程耗时（如果已运行，可直接使用之前的 single_thread_time）
print("=" * 60)
print("开始执行单线程处理...")
single_time = single_thread_cleaning()

# 2. 测试不同线程数的多线程耗时（2/4/5/8/10）
thread_nums = [2, 4, 5, 8, 10]
multi_times = {}
print("\n" + "=" * 60)
print("开始执行不同线程数的多线程处理...")
for num in thread_nums:
    print(f"\n--- 正在测试 {num} 个线程 ---")
    multi_time = multi_thread_cleaning_advanced(thread_num=num)
    multi_times[num] = multi_time

# 3. 输出对比结果
print("\n" + "=" * 60)
print("📈 单线程 vs 多线程 耗时对比结果")
print(f"单线程总耗时：{single_time:.2f} 秒")
print("-" * 40)
for num, cost_time in multi_times.items():
    speedup_rate = (single_time - cost_time) / single_time * 100
    print(f"{num} 线程总耗时：{cost_time:.2f} 秒，提速 {speedup_rate:.2f}%")

# 自动化优化线程数
'''
def get_optimal_thread_num(dataset_count, max_thread_limit=20):
    """
    计算最优线程数
    :param dataset_count: 数据集数量
    :param max_thread_limit: 最大线程数限制（避免资源耗尽）
    :return: 推荐线程数
    """
    cpu_core_num = os.cpu_count()  # 获取 CPU 核心数（如 8 核）
    # IO 密集型任务推荐：CPU 核心数 * 2
    recommend_thread_num = cpu_core_num * 2
    
    # 最终线程数取 3 个值的最小值：推荐值、数据集数量、最大线程限制
    optimal_thread_num = min(recommend_thread_num, dataset_count, max_thread_limit)
    
    print(f"系统 CPU 核心数：{cpu_core_num}")
    print(f"推荐线程数：{recommend_thread_num}，最终最优线程数：{optimal_thread_num}")
    return optimal_thread_num

# 使用最优线程数执行清洗
dataset_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
optimal_threads = get_optimal_thread_num(dataset_count=len(dataset_files))
print("\n" + "=" * 60)
print(f"使用最优线程数 {optimal_threads} 执行清洗...")
optimal_multi_time = multi_thread_cleaning_advanced(thread_num=optimal_threads)
'''