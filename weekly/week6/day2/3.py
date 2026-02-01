import os
import shutil
import time
import csv
from datetime import datetime, timedelta
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------- 1. 辅助函数：生成模拟数据（用于测试，提供10个数据集）----------------------
def generate_simulated_datasets(num_datasets=10, rows_per_dataset=10000, save_dir="./raw_datasets"):
    """生成模拟数据集（直接保存到待清洗目录，用于测试）"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for dataset_id in range(1, num_datasets + 1):
        file_path = os.path.join(save_dir, f"dataset_{dataset_id}.csv")
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "age", "salary"])
            for row_id in range(1, rows_per_dataset + 1):
                name = f"User_{row_id}_{int(time.time()) % 9999}"
                age = random.choice([None, str(random.randint(-10, 200))]) if random.random() < 0.1 else str(random.randint(18, 60))
                salary = None if random.random() < 0.05 else str(random.randint(3000, 50000))
                writer.writerow([str(row_id), name, age, salary])
    
    print(f"✅ 已生成 {num_datasets} 个模拟数据集，保存在 {save_dir} 目录下")

# ---------------------- 2. 核心清洗函数 ----------------------
def clean_single_dataset(file_path, output_dir="./temp_cleaned"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, f"cleaned_{file_name}")
    
    try:
        cleaned_data = []
        seen_ids = set()
        
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row["age"] or not row["salary"]:
                    continue
                try:
                    row["age"] = int(row["age"])
                    row["salary"] = int(row["salary"])
                except ValueError:
                    continue
                if not (18 <= row["age"] <= 60) or not (3000 <= row["salary"] <= 50000):
                    continue
                if row["id"] not in seen_ids:
                    seen_ids.add(row["id"])
                    cleaned_data.append(row)
        
        with open(output_file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "age", "salary"])
            writer.writeheader()
            writer.writerows(cleaned_data)
        
        time.sleep(0.1)
        return True, f"✅ 清洗完成：{file_name}", file_path, output_file_path
    except Exception as e:
        return False, f"❌ 清洗失败：{file_name}，错误信息：{str(e)}", file_path, None

# ---------------------- 3. 归档函数 ----------------------
def archive_processed_files(processed_raw_files, processed_cleaned_files, archive_root="./archive_datasets"):
    today_date = datetime.now().strftime("%Y-%m-%d")
    archive_dir = os.path.join(archive_root, today_date)
    archive_raw_dir = os.path.join(archive_dir, "raw")
    archive_cleaned_dir = os.path.join(archive_dir, "cleaned")
    
    for dir_path in [archive_dir, archive_raw_dir, archive_cleaned_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    for raw_file in processed_raw_files:
        if os.path.exists(raw_file):
            file_name = os.path.basename(raw_file)
            target_path = os.path.join(archive_raw_dir, file_name)
            shutil.move(raw_file, target_path)
    
    for cleaned_file in processed_cleaned_files:
        if os.path.exists(cleaned_file):
            file_name = os.path.basename(cleaned_file)
            target_path = os.path.join(archive_cleaned_dir, file_name)
            shutil.move(cleaned_file, target_path)
    
    if os.path.exists("./temp_cleaned") and not os.listdir("./temp_cleaned"):
        os.rmdir("./temp_cleaned")
    
    print(f"\n🎉 全部归档完成！归档目录：{archive_dir}")
    return archive_dir

# ---------------------- 4. 质量评估辅助函数 ----------------------
def read_csv_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    
    data_list = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data_list = [row for row in reader]
    except Exception as e:
        raise Exception(f"读取CSV文件失败：{str(e)}")
    
    return data_list, len(data_list)

def evaluate_clean_quality(raw_file_path, cleaned_file_path):
    try:
        raw_data, raw_row_count = read_csv_data(raw_file_path)
        cleaned_data, cleaned_row_count = read_csv_data(cleaned_file_path)
        
        # 计算清洗率
        clean_rate = (cleaned_row_count / raw_row_count) * 100 if raw_row_count > 0 else 0.0
        
        # 计算有效数据占比
        valid_cleaned_count = 0
        seen_ids_in_cleaned = set()
        for row in cleaned_data:
            is_valid = True
            if not row.get("age") or not row.get("salary"):
                is_valid = False
            try:
                age = int(row.get("age", 0))
                salary = int(row.get("salary", 0))
                if not (18 <= age <= 60) or not (3000 <= salary <= 50000):
                    is_valid = False
            except (ValueError, TypeError):
                is_valid = False
            row_id = row.get("id")
            if row_id in seen_ids_in_cleaned:
                is_valid = False
            else:
                seen_ids_in_cleaned.add(row_id)
            if is_valid:
                valid_cleaned_count += 1
        valid_data_ratio = (valid_cleaned_count / cleaned_row_count) * 100 if cleaned_row_count > 0 else 0.0
        
        # 计算格式合规率
        compliant_count = 0
        for row in cleaned_data:
            is_compliant = True
            required_fields = ["id", "age", "salary"]
            for field in required_fields:
                try:
                    int(row.get(field, ""))
                except (ValueError, TypeError):
                    is_compliant = False
                    break
            name = row.get("name", "")
            if not name or not all(c.isalnum() or c == "_" for c in name):
                is_compliant = False
            if is_compliant:
                compliant_count += 1
        format_compliance_rate = (compliant_count / cleaned_row_count) * 100 if cleaned_row_count > 0 else 0.0
        
        return {
            "basic_info": {
                "raw_file": os.path.basename(raw_file_path),
                "cleaned_file": os.path.basename(cleaned_file_path),
                "raw_row_count": raw_row_count,
                "cleaned_row_count": cleaned_row_count
            },
            "metrics": {
                "clean_rate": round(clean_rate, 2),
                "valid_data_ratio": round(valid_data_ratio, 2),
                "format_compliance_rate": round(format_compliance_rate, 2)
            }
        }
    except Exception as e:
        return {"error": f"质量评估失败：{str(e)}", "raw_file": os.path.basename(raw_file_path) if os.path.exists(raw_file_path) else "未知文件"}

def generate_quality_report(quality_result, report_dir="./archive_datasets/quality_reports"):
    if "error" in quality_result:
        print(f"❌ 无法生成报告：{quality_result['error']}")
        return None
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    daily_report_dir = os.path.join(report_dir, today_date)
    if not os.path.exists(daily_report_dir):
        os.makedirs(daily_report_dir)
    
    raw_file_name = quality_result["basic_info"]["raw_file"]
    report_file_name = f"quality_report_{raw_file_name.replace('.csv', '.txt')}"
    report_file_path = os.path.join(daily_report_dir, report_file_name)
    
    try:
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"📊 数据清洗质量评估报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"原始数据文件：{quality_result['basic_info']['raw_file']}\n")
            f.write(f"清洗后文件：{quality_result['basic_info']['cleaned_file']}\n")
            f.write(f"原始数据行数（排除表头）：{quality_result['basic_info']['raw_row_count']}\n")
            f.write(f"清洗后数据行数（排除表头）：{quality_result['basic_info']['cleaned_row_count']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"核心质量指标\n")
            f.write("-" * 60 + "\n")
            f.write(f"1. 清洗率：{quality_result['metrics']['clean_rate']}% \n")
            f.write(f"   （说明：清洗后有效数据占原始数据的比例，越高表示原始数据质量越好）\n")
            f.write(f"2. 有效数据占比：{quality_result['metrics']['valid_data_ratio']}% \n")
            f.write(f"   （说明：清洗后无缺失、无异常、无重复的数据比例，理想值为100%）\n")
            f.write(f"3. 格式合规率：{quality_result['metrics']['format_compliance_rate']}% \n")
            f.write(f"   （说明：清洗后字段格式完全规范的数据比例，理想值为100%）\n")
            f.write("=" * 60 + "\n")
        
        print(f"📄 质量报告生成完成：{report_file_name}")
        return report_file_path
    except Exception as e:
        print(f"❌ 生成质量报告失败：{str(e)}")
        return None

# ---------------------- 5. 自动化清洗+评估+报告+归档（更新版）----------------------
def auto_clean_and_archive(raw_dir="./raw_datasets", temp_cleaned_dir="./temp_cleaned"):
    print("=" * 80)
    print(f"📅 开始执行自动化清洗任务，当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        print(f"⚠️  待清洗目录不存在，已创建：{raw_dir}，当前无新数据需要清洗")
        return
    
    raw_dataset_files = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.endswith(".csv") and os.path.isfile(os.path.join(raw_dir, f))
    ]
    
    if not raw_dataset_files:
        print("⚠️  待清洗目录下无新的 CSV 数据集，无需执行清洗任务")
        return
    
    print(f"🔍 发现 {len(raw_dataset_files)} 个待清洗的新数据集，开始批量清洗...")
    
    processed_raw_files = []
    processed_cleaned_files = []
    generated_reports = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        future_to_file = {
            executor.submit(clean_single_dataset, file_path, temp_cleaned_dir): file_path
            for file_path in raw_dataset_files
        }
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                success, result_msg, raw_file, cleaned_file = future.result()
                print(result_msg)
                if success and raw_file and cleaned_file:
                    processed_raw_files.append(raw_file)
                    processed_cleaned_files.append(cleaned_file)
                    
                    # 质量评估 + 生成报告
                    print(f"🔎 评估 {os.path.basename(raw_file)} 质量...")
                    quality_result = evaluate_clean_quality(raw_file, cleaned_file)
                    report_path = generate_quality_report(quality_result)
                    if report_path:
                        generated_reports.append(report_path)
            except Exception as e:
                print(f"❌ 处理 {os.path.basename(file_path)} 异常：{str(e)}")
    
    if processed_raw_files and processed_cleaned_files:
        archive_processed_files(processed_raw_files, processed_cleaned_files)
    
    if generated_reports:
        print(f"\n📊 本次共生成 {len(generated_reports)} 份质量报告，保存在 ./archive_datasets/quality_reports/")
    else:
        print("⚠️  无成功生成的质量报告")
    
    print("=" * 80)
    print(f"🏁 自动化清洗任务执行完毕，当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n\n")

# ---------------------- 6. 测试函数：生成10个数据集+批量生成报告 ----------------------
def test_batch_10_reports():
    """测试：生成10个模拟数据集，批量清洗并生成10份质量报告"""
    # 步骤 1：生成10个模拟数据集（直接保存到待清洗目录）
    generate_simulated_datasets(num_datasets=10, rows_per_dataset=10000)
    
    # 步骤 2：执行自动化清洗+质量评估+报告生成
    auto_clean_and_archive()

# ---------------------- 7. 运行入口 ----------------------
if __name__ == "__main__":
    # 导入random（生成模拟数据需要）
    import random
    # 运行测试，生成10份质量报告
    test_batch_10_reports()