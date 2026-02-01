import os
import shutil
import time
import csv
from datetime import datetime, timedelta
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. 核心清洗函数（复用）
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

# 2. 归档函数
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
            print(f"📦 原始数据归档完成：{file_name} -> {archive_raw_dir}")
    
    for cleaned_file in processed_cleaned_files:
        if os.path.exists(cleaned_file):
            file_name = os.path.basename(cleaned_file)
            target_path = os.path.join(archive_cleaned_dir, file_name)
            shutil.move(cleaned_file, target_path)
            print(f"📦 清洗结果归档完成：{file_name} -> {archive_cleaned_dir}")
    
    if os.path.exists("./temp_cleaned") and not os.listdir("./temp_cleaned"):
        os.rmdir("./temp_cleaned")
    
    print(f"\n🎉 全部归档完成！归档目录：{archive_dir}")
    return archive_dir

# 3. 自动化清洗 + 归档核心任务
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
            except Exception as e:
                print(f"❌ 处理 {os.path.basename(file_path)} 时发生未知异常：{str(e)}")
    
    if processed_raw_files and processed_cleaned_files:
        archive_processed_files(processed_raw_files, processed_cleaned_files)
    else:
        print("⚠️  无成功清洗的文件，无需归档")
    
    print("=" * 80)
    print(f"🏁 自动化清洗任务执行完毕，当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n\n")

# 4. 定时任务配置（每天凌晨 1 点）
def configure_scheduled_task():
    schedule.every().day.at("01:00").do(auto_clean_and_archive)
    
    print("=" * 80)
    print("⏰ 定时任务配置完成！")
    print(f"📌 任务规则：每天凌晨 1 点自动清洗 {os.path.abspath('./raw_datasets')} 目录下的新数据")
    print(f"📌 归档目录：{os.path.abspath('./archive_datasets')}")
    print("📌 脚本将持续运行，按 Ctrl+C 可终止程序")
    print("=" * 80 + "\n\n")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# 5. 测试函数 1：即时测试（快速验证核心功能）
def test_immediate_task():
    print("🚀 开始执行即时测试（直接运行清洗 + 归档任务）")
    auto_clean_and_archive()

# 6. 测试函数 2：快速定时测试（1 分钟后执行，验证定时触发）
def test_fast_scheduled_task():
    one_minute_later = (datetime.now() + timedelta(minutes=1)).strftime("%H:%M")
    schedule.every().day.at(one_minute_later).do(auto_clean_and_archive)
    
    print("=" * 80)
    print(f"⏰ 快速定时测试配置完成！")
    print(f"📌 任务将在 {one_minute_later} 自动执行（约 1 分钟后）")
    print(f"📌 脚本将持续运行，按 Ctrl+C 可终止程序")
    print("=" * 80 + "\n\n")
    
    while True:
        schedule.run_pending()
        time.sleep(10)

# 7. 脚本运行入口（按需选择运行模式）
if __name__ == "__main__":
    # 选择运行模式：取消注释对应行即可
    # 模式 1：即时测试（优先验证清洗 + 归档功能）
    # test_immediate_task()
    
    # 模式 2：快速定时测试（验证定时触发功能）
    # test_fast_scheduled_task()
    
    # 模式 3：正式环境（每天凌晨 1 点执行）
    configure_scheduled_task()