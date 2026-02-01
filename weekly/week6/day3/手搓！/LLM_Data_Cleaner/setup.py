from setuptools import setup,find_packages
import os

def read_requirements():
    """读取依赖列表"""
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# 读取README（可选）
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return ""

setup(
    # 包名
    name="LLM_Data_Cleaner",
    # 版本号
    version="1.0.0",
    # 作者信息
    author="Loria123",
    author_email="",
    # 描述
    description="大模型数据清洗工具箱 V1.0，整合基础清洗、文本专项、结构化处理、批量自动化、质量评估",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    # 包路径
    packages=find_packages(),
    # 依赖
    install_requires=read_requirements(),
    # Python版本要求
    python_requires=">=3.7",
    # 分类
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # 关键词
    keywords=["LLM", "data cleaning", "文本清洗", "结构化处理", "质量评估"],
    # # 入口点（可选，命令行工具）
    # entry_points={
    #     "console_scripts": [
    #         "llm-clean = llm_data_cleaner.batch_auto:batch_process_files",
    #     ]
    # }
)