import json
import random
import os
from datasets import load_dataset
from tqdm import tqdm

# ================= 配置区域 =================

# 输出文件名
OUTPUT_FILE = "mixed_finetune_dataset.jsonl"
MY_REPORT_DATA_PATH = "my_dataset.json" 

# 数据源配置
DATASETS_CONFIG = {
    "orca": "Open-Orca/SlimOrca",         # 逻辑推理与解释能力 
    "pubmed": "qiaojin/PubMedQA",         # 生物医药专业解释
}

# 目标总样本数 (根据显存和训练时间调整)
TOTAL_SAMPLES = 2000 

# 混合比例
RATIOS = {
    "reports": 0.5,    # 50% 核心研报 (AI + IC + Bio)
    "orca": 0.25,      # 25% SlimOrca (逻辑思维 + 通用对话)
    "pubmed": 0.25     # 25% PubMedQA (生物医学严谨性)
}

# ================= 核心逻辑 =================

def load_local_reports(filepath, target_count):
    """
    加载本地研报数据，支持 jsonl 或 json list 格式
    """
    print(f"正在加载本地研报数据: {filepath}...")
    data = []

    with open(filepath, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                # 这是一个 JSON List
                data = json.load(f)
            else:
                # 这是一个 JSONL 文件
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
    print(f"本地数据原始数量: {len(data)}")

    # 过采样 (Upsampling) 或 采样 (Sampling)
    if len(data) < target_count:
        print(f"    注意：本地数据量少于目标 ({target_count})，将循环填充。")
        full_data = data * (target_count // len(data)) + data[:target_count % len(data)]
        return full_data
    else:
        return random.sample(data, target_count)

def load_slim_orca(target_count):
    """
    加载 SlimOrca 数据集 (Stream模式，避免下载整个大数据集)
    SlimOrca 是 GPT-4 生成的思维链数据，非常适合做 CoT 微调
    """
    print(f"正在加载 SlimOrca (逻辑与推理)... 目标: {target_count}")
    data = []
    try:
        # 使用 streaming=True，随用随下，速度快
        dataset = load_dataset(DATASETS_CONFIG["orca"], split="train", streaming=True)
        
        # 进度条
        pbar = tqdm(total=target_count)
        
        for item in dataset:
            conversations = item.get('conversations', [])
            if not conversations:
                continue
            
            # 解析对话格式 (System -> Human -> GPT)
            # 我们提取 Human 的最后一个问题和 GPT 的回答
            input_text = ""
            output_text = ""
            system_prompt = ""
            
            for msg in conversations:
                if msg['from'] == 'system':
                    system_prompt = msg['value']
                elif msg['from'] == 'human':
                    input_text = msg['value']
                elif msg['from'] == 'gpt':
                    output_text = msg['value']
            
            # 组合 Instruction
            if system_prompt:
                instruction = system_prompt
            else:
                instruction = "You are a helpful assistant. Please answer the following question clearly."

            if input_text and output_text:
                data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                })
                pbar.update(1)
            
            if len(data) >= target_count:
                break
        
        pbar.close()
    except Exception as e:
        print(f"加载 SlimOrca 失败: {e}")
    
    return data

def load_pubmed_qa(target_count):
    """
    加载 PubMedQA (Labeled Subset)
    格式转化：Context + Question -> Long Answer
    """
    print(f"[-] 正在加载 PubMedQA (生物医药)... 目标: {target_count}")
    data = []
    try:
        # 使用 pqa_labeled 子集，这是人工标注的高质量数据
        dataset = load_dataset(DATASETS_CONFIG["pubmed"], "pqa_labeled", split="train", streaming=True)
        
        pbar = tqdm(total=target_count)
        
        for item in dataset:
            # item 结构: context(dict), question(str), long_answer(str), etc.
            
            # 提取背景 (通常是摘要)
            contexts = item.get('context', {}).get('contexts', [])
            context_str = "\n".join(contexts) if contexts else ""
            
            question = item.get('question', "")
            answer = item.get('long_answer', "")
            
            if not answer: 
                continue

            instruction = "作为生物医学专家，请基于提供的背景信息详细回答问题。"
            input_content = f"背景信息：\n{context_str}\n\n问题：{question}"
            
            data.append({
                "instruction": instruction,
                "input": input_content,
                "output": answer
            })
            pbar.update(1)
            
            if len(data) >= target_count:
                break
        
        pbar.close()

    except Exception as e:
        print(f"加载 PubMedQA 失败: {e}")
    
    return data

def main():
    # 1. 计算数量
    count_reports = int(TOTAL_SAMPLES * RATIOS["reports"])
    count_orca = int(TOTAL_SAMPLES * RATIOS["orca"])
    count_pubmed = int(TOTAL_SAMPLES * RATIOS["pubmed"])
    
    print(f"开始构建数据集 (总目标: {TOTAL_SAMPLES}) ===")
    print(f"计划配比: 研报 {count_reports} | SlimOrca {count_orca} | PubMedQA {count_pubmed}")

    # 2. 加载数据
    data_reports = load_local_reports(MY_REPORT_DATA_PATH, count_reports)
    data_orca = load_slim_orca(count_orca)
    data_pubmed = load_pubmed_qa(count_pubmed)

    # 3. 检查数据量
    print(f"\n实际加载数量:")
    print(f"  - 金融研报和行业资料: {len(data_reports)}")
    print(f"  - Orca: {len(data_orca)}")
    print(f"  - PubMed: {len(data_pubmed)}")

    # 4. 合并
    all_data = data_reports + data_orca + data_pubmed
    
    # 5. 打散 (非常重要，防止模型训练出现遗忘)
    print("正在打乱数据顺序 (Shuffling)...")
    random.shuffle(all_data)

    # 6. 保存
    print(f"正在写入 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"=== 完成！数据集已保存，可以直接用于微调 ===")

if __name__ == "__main__":
    main()