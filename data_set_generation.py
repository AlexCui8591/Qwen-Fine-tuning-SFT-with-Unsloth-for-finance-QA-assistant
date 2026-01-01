import json
import random
import os
from datasets import load_dataset
from tqdm import tqdm

# ================= 配置区域 =================

OUTPUT_FILE = "mixed_finetune_dataset.jsonl"
MY_REPORT_DATA_PATH = "my_dataset.json" 

# 你的研报只有60000条，作为25%的基准
TOTAL_SAMPLES = 240000

# 各部分在总数中的占比
RATIOS = {
    "my_reports": 0.25, # ~600条
    "orca": 0.15,       # ~360条
    "bio": 0.20,        # ~480条
    "ai": 0.20,         # ~480条
    "ic": 0.20          # ~480条
}

DATASETS_CONFIG = {
    "orca": "Open-Orca/SlimOrca",
    "pubmed": "qiaojin/PubMedQA",
    "stack_exchange": "HuggingFaceH4/stack-exchange-preferences" 
}

# 领域关键词过滤器
KEYWORDS = {
    "ai": [
        "neural network", "deep learning", "transformer", "backpropagation", 
        "reinforcement learning", "llm", "pytorch", "tensorflow", "gradient descent",
        "computer vision", "nlp", "lstm", "gan"
    ],
    "ic": [
        "transistor", "mosfet", "fpga", "verilog", "vhdl", "integrated circuit", 
        "pcb design", "microcontroller", "risc-v", "cmos", "logic gate", 
        "signal processing", "amplifier", "semiconductor", "asic"
    ],
    "biomed": [
    "protein", "gene", "dna", "rna", "antibody", "enzyme", 
    "crispr", "pcr", "sequencing", "ngs", "cell culture", 
    "assay", "biomarker", "peptide", "microfluidics", "plasmid"
    ]
}

# ================= 核心逻辑 =================

def load_local_reports(filepath, target_count):
    print(f"[-] [Local] 正在加载研报数据... 目标: {target_count}")
    data = []
    if not os.path.exists(filepath):
        print(f"    警告: {filepath} 未找到，将生成测试数据。")
        return [{"instruction": "测试", "output": "测试"}] * target_count

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first = f.read(1)
            f.seek(0)
            if first == '[':
                data = json.load(f)
            else:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
    except Exception as e:
        print(f"    ! 读取错误: {e}")
        return []

    if len(data) < target_count:
        full_data = data * (target_count // len(data)) + data[:target_count % len(data)]
        return full_data
    
    return random.sample(data, target_count)

def load_slim_orca(target_count):
    print(f"[-] [Logic] 正在加载 SlimOrca... 目标: {target_count}")
    data = []
    try:
        dataset = load_dataset(DATASETS_CONFIG["orca"], split="train", streaming=True)
        for item in tqdm(dataset, total=target_count, desc="Orca"):
            convs = item.get('conversations', [])
            input_text, output_text = "", ""
            for msg in convs:
                if msg['from'] == 'human': input_text = msg['value']
                if msg['from'] == 'gpt': output_text = msg['value']
            
            if input_text and output_text:
                data.append({
                    "instruction": "请详细回答以下问题。",
                    "input": input_text,
                    "output": output_text
                })
            if len(data) >= target_count: break
    except Exception as e:
        print(f"    ! 加载 Orca 失败: {e}")
    return data

def load_pubmed_qa(target_count):
    print(f"[-] [Bio] 正在加载 PubMedQA... 目标: {target_count}")
    data = []
    try:
        dataset = load_dataset(DATASETS_CONFIG["pubmed"], "pqa_labeled", split="train", streaming=True)
        for item in tqdm(dataset, total=target_count, desc="PubMed"):
            ctx = "\n".join(item.get('context', {}).get('contexts', []))
            ques = item.get('question', "")
            ans = item.get('long_answer', "")
            
            if ans:
                data.append({
                    "instruction": "基于医学背景回答。",
                    "input": f"背景：{ctx}\n问题：{ques}",
                    "output": ans
                })
            if len(data) >= target_count: break
    except Exception as e:
        print(f"    ! 加载 PubMedQA 失败: {e}")
    return data

def load_stack_exchange_domain(target_count, domain_key):

    print(f"[{domain_key.upper()}] 正在从 StackExchange 筛选数据... 目标: {target_count}")
    data = []
    keywords = KEYWORDS[domain_key]
    
    try:
        # 使用 streaming 模式
        dataset = load_dataset(DATASETS_CONFIG["stack_exchange"], split="train", streaming=True)
        
        pbar = tqdm(total=target_count, desc=f"Scanning {domain_key.upper()}")
        
        for item in dataset:
            # === [FIX 1] 使用正确的列名 'question' ===
            question = item.get('question', "") 
            
            # === [FIX 2] 处理 answers 列表，选出最高分回答 ===
            answers_list = item.get('answers', [])
            if not question or not answers_list:
                continue
                
            # 按 pm_score 降序排列，取第一个（最高分）
            try:
                best_answer_obj = sorted(answers_list, key=lambda x: x.get('pm_score', 0), reverse=True)[0]
                answer_text = best_answer_obj.get('text', "")
            except IndexError:
                continue

            if not answer_text: 
                continue

            # === 关键词过滤 ===
            content_to_check = (question + answer_text).lower()
            if any(k in content_to_check for k in keywords):
                data.append({
                    "instruction": f"请解释以下关于 {domain_key.upper()} 的技术问题。",
                    "input": question,
                    "output": answer_text
                })
                pbar.update(1)
            
            if len(data) >= target_count:
                break
        pbar.close()
        
    except Exception as e:
        print(f"    ! 加载 StackExchange ({domain_key}) 失败: {e}")
        
    return data

def main():
    # 1. 计算目标数量
    c_local = int(TOTAL_SAMPLES * RATIOS["my_reports"])
    c_orca = int(TOTAL_SAMPLES * RATIOS["orca"])
    c_bio = int(TOTAL_SAMPLES * RATIOS["bio"])
    c_ai = int(TOTAL_SAMPLES * RATIOS["ai"])
    c_ic = int(TOTAL_SAMPLES * RATIOS["ic"])

    print(f"数据集构建计划 (Total: {TOTAL_SAMPLES}) ===")
    print(f"1. Local Reports (25%): {c_local}")
    print(f"2. SlimOrca Logic (15%): {c_orca}")
    print(f"3. Bio (PubMedQA) (20%): {c_bio}")
    print(f"4. AI (StackEx)   (20%): {c_ai}")
    print(f"5. IC (StackEx)   (20%): {c_ic}")
    print("="*40)

    # 2. 执行加载
    data_local = load_local_reports(MY_REPORT_DATA_PATH, c_local)
    data_orca = load_slim_orca(c_orca)
    data_bio = load_pubmed_qa(c_bio)
    data_ai = load_stack_exchange_domain(c_ai, "ai")
    data_ic = load_stack_exchange_domain(c_ic, "ic")

    # 3. 合并与打散
    all_data = data_local + data_orca + data_bio + data_ai + data_ic
    print(f"\n正在混合并打乱 {len(all_data)} 条数据...")
    random.shuffle(all_data)

    # 4. 保存
    print(f"正在写入 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nSUCCESS: 数据集准备完毕！包含 {len(all_data)} 条样本。")

if __name__ == "__main__":
    main()