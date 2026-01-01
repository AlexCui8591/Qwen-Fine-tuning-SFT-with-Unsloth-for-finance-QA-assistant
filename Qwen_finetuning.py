from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# 1. 云端加速配置 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 模型配置 
max_seq_length = 2048
dtype = None 
load_in_4bit = True 
model_name = "Qwen/Qwen2.5-7B-Instruct" 

print(f"正在从镜像下载模型 {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. LoRA 配置 
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 数据处理 
# 加载原始数据
dataset = load_dataset("json", data_files="mixed_finetune_dataset.jsonl", split="train")

# 自动切分：90% 用于训练，10% 用于评估
# seed=42 保证每次切分的结果是一样的
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split['train']
eval_dataset  = dataset_split['test']

print(f"数据集划分完成：训练集 {len(train_dataset)} 条，验证集 {len(eval_dataset)} 条")

system_prompt = """你是一位精通多领域的跨学科专家，并且你专精于金融投资领域的知识，特别专长于以下三大支柱领域：
1. 人工智能 (AI) 与大语言模型
2. 集成电路设计 (IC Design) 与数字逻辑
3. 生物医药 (Biomedicine) 与前沿生命科学

你的回答必须遵循以下原则：
- 专业严谨：使用准确的学术和行业术语。
- 逻辑清晰：对于复杂问题，请进行分步推导（Chain of Thought）。
- 实用导向：在提供专业分析或方案时，优先考虑可行性和最佳实践。
- 拒绝幻觉：如果超出你的知识范围，请诚实告知，不要编造事实。"""

def format_prompts_func(examples):
    instructions = examples.get("instruction", None)
    inputs       = examples.get("input", None)
    outputs      = examples.get("output", None)

    if not isinstance(instructions, list):
        instructions = [instructions]
        inputs       = [inputs]
        outputs      = [outputs]

    output_texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        instruction = instruction if instruction is not None else ""
        input_text  = input_text if input_text is not None else ""
        output      = output if output is not None else ""
        
        if input_text:
            user_content = f"{instruction}\n\n补充信息：\n{input_text}"
        else:
            user_content = instruction

        text = (
            f"<|im_start|>system\n{system_prompt}。<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
        )
        text += tokenizer.eos_token 
        output_texts.append(text)
        
    return output_texts

# 5. 训练参数 
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    
    # 分别传入训练集和验证集
    train_dataset = train_dataset,
    eval_dataset = eval_dataset, 
    
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    formatting_func = format_prompts_func,
    args = TrainingArguments(
        num_train_epochs = 3,
        per_device_train_batch_size = 2, 
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        neftune_noise_alpha = 5,
        
        # 评估相关参数
        eval_strategy = "steps", # 按步数评估 
        eval_steps = 50,               
        per_device_eval_batch_size = 2, 
        save_strategy = "steps",       # 保存策略通常和评估策略对齐
        save_steps = 50,               
        load_best_model_at_end = True, 
    ),
)

print("开始训练...")
trainer_stats = trainer.train()

# 保存
print("正在保存最佳模型...")
model.save_pretrained("Qwen_finetuning_model")
tokenizer.save_pretrained("Qwen_finetuning_tokenizer")
print("训练完成！")