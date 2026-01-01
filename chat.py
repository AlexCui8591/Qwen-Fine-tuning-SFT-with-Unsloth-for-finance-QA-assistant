from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
import os

# 1. 加载模型
print("正在初始化聊天机器人...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen_finetuning_model",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) 

# 定义 System Prompt 
SYSTEM_PROMPT = """你是一位精通AI、生物医药和IC设计的跨领域专家。
回答请专业、准确，并尽量通俗易懂。"""

def chat_loop():
    # 清屏 (Linux/Mac用 clear, Windows用 cls)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("Expert Agent 终端 (输入 'exit' 或 'quit' 退出)")
    print(f"[System Info]: {SYSTEM_PROMPT}\n")

    while True:
        try:
            # 获取用户输入
            query = input("\n用户 (User): ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("再见！")
                break
            
            if not query:
                continue
            prompt = f"""<|im_start|>system
            
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
            

            inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
            
# 流式输出
            print("\n专家 (AI): ", end="")
            text_streamer = TextStreamer(tokenizer, skip_prompt=True)
            
            
            _ = model.generate(
                **inputs,
                streamer = text_streamer,
                max_new_tokens = 512,  
                
                # 1. 提升重复惩罚，狠狠惩罚那些已经出现过的词，迫使模型说新词
                repetition_penalty = 1.3, 
                
                # 2. 强制禁止 N-gram 重复 
                # 绝对不允许出现连续 3 个字和之前完全一样的情况
                no_repeat_ngram_size = 3, 

                # 3. 增加随机性
                temperature = 0.8,    
                top_p = 0.9,
                
                use_cache = True,
            )
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    chat_loop()