# 1. 先导入 unsloth
from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

print("[-] 正在加载你微调的模型...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen_finetuning_model", 
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. 开启推理加速模式
FastLanguageModel.for_inference(model)

# 3. 准备问题
instruction = "请解释一下什么是CRISPR技术，以及它在基因编辑中的作用？"
system_prompt = "你是一位生物医药领域的专家，请用通俗易懂的语言回答。"

prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

# 4. 编码与生成
inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

_ = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 512,
    use_cache = True
)
