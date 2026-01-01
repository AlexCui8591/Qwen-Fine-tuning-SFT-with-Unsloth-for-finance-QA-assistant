````markdown
# QWEN_FINETUNING：Qwen2.5-7B-Instruct 的 LoRA SFT 微调（Unsloth + TRL）

> 使用 **Unsloth** + **TRL SFTTrainer** 对 `Qwen/Qwen2.5-7B-Instruct` 做 **4-bit 量化加载 + LoRA 适配器** 的 **SFT（监督微调）**。  
> 项目包含：**数据集混合构建（两套方案）/ 训练 / 评估（loss & perplexity）/ 推理示例 / 终端聊天（流式输出）**。

---

## 目录

- [项目亮点](#项目亮点)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [环境与安装](#环境与安装)
- [数据集](#数据集)
  - [数据格式（JSONL）](#数据格式jsonl)
  - [Option A：小规模混合构建（快速验证）](#option-a小规模混合构建快速验证)
  - [Option B：大规模混合构建（领域筛选）](#option-b大规模混合构建领域筛选)
- [训练（SFT + LoRA + 4bit）](#训练sft--lora--4bit)
  - [默认训练配方（来自代码）](#默认训练配方来自代码)
  - [Prompt 模板](#prompt-模板)
  - [训练输出](#训练输出)
- [评估（loss & perplexity）](#评估loss--perplexity)
- [推理（单条示例）](#推理单条示例)
- [终端聊天（CLI）](#终端聊天cli)
- [可复现性与推荐实践](#可复现性与推荐实践)
- [发布到 GitHub：所有选项（必读）](#发布到-github所有选项必读)
  - [Option 1：只开源代码（推荐默认）](#option-1只开源代码推荐默认)
  - [Option 2：发布 LoRA 权重（Git LFS）](#option-2发布-lora-权重git-lfs)
  - [Option 3：发布到 Hugging Face / GitHub Release](#option-3发布到-hugging-face--github-release)
- [附录：.gitignore / Git LFS / requirements 模板](#附录gitignore--git-lfs--requirements-模板)
- [常见问题（FAQ）](#常见问题faq)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 项目要点

- **SFT 监督微调**：使用 `trl.SFTTrainer`
- **4-bit 量化加载**：显存友好（`load_in_4bit=True`）
- **LoRA 覆盖 Qwen 常见投影层**：`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`
- **自动划分训练/验证集**：90% / 10%（固定 seed）
- **两套数据集混合构建脚本**
  - Option A：小规模混合（更快）
  - Option B：大规模混合 + 领域关键词筛选（更强，但更慢/更大）
- **评估脚本**：输出 `eval_loss` 与 `perplexity`
- **推理示例与 CLI 聊天**：流式输出、重复惩罚等常用解码参数

---

## 项目结构

> 下面结构与你当前脚本命名一致：

```bash
QWEN_FINETUNING/
├─ Qwen_finetuning.py               # 训练入口（Unsloth + TRL SFT）
├─ evaluate.py                      # 评估：loss & perplexity
├─ inference.py                     # 推理：单条示例
├─ chat.py                          # 终端聊天：交互式 + 流式输出
├─ data_integration.py              # 数据混合构建（Option A：小规模）
├─ data_set_generation.py           # 数据混合构建（Option B：大规模 + 领域筛选）
├─ my_dataset.json                  # 你的自定义/私有数据
├─ mixed_finetune_dataset.jsonl     # 混合后的训练数据
├─ outputs/                         
├─ Qwen_finetuning_model/           # 训练产物：LoRA adapter
└─ Qwen_finetuning_tokenizer/       # tokenizer 
````

---

## 快速开始

```bash
# 1) 安装依赖
pip install -U pip
pip install unsloth transformers datasets trl accelerate peft bitsandbytes tqdm torch

# 2) 构建数据（任选一个 Option）
python data_integration.py
# 或：
python data_set_generation.py

# 3) 训练
python Qwen_finetuning.py

# 4) 评估
python evaluate.py

# 5) 体验推理 / 聊天
python inference.py
python chat.py
```

---

## 环境与安装

### 运行前提

* Python 3.10+（推荐）
* NVIDIA GPU + CUDA（推荐；脚本会把输入 `.to("cuda")`）
* `bitsandbytes`（4-bit 量化依赖）

### 安装依赖

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -U pip
pip install unsloth transformers datasets trl accelerate peft bitsandbytes tqdm torch
```
> `pip freeze > requirements.txt`

---

## 数据集

你的训练脚本默认读取：
* `mixed_finetune_dataset.jsonl`（位于项目根目录）

### 数据格式（JSONL）

每行一条 JSON，包含以下字段：

* `instruction`：指令/问题（字符串）
* `input`：补充信息（字符串，可为空）
* `output`：期望回答（字符串）

示例：

```json
{"instruction":"解释什么是LoRA微调","input":"","output":"LoRA（Low-Rank Adaptation）是一种参数高效微调方法..."}
{"instruction":"总结以下内容","input":"这里是补充信息...","output":"总结如下：..."}
```

---

### Option A：小规模混合构建（快速验证）

脚本：`data_integration.py`

```bash
python data_integration.py
```

默认行为（来自代码）：

* 读取本地：`my_dataset.json`
* streaming 拉取并采样：

  * `Open-Orca/SlimOrca`
  * `qiaojin/PubMedQA`（pqa_labeled）
* 按比例混合并打乱写入：

  * 输出：`mixed_finetune_dataset.jsonl`

你可以在脚本中调整：

* `TOTAL_SAMPLES`（默认 2000）
* `RATIOS`（reports/orca/pubmed）
* `MY_REPORT_DATA_PATH` / `OUTPUT_FILE`

> 适用场景：先跑通流程、快速验证训练是否收敛、检查 prompt/格式是否正确。

---

### Option B：大规模混合构建（领域筛选）

脚本：`data_set_generation.py`

```bash
python data_set_generation.py
```

默认行为（来自代码）：

* 读取本地：`my_dataset.json`
* streaming 拉取并采样：

  * `Open-Orca/SlimOrca`
  * `qiaojin/PubMedQA`（pqa_labeled）
  * `HuggingFaceH4/stack-exchange-preferences`
* 对 StackExchange 数据按关键词过滤，拆成：
  * AI 域
  * IC 域
  * biomed域
* 混合并打乱写入：

  * 输出：`mixed_finetune_dataset.jsonl`

你可以在脚本中调整（常用项）：

* `TOTAL_SAMPLES`（默认 240000，较大！）
* `RATIOS`（my_reports/orca/bio/ai/ic）
* `KEYWORDS`（领域关键词列表）
* `MY_REPORT_DATA_PATH` / `OUTPUT_FILE`

> 适用场景：希望引入更大规模、更多样的跨领域数据，提高泛化能力（但构建耗时更长、数据体积更大）。

---

## 训练（SFT + LoRA + 4bit）

训练入口：`Qwen_finetuning.py`

```bash
python Qwen_finetuning.py
```

### 默认训练配方（来自代码）

* Base 模型：`Qwen/Qwen2.5-7B-Instruct`
* `max_seq_length = 2048`
* `load_in_4bit = True`
* LoRA：

  * `r = 16`
  * `lora_alpha = 32`
  * `lora_dropout = 0`
  * target modules：

    * `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
* 数据切分：90% train / 10% eval（`seed=42`）
* TrainingArguments：

  * `num_train_epochs=3`
  * `per_device_train_batch_size=2`
  * `gradient_accumulation_steps=4`
  * `learning_rate=2e-4`
  * `warmup_steps=5`
  * `optim="adamw_8bit"`
  * `eval_strategy="steps"`, `eval_steps=50`
  * `save_strategy="steps"`, `save_steps=50`
  * `load_best_model_at_end=True`
  * `output_dir="outputs"`
  * `neftune_noise_alpha=5`

### Prompt 模板

训练时会把样本拼成 ChatML（你代码里手动拼接）：

* system：固定的跨领域专家系统提示词
* user：`instruction` + 可选 `input`（作为“补充信息”）
* assistant：目标 `output`

---

### 训练输出

训练完成后会生成：

* `outputs/`：训练日志/中间 checkpoint（**建议不要提交到 GitHub**）
* `Qwen_finetuning_model/`：LoRA adapter（`save_pretrained`）
* `Qwen_finetuning_tokenizer/`：tokenizer（`save_pretrained`）

> 注意：你的 `inference.py/chat.py` 目前从 `Qwen_finetuning_model/` 加载 tokenizer。
> 若 tokenizer 实际保存在 `Qwen_finetuning_tokenizer/`，请保持一致（见 FAQ）。

---

## 评估（loss & perplexity）

运行：

```bash
python evaluate.py
```

该脚本会：

* 加载 `Qwen_finetuning_model`
* 读取 `mixed_finetune_dataset.jsonl`
* 重新按 90/10 切分验证集
* 输出：

  * `Eval Loss`
  * `Perplexity = exp(eval_loss)`

> 重要：`evaluate.py` 默认 `max_seq_length=1024`，与训练的 2048 不一致。
> 如果你的样本很长，建议把评估脚本也调到 2048 以减少截断影响。

---

## 推理（单条示例）

运行：

```bash
python inference.py
```

默认会生成一条示例回答（流式输出）。
你可以修改脚本中的：

* `instruction`
* `system_prompt`

---

## 终端聊天（CLI）

运行：

```bash
python chat.py
```

特性：

* 输入 `exit`/`quit` 退出
* 流式输出（`TextStreamer`）
* 解码参数（默认）：

  * `max_new_tokens=512`
  * `repetition_penalty=1.3`
  * `no_repeat_ngram_size=3`
  * `temperature=0.8`
  * `top_p=0.9`

---

## 可复现性与推荐实践

* 数据切分 seed：`seed=42`
* 训练 seed：`seed=3407`

---


## 常见问题（FAQ）

### Q1：为什么训练保存 tokenizer 到 `Qwen_finetuning_tokenizer/`，但推理/聊天从 `Qwen_finetuning_model/` 加载？

你的 `chat.py/inference.py` 是从 `Qwen_finetuning_model` 加载 tokenizer。
解决方案（任选一个）：

* **方案 A（推荐）**：修改 `chat.py/inference.py`，让 tokenizer 从 `Qwen_finetuning_tokenizer` 加载
* **方案 B**：把 tokenizer 文件复制到 `Qwen_finetuning_model/`，确保两边一致

### Q2：评估的 `max_seq_length=1024` 会影响结果吗？

会。长文本会被截断，导致 loss/perplexity 不完全可比。
建议改成与训练一致（2048）或明确记录差异。

### Q4：显存不够怎么办？

* 降低 `per_device_train_batch_size`
* 提高 `gradient_accumulation_steps`
* 降低 `max_seq_length`
* 减少数据规模（先用 Option A）

---

## 致谢

* Unsloth：高效训练与推理加速
* Hugging Face TRL：SFTTrainer
* PEFT / bitsandbytes：LoRA 与 4-bit 生态

---

