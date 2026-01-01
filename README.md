# Qwen-Fine-tuning-SFT-with-Unsloth-for-finance-QA-assistant
This project establishes a proprietary dataset for the financial investment sector by compiling relevant research reports and materials from the financial domain. It employs Unsloth to conduct LoRA + 4-bit Supervised Fine-Tuning (SFT) on Qwen2.5-7B-Instruct, subsequently exporting the LoRA Adapter weights and tokeniser while conducting validation. 

# QWEN_FINETUNING — Qwen2.5-7B-Instruct LoRA SFT (Unsloth + TRL)

Supervised fine-tuning (SFT) for **Qwen/Qwen2.5-7B-Instruct** using **Unsloth** + **TRL SFTTrainer** with **4-bit quantization** and **LoRA adapters**.

> ✅ Includes: dataset mixing scripts, training, evaluation (loss & perplexity), inference demo, and an interactive CLI chat.

---

## Table of Contents

- [Features](#features)
- [Project Layout](#project-layout)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Dataset](#dataset)
  - [Data Format (JSONL)](#data-format-jsonl)
  - [Build Mixed Dataset (Two Options)](#build-mixed-dataset-two-options)
- [Training](#training)
  - [Default Training Recipe](#default-training-recipe)
  - [Prompt Template](#prompt-template)
  - [Outputs](#outputs)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Chat CLI](#chat-cli)
- [Reproducibility](#reproducibility)
- [GitHub Publishing Guide](#github-publishing-guide)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Features

- **SFT (Supervised Fine-Tuning)** via `trl.SFTTrainer`
- **4-bit loading** (`load_in_4bit=True`) for memory-efficient training
- **LoRA adapters** targeting Qwen projection modules:
  - `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **Auto train/val split** (90/10) with a fixed seed
- **Dataset mixing utilities**
  - small-scale quick build (`data_integration.py`)
  - large-scale domain-filtered build (`data_set_generation.py`)
- **Evaluation**: `eval_loss` + `perplexity`
- **Inference demo** + **CLI chat** with streaming generation

---

## Project Layout

Recommended layout (matches your current scripts):

```bash
QWEN_FINETUNING/
├─ Qwen_finetuning.py               # SFT training entry (Unsloth + TRL)
├─ evaluate.py                      # eval loss & perplexity
├─ inference.py                     # single-turn inference demo
├─ chat.py                          # interactive CLI chat (streaming)
├─ data_integration.py              # mixed dataset builder (small)
├─ data_set_generation.py           # mixed dataset builder (large + domain filtering)
├─ data/
│  ├─ my_dataset.json               # your private/custom data (recommended: DO NOT commit)
│  └─ mixed_finetune_dataset.jsonl  # final training JSONL (recommended: DO NOT commit / or only sample)
├─ Qwen_finetuning_model/           # saved LoRA adapter output (optional to publish)
├─ Qwen_finetuning_tokenizer/       # saved tokenizer output (optional)
└─ README.md
Quickstart
Minimal “run it end-to-end” flow:

bash
复制代码
# 1) install deps
pip install -U pip
pip install unsloth transformers datasets trl accelerate peft bitsandbytes tqdm torch

# 2) build dataset (choose ONE)
python data_integration.py
# or:
python data_set_generation.py

# 3) train
python Qwen_finetuning.py

# 4) evaluate
python evaluate.py

# 5) try inference / chat
python inference.py
python chat.py
Installation
Prerequisites
Python 3.10+ recommended

NVIDIA GPU + CUDA recommended (4-bit quantization uses bitsandbytes)

Install dependencies
bash
复制代码
pip install -U pip
pip install unsloth transformers datasets trl accelerate peft bitsandbytes tqdm torch
Tip: If you want strict reproducibility, create a requirements.txt with pip freeze > requirements.txt after a successful run.

Dataset
Data Format (JSONL)
Training reads a local JSONL file:

default: mixed_finetune_dataset.jsonl

Each line must be a JSON object containing:

instruction (string)

input (string, can be empty)

output (string)

Example:

json
复制代码
{"instruction":"Explain what LoRA fine-tuning is.","input":"","output":"LoRA (Low-Rank Adaptation) is a parameter-efficient finetuning method..."}
{"instruction":"Summarize the following paper.","input":"Paper abstract: ...","output":"This paper proposes ..."}
Build Mixed Dataset (Two Options)
You have two dataset builders:

Option A — Small-scale quick build (fast sanity check)
Script: data_integration.py

bash
复制代码
python data_integration.py
Default behavior:

mixes:

your local my_dataset.json

Open-Orca/SlimOrca (streaming)

qiaojin/PubMedQA (pqa_labeled, streaming)

writes: mixed_finetune_dataset.jsonl

You can edit:

TOTAL_SAMPLES

RATIOS

paths: MY_REPORT_DATA_PATH, OUTPUT_FILE

Option B — Large-scale build + domain filtering
Script: data_set_generation.py

bash
复制代码
python data_set_generation.py
Default behavior:

mixes:

your local my_dataset.json

Open-Orca/SlimOrca (streaming)

qiaojin/PubMedQA (pqa_labeled, streaming)

HuggingFaceH4/stack-exchange-preferences (streaming)

filters StackExchange samples using keyword lists for:

AI domain

IC domain

writes: mixed_finetune_dataset.jsonl

You can edit:

TOTAL_SAMPLES, RATIOS

KEYWORDS for domain filtering

paths: MY_REPORT_DATA_PATH, OUTPUT_FILE

Training
Entry: Qwen_finetuning.py

bash
复制代码
python Qwen_finetuning.py
Default Training Recipe
Model:

Base: Qwen/Qwen2.5-7B-Instruct

max_seq_length = 2048

load_in_4bit = True

LoRA:

r = 16

lora_alpha = 32

lora_dropout = 0

bias = "none"

target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

TrainingArguments (defaults in code):

epochs: 3

batch size: 2

grad accumulation: 4

lr: 2e-4

warmup steps: 5

optimizer: adamw_8bit

eval: every 50 steps

save: every 50 steps

load_best_model_at_end = True

output dir: outputs/

Note: Your script sets HF_ENDPOINT=https://hf-mirror.com to speed up HF downloads in restricted networks.

Prompt Template
Your training builds a ChatML-style prompt with:

system: a fixed multi-domain expert system prompt (AI / IC / Biomedicine)

user: instruction + optional input as “补充信息”

assistant: the target output

Outputs
After training:

checkpoints/logs: outputs/

LoRA adapter: Qwen_finetuning_model/

tokenizer: Qwen_finetuning_tokenizer/

Qwen_finetuning_model/ typically contains:

adapter_model.safetensors

adapter_config.json

(optionally) tokenizer + chat template files if copied there

Evaluation
Entry: evaluate.py

bash
复制代码
python evaluate.py
What it does:

loads Qwen_finetuning_model

loads mixed_finetune_dataset.jsonl

re-splits 90/10 with fixed seed

reports:

eval_loss

perplexity = exp(eval_loss)

Important: evaluate.py uses max_seq_length=1024 by default.
If your samples are long, consider increasing it to match training (2048) for more consistent evaluation.
