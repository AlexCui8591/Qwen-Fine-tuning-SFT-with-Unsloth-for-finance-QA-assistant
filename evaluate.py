import os
import json
import math
import argparse
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from datasets import load_dataset

from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM


DEFAULT_SYSTEM_PROMPT = """你是一位精通多领域的跨学科专家，并且你专精于金融投资领域的知识，特别专长于以下三大支柱领域：
1. 人工智能 (AI) 与大语言模型
2. 集成电路设计 (IC Design) 与数字逻辑
3. 生物医药 (Biomedicine) 与前沿生命科学

你的回答必须遵循以下原则：
- 专业严谨：使用准确的学术和行业术语。
- 逻辑清晰：对于复杂问题，请进行分步推导（Chain of Thought）。
- 实用导向：在提供专业分析或方案时，优先考虑可行性和最佳实践。
- 拒绝幻觉：如果超出你的知识范围，请诚实告知，不要编造事实。"""


def build_user_content(instruction: str, input_text: str) -> str:
    instruction = instruction or ""
    input_text = input_text or ""
    if input_text.strip():
        return f"{instruction}\n\n补充信息：\n{input_text}"
    return instruction


def build_prompt(system_prompt: str, user_content: str) -> str:
    # 只包含 system + user + assistant 起始，用于生成式评估
    return (
        f"<|im_start|>system\n{system_prompt}。<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_full_text_with_ref(system_prompt: str, user_content: str, ref_output: str, tokenizer) -> str:
    ref_output = ref_output or ""
    text = (
        f"<|im_start|>system\n{system_prompt}。<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{ref_output}<|im_end|>"
    )
    text += tokenizer.eos_token
    return text


def strip_special_tokens(text: str) -> str:
    # 生成结果里可能带 <|im_end|> 或 eos，把后面的截掉更利于 ROUGE
    if text is None:
        return ""
    for stopper in ["<|im_end|>", "<|endoftext|>"]:
        if stopper in text:
            text = text.split(stopper, 1)[0]
    return text.strip()


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    total_ngrams = 0
    uniq_ngrams = set()
    for t in texts:
        toks = t.strip().split()
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            ng = tuple(toks[i:i+n])
            uniq_ngrams.add(ng)
            total_ngrams += 1
    return (len(uniq_ngrams) / total_ngrams) if total_ngrams > 0 else 0.0


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)



# 主流程
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen_finetuning_model",
                        help="你的 LoRA/微调模型保存目录（model.save_pretrained 输出的目录）")
    parser.add_argument("--data_file", type=str, default="mixed_finetune_dataset.jsonl",
                        help="你的 JSONL 数据文件（包含 instruction/input/output）")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="评估输出目录")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="建议与训练保持一致（你的训练是 2048）")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="生成式评估时的最大输出长度")
    parser.add_argument("--eval_ratio", type=float, default=0.1,
                        help="验证集比例（默认 10%）")
    parser.add_argument("--seed", type=int, default=42,
                        help="数据切分随机种子（与你训练一致）")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="只评估前 N 条（0 表示全量），用于快速调试")
    parser.add_argument("--rouge", action="store_true",
                        help="启用 ROUGE 评估（需要 pip install evaluate rouge-score）")
    parser.add_argument("--bf16", action="store_true",
                        help="强制 bf16（仅在支持时生效），默认自动判断")
    args = parser.parse_args()

    safe_makedirs(args.output_dir)

    
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 2.1 加载模型
    print(f"[1/6] 正在加载模型: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # 2.2 加载数据并切分 eval
    print(f"[2/6] 正在加载数据: {args.data_file}")
    dataset = load_dataset("json", data_files=args.data_file, split="train")
    dataset_split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    eval_dataset = dataset_split["test"]

    if args.num_samples and args.num_samples > 0:
        eval_dataset = eval_dataset.select(range(min(args.num_samples, len(eval_dataset))))

    print(f"验证集条数: {len(eval_dataset)}")

    # 2.3 统一 system prompt
    system_prompt = DEFAULT_SYSTEM_PROMPT

    # 2.4 统计截断率
    print("[3/6] 统计样本长度与截断情况...")
    lengths = []
    trunc_count = 0
    for ex in tqdm(eval_dataset, desc="LenStats"):
        user_content = build_user_content(ex.get("instruction", ""), ex.get("input", ""))
        full_text = build_full_text_with_ref(system_prompt, user_content, ex.get("output", ""), tokenizer)
        l = token_len(tokenizer, full_text)
        lengths.append(l)
        if l > args.max_seq_length:
            trunc_count += 1

    trunc_rate = trunc_count / max(1, len(eval_dataset))
    avg_len = sum(lengths) / max(1, len(lengths))
    max_len = max(lengths) if lengths else 0

 
    # A) Teacher-forcing loss/ppl（Completion-only）
    print("[4/6] 计算 Teacher-Forcing 指标（eval_loss / ppl，assistant-only masking）...")

    # formatting_func：返回“完整文本（含参考输出）”
    def formatting_func(examples):
        instructions = examples.get("instruction", None)
        inputs = examples.get("input", None)
        outputs = examples.get("output", None)

        if not isinstance(instructions, list):
            instructions = [instructions]
            inputs = [inputs]
            outputs = [outputs]

        texts = []
        for ins, inp, out in zip(instructions, inputs, outputs):
            user_content = build_user_content(ins or "", inp or "")
            texts.append(build_full_text_with_ref(system_prompt, user_content, out or "", tokenizer))
        return texts

    # 只在 assistant completion 上计算 loss
    response_template = "<|im_start|>assistant\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=1,
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )
    if args.bf16:
        training_args.fp16 = False
        training_args.bf16 = torch.cuda.is_bf16_supported()

    # SFTTrainer 通常需要 train_dataset，这里给一个最小 dummy，避免误解也避免潜在报错
    dummy_train = eval_dataset.select(range(min(1, len(eval_dataset))))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dummy_train,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_func,
        data_collator=data_collator,
        args=training_args,
    )

    tf_metrics = trainer.evaluate()
    eval_loss = float(tf_metrics.get("eval_loss", float("nan")))
    ppl = float(math.exp(eval_loss)) if math.isfinite(eval_loss) else float("nan")


 # B) 生成式评估：生成 predictions + ROUGE + 质量统计
    print("[5/6] 生成式评估：逐条生成预测并计算指标...")
    FastLanguageModel.for_inference(model)
    model.eval()

    predictions_path = os.path.join(args.output_dir, "predictions.jsonl")

    preds: List[str] = []
    refs: List[str] = []
    pred_lens: List[int] = []
    ref_lens: List[int] = []
    empty_count = 0

    with open(predictions_path, "w", encoding="utf-8") as f:
        for ex in tqdm(eval_dataset, desc="Generate"):
            instruction = ex.get("instruction", "") or ""
            input_text = ex.get("input", "") or ""
            ref_output = ex.get("output", "") or ""

            user_content = build_user_content(instruction, input_text)
            prompt = build_prompt(system_prompt, user_content)

            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            prompt_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,     
                    use_cache=True,
                )

            # 只取新生成部分
            new_tokens = gen_ids[0][prompt_len:]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
            pred_text = strip_special_tokens(pred_text)

            ref_text = (ref_output or "").strip()

            if not pred_text:
                empty_count += 1

            preds.append(pred_text)
            refs.append(ref_text)

            pred_lens.append(token_len(tokenizer, pred_text))
            ref_lens.append(token_len(tokenizer, ref_text))

            record = {
                "instruction": instruction,
                "input": input_text,
                "reference": ref_text,
                "prediction": pred_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    empty_rate = empty_count / max(1, len(eval_dataset))
    avg_pred_len = sum(pred_lens) / max(1, len(pred_lens))
    avg_ref_len = sum(ref_lens) / max(1, len(ref_lens))
    len_ratio = (avg_pred_len / avg_ref_len) if avg_ref_len > 0 else 0.0

    distinct_1 = compute_distinct_n(preds, n=1)
    distinct_2 = compute_distinct_n(preds, n=2)

    # ROUGE
    rouge_scores = {}
    if args.rouge:
        try:
            import evaluate as hf_evaluate
            rouge = hf_evaluate.load("rouge")
            rouge_scores = rouge.compute(
                predictions=preds,
                references=refs,
                use_stemmer=True,
            )
        except Exception as e:
            print(f"[WARN] ROUGE 计算失败：{e}")
            rouge_scores = {}

    # 汇总并保存
    print("[6/6] 保存指标到 metrics.json ...")
    metrics_out = {
        "teacher_forcing": {
            "eval_loss": eval_loss,
            "perplexity": ppl,
            "max_seq_length": args.max_seq_length,
            "assistant_only_loss": True,
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "avg_pred_len_tokens": avg_pred_len,
            "avg_ref_len_tokens": avg_ref_len,
            "len_ratio_pred_over_ref": len_ratio,
            "empty_rate": empty_rate,
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "rouge": rouge_scores,
        },
        "data_stats": {
            "eval_size": len(eval_dataset),
            "avg_full_text_len_tokens": avg_len,
            "max_full_text_len_tokens": max_len,
            "truncation_rate_full_text_over_max_seq": trunc_rate,
            "eval_ratio": args.eval_ratio,
            "seed": args.seed,
        },
        "artifacts": {
            "predictions_jsonl": predictions_path,
        }
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    # 打印报告
    print("\n" + "=" * 40)
    print("评估结果（Summary）")
    print("=" * 40)
    print(f"[Teacher-Forcing] eval_loss = {eval_loss:.4f}")
    print(f"[Teacher-Forcing] perplexity = {ppl:.4f}")
    print(f"[Data] truncation_rate(full_text) = {trunc_rate*100:.2f}%  (max_seq_length={args.max_seq_length})")
    print(f"[Gen] empty_rate = {empty_rate*100:.2f}%")
    print(f"[Gen] avg_pred_len_tokens = {avg_pred_len:.1f} | avg_ref_len_tokens = {avg_ref_len:.1f} | ratio = {len_ratio:.2f}")
    print(f"[Gen] distinct-1 = {distinct_1:.4f} | distinct-2 = {distinct_2:.4f}")
    if rouge_scores:
        print(f"[Gen] ROUGE-1 = {rouge_scores.get('rouge1', 0):.4f} | "
              f"ROUGE-2 = {rouge_scores.get('rouge2', 0):.4f} | "
              f"ROUGE-L = {rouge_scores.get('rougeL', 0):.4f}")
    print(f"[Files] metrics: {metrics_path}")
    print(f"[Files] predictions: {predictions_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
