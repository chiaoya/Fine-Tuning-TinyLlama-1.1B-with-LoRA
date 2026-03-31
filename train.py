from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune TinyLlama-1.1B-Chat-v1.0 on Alpaca with LoRA."
    )
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset-id", default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-sample-size", type=int, default=8000)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--output-dir", default="outputs/tinyllama-alpaca-lora")
    parser.add_argument("--adapter-dir", default="outputs/tinyllama-alpaca-lora-adapter")
    parser.add_argument("--eval-output", default="outputs/evaluation_base_vs_ft.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--quantization", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--eval-prompts-file")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_alpaca(example):
    instruction = str(example.get("instruction", "")).strip()
    inp = str(example.get("input", "")).strip()
    output = str(example.get("output", "")).strip()

    if inp:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    return {"text": prompt + output}


def load_and_prepare_datasets(args) -> tuple[Dataset, Dataset]:
    ds = load_dataset(args.dataset_id, split=args.dataset_split)
    ds = ds.map(format_alpaca, remove_columns=ds.column_names)
    ds = ds.shuffle(seed=args.seed)

    if args.dataset_sample_size and args.dataset_sample_size > 0:
        ds = ds.select(range(min(args.dataset_sample_size, len(ds))))

    if args.validation_size and args.validation_size > 0 and len(ds) > args.validation_size:
        split = ds.train_test_split(test_size=args.validation_size, seed=args.seed)
        return split["train"], split["test"]

    return ds, Dataset.from_list([])


def resolve_precision(precision: str) -> tuple[bool, bool]:
    if precision == "fp16":
        return True, False
    if precision == "bf16":
        return False, True
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return (not bf16_supported), bf16_supported


def build_quantization_config(args):
    if args.quantization == "none":
        return None

    compute_dtype = torch.bfloat16 if resolve_precision(args.precision)[1] else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(args, for_training: bool = True):
    quantization_config = build_quantization_config(args)
    fp16, bf16 = resolve_precision(args.precision)
    torch_dtype = torch.bfloat16 if bf16 else torch.float16

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": False,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    elif not for_training:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False if for_training else True
    return model, fp16, bf16


def build_lora_config(args):
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def train(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    adapter_dir = Path(args.adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_ds, eval_ds = load_and_prepare_datasets(args)
    tokenizer = load_tokenizer(args.model_id)
    model, fp16, bf16 = load_model(args, for_training=True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds if len(eval_ds) else None,
        peft_config=build_lora_config(args),
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            fp16=fp16,
            bf16=bf16,
            optim="paged_adamw_8bit",
            report_to="none",
            max_seq_length=args.max_seq_length,
            dataset_text_field="text",
            save_strategy="steps",
            evaluation_strategy="steps" if len(eval_ds) else "no",
            eval_steps=args.save_steps if len(eval_ds) else None,
            logging_strategy="steps",
            seed=args.seed,
            remove_unused_columns=False,
            gradient_checkpointing=True,
        ),
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics_path = output_dir / "training_summary.json"
    summary = {
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "dataset_sample_size": len(train_ds) + len(eval_ds),
        "train_rows": len(train_ds),
        "validation_rows": len(eval_ds),
        "precision": "bf16" if bf16 else "fp16",
        "quantization": args.quantization,
        "adapter_dir": str(adapter_dir),
    }
    metrics_path.write_text(json.dumps(summary, indent=2))
    return tokenizer


def default_eval_prompts():
    return [
        "### Instruction:\nGive me 3 practical tips to reduce stockouts in a retail supply chain.\n\n### Response:\n",
        "### Instruction:\nExplain the difference between safety stock and reorder point in plain English.\n\n### Response:\n",
        "### Instruction:\nReturn a JSON object with keys \"risk\", \"mitigation\", and \"owner\" for a delayed supplier shipment.\n\n### Response:\n",
    ]


def load_eval_prompts(args):
    if not args.eval_prompts_file:
        return default_eval_prompts()

    path = Path(args.eval_prompts_file)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("JSON eval prompt file must contain a list of prompt strings.")
        return [str(item) for item in data]

    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def generate_det(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def extract_response(full_text):
    parts = re.split(r"### Response:\s*", full_text, maxsplit=1)
    return parts[-1].strip() if len(parts) > 1 else full_text.strip()


def json_parseable(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return False
    try:
        json.loads(match.group(0))
        return True
    except json.JSONDecodeError:
        return False


def evaluate_adapter(args, tokenizer):
    prompts = load_eval_prompts(args)
    base_model, _, _ = load_model(args, for_training=False)
    ft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    ft_model.eval()

    rows = []
    for prompt in prompts:
        with ft_model.disable_adapter():
            base_full = generate_det(ft_model, tokenizer, prompt, 220)
        ft_full = generate_det(ft_model, tokenizer, prompt, 220)

        base_out = extract_response(base_full)
        ft_out = extract_response(ft_full)
        needs_json = "JSON" in prompt.upper()

        rows.append(
            {
                "prompt": prompt,
                "base_output": base_out,
                "ft_output": ft_out,
                "base_length": len(base_out),
                "ft_length": len(ft_out),
                "base_json_ok": json_parseable(base_out) if needs_json else None,
                "ft_json_ok": json_parseable(ft_out) if needs_json else None,
            }
        )

    df = pd.DataFrame(rows)
    eval_output = Path(args.eval_output)
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(eval_output, index=False)

    summary = {
        "total_prompts": len(df),
        "average_base_length": float(df["base_length"].mean()),
        "average_ft_length": float(df["ft_length"].mean()),
    }
    json_rows = df[df["base_json_ok"].notna()]
    if len(json_rows):
        summary["base_json_pass_rate"] = float(json_rows["base_json_ok"].mean())
        summary["ft_json_pass_rate"] = float(json_rows["ft_json_ok"].mean())

    summary_path = eval_output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    args = parse_args()
    tokenizer = train(args)
    if not args.skip_eval:
        summary = evaluate_adapter(args, tokenizer)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
