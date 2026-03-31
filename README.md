# Fine-Tuning-TinyLlama-1.1B-with-LoRA

This repository fine-tunes `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on the Alpaca instruction dataset using LoRA with TRL, PEFT, and optional 4-bit quantization. It now includes a reproducible CLI training script in addition to the original notebook.

## What Changed

- `train.py` provides a scriptable training and evaluation workflow.
- `requirements.txt` now contains only project-relevant dependencies.
- The original notebook, `FineTuning_TinyLlama_LoRA.ipynb`, remains in the repo for interactive exploration.

## Features

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Fine-tuning method: LoRA via PEFT
- Trainer: `trl.SFTTrainer`
- Dataset: `tatsu-lab/alpaca`
- Optional 4-bit loading through `bitsandbytes`
- Built-in deterministic adapter-vs-base evaluation export

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Quick Start

Train on a smaller sample that fits common single-GPU setups:

```bash
python3 train.py \
  --dataset-sample-size 8000 \
  --validation-size 256 \
  --output-dir outputs/tinyllama-alpaca-lora \
  --adapter-dir outputs/tinyllama-alpaca-lora-adapter
```

If you are running on a T4 or another GPU without strong `bf16` support, force `fp16`:

```bash
python3 train.py --precision fp16
```

If you want to skip the prompt comparison export:

```bash
python3 train.py --skip-eval
```

## Outputs

After a run, the script writes:

- Adapter weights to `outputs/tinyllama-alpaca-lora-adapter`
- Trainer checkpoints and metadata to `outputs/tinyllama-alpaca-lora`
- Prompt comparison CSV to `outputs/evaluation_base_vs_ft.csv`
- JSON summaries for training and evaluation alongside those outputs

## Notes

- The default sample size is `8000` rows to keep the example practical for smaller hardware.
- The script creates a small validation split when possible so runs are easier to compare.
- Precision is set automatically, but you can override it with `--precision fp16` or `--precision bf16`.
- The evaluation is lightweight and prompt-based; it is useful for regression checks, not as a full benchmark.

## Notebook

The original notebook is still available for Colab-style experimentation:

- `FineTuning_TinyLlama_LoRA.ipynb`

## License

This project is licensed under the MIT License.
