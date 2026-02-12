# Fine-Tuning-TinyLlama-1.1B-with-LoRA
This repository contains a Jupyter Notebook for fine-tuning the TinyLlama-1.1B model using Parameter-Efficient Fine-Tuning (PEFT) techniques, specifically LoRA. The model is trained on the Alpaca dataset to improve its instruction-following capabilities.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/YOUR_NOTEBOOK.ipynb)

## 🚀 Features
- **Model:** TinyLlama-1.1B-Chat-v1.0.
- **Technique:** LoRA (Low-Rank Adaptation) for efficient fine-tuning.
- **Quantization:** 4-bit quantization via bitsandbytes to enable training on consumer-grade GPUs (like the Google Colab T4).
- **Dataset:** Alpaca (52k instruction-following records).
- **Frameworks:** transformers, peft, trl, and bitsandbytes.

## 🛠️ Installation
To run this notebook, you will need to install the following dependencies:

```python
Bash
pip install -q -U transformers peft trl bitsandbytes accelerate datasets
```


## 📈 Training Configuration
The notebook is optimized for a single NVIDIA T4 GPU:

- **Batch Size:** 2.
- **Gradient Accumulation Steps:** 8 (Effective batch size of 16).
- **Optimizer:** Paged AdamW 32-bit.
- **Learning Rate:** 2e-4.
- **Precision:** 4-bit NormalFloat.

## 💡 Usage

1. Training
The training process uses the SFTTrainer from the TRL library. It processes the Alpaca dataset and saves LoRA adapters that can be merged back into the base model.

2. Inference
You can load the trained adapters and test the model using the following snippet included in the notebook:

```python
Python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model and adapter
base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ft_model = PeftModel.from_pretrained(base_model, "path_to_your_adapter")

# Generate response
prompt = "### Instruction:\nGive me 3 practical tips to reduce stockouts.\n\n### Response:\n"
# ... generation code ...
```

## 📊 Results
The notebook includes a comparison between the Base Model and the Fine-Tuned Model. The fine-tuned version demonstrates significantly better adherence to the "Alpaca" instruction format and provides more structured, concise answers.

## 📄 License
This project is licensed under the MIT License.
