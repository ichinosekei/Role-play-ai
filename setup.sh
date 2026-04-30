#!/bin/bash
# ══════════════════════════════════════════════════════
#  Setup — Qwen2.5-32B-Instruct SFW RP fine-tune
# ══════════════════════════════════════════════════════

set -e

echo "╔════════════════════════════════════════╗"
echo "║  Qwen2.5-32B SFW RP Setup              ║"
echo "╚════════════════════════════════════════╝"

# 1. Проверка
echo ""
echo "[1/4] Проверка системы..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -h | head -2
df -h ~ | tail -1

# 2. venv
echo ""
echo "[2/4] Создаём venv..."
python3 -m venv ~/rp-venv
source ~/rp-venv/bin/activate
pip install --upgrade pip wheel

# 3. PyTorch + Unsloth
echo ""
echo "[3/4] PyTorch + Unsloth..."
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -q "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install -q --no-deps trl peft accelerate bitsandbytes
pip install -q "datasets>=3.4.1" sentencepiece protobuf
pip install -q huggingface_hub hf_transfer
pip install -q "transformers>=4.46.0"

# 4. Метрики (новые, для evaluate.py)
echo ""
echo "[4/4] Метрики и анализ..."
pip install -q tensorboard psutil numpy
pip install -q sacrebleu                # BLEU + chrF++
pip install -q rouge_score              # ROUGE-L
pip install -q bert_score               # BERTScore
pip install -q nltk

export HF_HUB_ENABLE_HF_TRANSFER=1
echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc

echo ""
echo "Логин в HuggingFace:"
huggingface-cli login || true

echo ""
echo "╔════════════════════════════════════════╗"
echo "║  ✓ Готово                              ║"
echo "║                                        ║"
echo "║  Дальше:                               ║"
echo "║   1. python preflight_check.py         ║"
echo "║   2. python prepare_dataset.py         ║"
echo "║   3. nohup python train.py &           ║"
echo "║   4. python evaluate.py                ║"
echo "╚════════════════════════════════════════╝"
