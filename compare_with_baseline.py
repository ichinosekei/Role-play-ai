#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  COMPARE WITH BASELINE — сравнение с Qwen2.5-32B без LoRA
═══════════════════════════════════════════════════════════════

ВНИМАНИЕ: требует A100 80GB (или 40GB).
На RTX 4060 Ti 8GB НЕ запустится — 32B-модель не влезет.

Где запускать (по простоте):
  1. RunPod (~$0.5-1/час): https://www.runpod.io/ → Spot A100
  2. vast.ai (~$0.3-0.5/час): https://vast.ai
  3. Yandex DataSphere (A100 за рубли)
  4. Google Colab Pro+ (иногда даёт A100)

Время работы: 30-40 минут на A100.

Кастомный HF cache (если на диске мало места в home):
    export HF_HOME=/mnt/data/hf_cache
    export TRANSFORMERS_CACHE=/mnt/data/hf_cache
    python compare_with_baseline.py

Использование:
    python compare_with_baseline.py
"""

import os

# ═══════════════════════════════════════════════════════════
#  Настройка путей кеша (если нужно поменять)
# ═══════════════════════════════════════════════════════════
# Раскомментируй и поменяй пути если HF cache съедает диск C/home:
# os.environ["HF_HOME"] = "/mnt/data/hf_cache"
# os.environ["TRANSFORMERS_CACHE"] = "/mnt/data/hf_cache"
# os.environ["HF_DATASETS_CACHE"] = "/mnt/data/hf_cache/datasets"

import json
import math
import sys
import gc
from pathlib import Path
import torch

# Используем функции из evaluate.py
sys.path.insert(0, ".")
from evaluate import (
    perplexity,
    reference_metrics,
    text_features,
    aggregate_features,
    length_distribution_match,
    vocabulary_overlap,
    type_token_ratio,
    cross_sample_diversity,
    style_match_score,
    RP,
)

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_from_disk

import numpy as np


def evaluate_model(model_name, save_path, max_seq=4096):
    """Прогоняет полный набор метрик на модели."""
    print(f"\n{'='*60}")
    print(f"  Оцениваем: {model_name}")
    print(f"{'='*60}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    eval_ds = load_from_disk("data/eval")

    results = {"model_name": model_name}

    # Group 1
    print("\n[1/5] Perplexity...")
    results["perplexity_overall"] = perplexity(model, tokenizer, eval_ds, max_n=300)
    results["perplexity_en"] = perplexity(model, tokenizer, eval_ds, max_n=200, lang_filter="en")
    results["perplexity_ru"] = perplexity(model, tokenizer, eval_ds, max_n=200, lang_filter="ru")
    if results["perplexity_overall"]:
        print(f"  PPL all: {results['perplexity_overall'].get('perplexity', '—'):.2f}")

    # Group 2
    print("\n[2/5] BLEU/ROUGE/BERTScore/chrF++...")
    out = reference_metrics(model, tokenizer, eval_ds, n=100)
    if isinstance(out, tuple) and len(out) == 3:
        ref_metrics, refs, preds = out
    else:
        ref_metrics = out
        refs, preds = [], []
    results["reference_metrics"] = ref_metrics
    if ref_metrics:
        print(f"  BLEU-4: {ref_metrics.get('bleu', 0):.2f}")
        print(f"  BERTScore F1: {ref_metrics.get('bertscore_f1', 0):.4f}")
        print(f"  chrF++: {ref_metrics.get('chrf++', 0):.2f}")
        print(f"  ROUGE-L: {ref_metrics.get('rouge_l', 0):.4f}")

    # Group 3
    print("\n[3/5] Style features...")
    ref_features = [text_features(r) for r in refs if text_features(r)]
    pred_features = [text_features(p) for p in preds if text_features(p)]
    results["reference_features"] = aggregate_features(ref_features)
    results["model_features"] = aggregate_features(pred_features)

    # Group 4
    print("\n[4/5] Length/vocab match + diversity...")
    results["length_js"] = length_distribution_match(refs, preds)
    results["vocab_overlap"] = vocabulary_overlap(refs, preds)
    results["ttr_model"] = type_token_ratio(preds)

    rp_samples = cross_sample_diversity(model, tokenizer, RP, n_per=3)
    if rp_samples:
        results["avg_distinct_2"] = float(np.mean([s["distinct_2"] for s in rp_samples]))
        results["avg_self_rep"] = float(np.mean([s["self_rep"] for s in rp_samples]))
        print(f"  Distinct-2: {results['avg_distinct_2']:.4f}")

    # Group 5
    print("\n[5/5] Style Match Score...")
    results["style_match_score"] = style_match_score(
        results["reference_features"],
        results["model_features"],
        results["length_js"],
    )
    if results["style_match_score"]:
        print(f"  Style Match: {results['style_match_score']:.4f}")

    # Save
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    Path("results").mkdir(exist_ok=True)

    # 1. Базовая модель БЕЗ LoRA
    print("Оцениваем базовую Qwen2.5-32B-Instruct (без файнтюна)...")
    base_results = evaluate_model(
        "Qwen/Qwen2.5-32B-Instruct",
        "results/eval_baseline.json"
    )

    # 2. Берём готовые результаты нашей модели
    with open("results/eval_full.json", encoding="utf-8") as f:
        our_results = json.load(f)

    # 3. Сравнительная таблица
    print(f"\n{'='*60}")
    print("  СРАВНЕНИЕ: Base Qwen vs Наша модель")
    print(f"{'='*60}")

    def gv(d, *path):
        for p in path:
            d = d.get(p) if isinstance(d, dict) else None
            if d is None:
                return None
        return d

    comparisons = [
        ("Perplexity",    ("perplexity_overall", "perplexity"), "lower"),
        ("PPL (EN)",      ("perplexity_en", "perplexity"),       "lower"),
        ("PPL (RU)",      ("perplexity_ru", "perplexity"),       "lower"),
        ("BLEU-4",        ("reference_metrics", "bleu"),         "higher"),
        ("BERTScore F1",  ("reference_metrics", "bertscore_f1"), "higher"),
        ("chrF++",        ("reference_metrics", "chrf++"),       "higher"),
        ("ROUGE-L",       ("reference_metrics", "rouge_l"),      "higher"),
        ("Length JS-div", ("length_js",),                        "lower"),
        ("Vocab overlap", ("vocab_overlap",),                    "higher"),
        ("Style Match",   ("style_match_score",),                "higher"),
        ("Distinct-2",    ("avg_distinct_2",),                   "higher"),
        ("Self-rep",      ("avg_self_rep",),                     "lower"),
    ]

    rows = []
    print(f"\n{'Метрика':<18} {'Base':>10} {'Ours':>10} {'Δ':>12} {'?':>3}")
    print('-'*60)
    for name, path, direction in comparisons:
        bv = gv(base_results, *path)
        ov = gv(our_results, *path)
        if bv is None or ov is None:
            continue
        try:
            bv, ov = float(bv), float(ov)
        except (TypeError, ValueError):
            continue

        delta = ov - bv
        delta_pct = 100 * delta / abs(bv) if bv != 0 else 0

        if direction == "lower":
            better = "✓" if delta < 0 else "✗"
        else:
            better = "✓" if delta > 0 else "✗"

        print(f"{name:<18} {bv:>10.4f} {ov:>10.4f} {delta:>+10.4f}({delta_pct:+5.1f}%) {better}")
        rows.append((name, bv, ov, delta, delta_pct, direction))

    # 4. Сохраняем
    with open("results/comparison_base_vs_ours.json", "w", encoding="utf-8") as f:
        json.dump({"base": base_results, "ours": our_results,
                   "comparison": [{"metric": r[0], "base": r[1], "ours": r[2],
                                   "delta": r[3], "delta_pct": r[4],
                                   "direction": r[5]} for r in rows]},
                  f, indent=2, default=str, ensure_ascii=False)

    # 5. LaTeX-таблица для прямой вставки в отчёт
    latex = "\\begin{table}[tbh!]\n\\begin{center}\n\\begin{tabular}{|l|cc|c|}\n\\hline\n"
    latex += "Метрика & Base Qwen2.5-32B & Наша модель & $\\Delta$ \\\\\n\\hline\n"
    for name, bv, ov, delta, delta_pct, direction in rows:
        sign = "+" if delta > 0 else ""
        latex += f"{name} & {bv:.4f} & {ov:.4f} & {sign}{delta:.4f} ({sign}{delta_pct:.1f}\\%) \\\\\n"
    latex += "\\hline\n\\end{tabular}\n"
    latex += "\\caption{Сравнение базовой Qwen2.5-32B-Instruct и нашей дообученной модели.}\n"
    latex += "\\label{tab:baseline_comparison}\n\\end{center}\n\\end{table}\n"

    with open("results/baseline_comparison_table.tex", "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"\n✓ JSON: results/comparison_base_vs_ours.json")
    print(f"✓ LaTeX-таблица: results/baseline_comparison_table.tex")
    print(f"\nДля отчёта:")
    print(f"  1. Скопируй содержимое baseline_comparison_table.tex")
    print(f"  2. Вставь в report.tex в раздел 5.3 Baselines")
    print(f"  3. Замени фразу 'Не было выполнено в рамках текущей итерации' на ссылку на эту таблицу")


if __name__ == "__main__":
    main()
