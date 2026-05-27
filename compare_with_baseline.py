"""
Сравнение базовой `Qwen/Qwen2.5-32B-Instruct` и дообученной модели (LoRA).

Если нужно вынести HF cache:
  export HF_HOME=/mnt/data/hf_cache
  export TRANSFORMERS_CACHE=/mnt/data/hf_cache
"""

import os

import json
import math
import sys
import gc
from pathlib import Path
import torch

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
    print(f"  Оцениваем: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    eval_ds = load_from_disk("data/eval")

    results = {"model_name": model_name}

    print("\n[1/5] Perplexity...")
    results["perplexity_overall"] = perplexity(model, tokenizer, eval_ds, max_n=300)
    results["perplexity_en"] = perplexity(model, tokenizer, eval_ds, max_n=200, lang_filter="en")
    results["perplexity_ru"] = perplexity(model, tokenizer, eval_ds, max_n=200, lang_filter="ru")
    if results["perplexity_overall"]:
        print(f"  PPL all: {results['perplexity_overall'].get('perplexity', '—'):.2f}")

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

    print("\n[3/5] Style features...")
    ref_features = [text_features(r) for r in refs if text_features(r)]
    pred_features = [text_features(p) for p in preds if text_features(p)]
    results["reference_features"] = aggregate_features(ref_features)
    results["model_features"] = aggregate_features(pred_features)

    print("\n[4/5] Length/vocab match + diversity...")
    results["length_js"] = length_distribution_match(refs, preds)
    results["vocab_overlap"] = vocabulary_overlap(refs, preds)
    results["ttr_model"] = type_token_ratio(preds)

    rp_samples = cross_sample_diversity(model, tokenizer, RP, n_per=3)
    if rp_samples:
        results["avg_distinct_2"] = float(np.mean([s["distinct_2"] for s in rp_samples]))
        results["avg_self_rep"] = float(np.mean([s["self_rep"] for s in rp_samples]))
        print(f"  Distinct-2: {results['avg_distinct_2']:.4f}")

    print("\n[5/5] Style Match Score...")
    results["style_match_score"] = style_match_score(
        results["reference_features"],
        results["model_features"],
        results["length_js"],
    )
    if results["style_match_score"]:
        print(f"  Style Match: {results['style_match_score']:.4f}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    Path("results").mkdir(exist_ok=True)

    print("Оцениваем базовую Qwen2.5-32B-Instruct (без файнтюна)...")
    base_results = evaluate_model(
        "Qwen/Qwen2.5-32B-Instruct",
        "results/eval_baseline.json"
    )

    with open("results/eval_full.json", encoding="utf-8") as f:
        our_results = json.load(f)

    print("  СРАВНЕНИЕ: Base Qwen vs Наша модель")

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
    print()
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
            better = "V" if delta < 0 else "X"
        else:
            better = "V" if delta > 0 else "X"

        print(f"{name:<18} {bv:>10.4f} {ov:>10.4f} {delta:>+10.4f}({delta_pct:+5.1f}%) {better}")
        rows.append((name, bv, ov, delta, delta_pct, direction))

    with open("results/comparison_base_vs_ours.json", "w", encoding="utf-8") as f:
        json.dump({"base": base_results, "ours": our_results,
                   "comparison": [{"metric": r[0], "base": r[1], "ours": r[2],
                                   "delta": r[3], "delta_pct": r[4],
                                   "direction": r[5]} for r in rows]},
                  f, indent=2, default=str, ensure_ascii=False)

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

    print(f"\n JSON: results/comparison_base_vs_ours.json")
    print(f" LaTeX-таблица: results/baseline_comparison_table.tex")


if __name__ == "__main__":
    main()
