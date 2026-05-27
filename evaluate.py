                      
"""
Оценка модели (набор метрик + HTML отчёт).
"""

import os
import subprocess


def _ensure_cuda_linker_for_triton():
    """Фикс для окружений, где линкер не видит `-lcuda` при сборке Triton."""
    if os.environ.get("UNSLOTH_SKIP_CUDA_LINKER_FIX"):
        return
    try:
        proc = subprocess.run(
            ["ldconfig", "-p"],
            capture_output=True,
            text=True,
            check=False,
        )
        libcuda_so1 = None
        for line in proc.stdout.splitlines():
            if "libcuda.so.1" in line:
                libcuda_so1 = line.split()[-1]
                break
        if not libcuda_so1 or not os.path.isfile(libcuda_so1):
            return
        cuda_dir = os.path.dirname(libcuda_so1)
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            vlib = os.path.join(venv, "lib")
            os.makedirs(vlib, exist_ok=True)
            link = os.path.join(vlib, "libcuda.so")
            try:
                if os.path.lexists(link):
                    os.unlink(link)
                os.symlink(libcuda_so1, link)
            except OSError:
                pass

        def merge_env(key):
            sep = os.pathsep
            parts = [cuda_dir]
            if venv:
                parts.append(os.path.join(venv, "lib"))
            rest = os.environ.get(key, "")
            if rest:
                parts.extend(rest.split(sep))
            seen = set()
            out = []
            for p in parts:
                if p and p not in seen:
                    seen.add(p)
                    out.append(p)
            os.environ[key] = sep.join(out)

        merge_env("LD_LIBRARY_PATH")
        merge_env("LIBRARY_PATH")
    except Exception:
        pass


_ensure_cuda_linker_for_triton()

import re
import json
import math
import torch
import string
from pathlib import Path
from collections import Counter
from datetime import datetime
import numpy as np

from datasets import load_from_disk
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


LORA_PATH = "model_final"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
N_GEN_SAMPLES = 100                                             
N_PER_PROMPT = 5                                 


CASUAL = [
    {"lang": "en", "system": "You are a friendly assistant chatting casually.",
     "user": "Hey, I had a rough day. Cheer me up?"},
    {"lang": "ru", "system": "Ты дружелюбный AI-собеседник.",
     "user": "Привет! Какой твой любимый фильм?"},
    {"lang": "en", "system": "You are a knowledgeable assistant.",
     "user": "Explain quantum entanglement simply."},
    {"lang": "ru", "system": "Ты помощник пользователя.",
     "user": "Помоги написать резюме программиста."},
]

RP = [
    {"lang": "en", "system": ("You are Captain Maria, a sharp space pirate. "
                              "Cynical but caring. Use confident voice."),
     "user": "*walks onto deck* Captain, imperials are catching up."},
    {"lang": "ru", "system": ("Ты Лиза, 25, бариста в крафтовой кофейне. "
                              "Острая на язык, флиртуешь с интересными гостями."),
     "user": "*сажусь за стойку* Налей чего-нибудь интересного."},
    {"lang": "en", "system": ("You are Aldric, ancient elven mage. Wise, melancholic. "
                              "Use poetic phrases."),
     "user": "Old one, why do mortals fear death?"},
    {"lang": "en", "system": ("You are Mira, a tavern owner. Warm, motherly, "
                              "but firm with troublemakers."),
     "user": "*sits at the bar* Tough day, Mira."},
]


def perplexity(model, tokenizer, dataset, max_n=300, lang_filter=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    n = 0

    with torch.no_grad():
        for i in range(min(max_n, len(dataset))):
            text = dataset[i]["text"]
            cyr_ratio = sum(1 for c in text if 'а' <= c.lower() <= 'я') / max(len(text), 1)
            is_ru = cyr_ratio > 0.3
            if lang_filter == "ru" and not is_ru:
                continue
            if lang_filter == "en" and is_ru:
                continue

            inputs = tokenizer(text, return_tensors="pt",
                              truncation=True, max_length=2048).to("cuda")
            out = model(**inputs, labels=inputs["input_ids"])
            tlen = inputs["input_ids"].shape[1]
            total_loss += out.loss.item() * tlen
            total_tokens += tlen
            n += 1

    if total_tokens == 0:
        return None
    avg_loss = total_loss / total_tokens
    return {"avg_loss": avg_loss, "perplexity": math.exp(avg_loss), "n_samples": n}


def token_accuracy(model, tokenizer, dataset, max_n=100):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(min(max_n, len(dataset))):
            inputs = tokenizer(dataset[i]["text"], return_tensors="pt",
                              truncation=True, max_length=1024).to("cuda")
            ids = inputs["input_ids"]
            logits = model(**inputs).logits
            preds = logits[:, :-1].argmax(dim=-1)
            target = ids[:, 1:]
            correct += (preds == target).sum().item()
            total += target.numel()
    return correct / total if total > 0 else 0


def split_last_response(text, tokenizer):
    parts = text.split("<|im_start|>assistant\n")
    if len(parts) < 2:
        return None, None
    context = "<|im_start|>assistant\n".join(parts[:-1])
    ref = parts[-1].replace("<|im_end|>", "").strip()
    return context + "<|im_start|>assistant\n", ref


def gen(model, tokenizer, prompt, max_new=200):
    inputs = tokenizer(prompt, return_tensors="pt",
                      truncation=True, max_length=4096).to("cuda")
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new,
            temperature=0.8, top_p=0.9, top_k=50,
            repetition_penalty=1.05,
            do_sample=True,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).split("<|im_end|>")[0].strip()


def reference_metrics(model, tokenizer, dataset, n=N_GEN_SAMPLES):
    print(f"  Генерация {n} ответов для reference-метрик...")
    FastLanguageModel.for_inference(model)

    refs, preds = [], []
    for i in range(min(n, len(dataset))):
        prompt, ref = split_last_response(dataset[i]["text"], tokenizer)
        if not prompt or not ref or len(ref) < 30:
            continue
        try:
            pred = gen(model, tokenizer, prompt, max_new=200)
            if pred:
                refs.append(ref[:500])
                preds.append(pred[:500])
        except Exception:
            continue
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n}")

    if not refs:
        return {}

    out = {"n_pairs": len(refs)}

    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        out["bleu"] = bleu.score
    except Exception as e:
        print(f"    BLEU failed: {e}")

    try:
        import sacrebleu
        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
        out["chrf++"] = chrf.score
    except Exception as e:
        print(f"    chrF++ failed: {e}")

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]
        out["rouge_l"] = sum(scores) / len(scores)
    except Exception as e:
        print(f"    ROUGE failed: {e}")

    try:
        from bert_score import score as bs
        P, R, F1 = bs(preds, refs, lang="en", verbose=False)
        out["bertscore_f1"] = F1.mean().item()
        out["bertscore_p"] = P.mean().item()
        out["bertscore_r"] = R.mean().item()
    except Exception as e:
        print(f"    BERTScore failed: {e}")

    return out, refs, preds


EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002600-\U000027BF"
    "\U0001F600-\U0001F64F"
    "]+", flags=re.UNICODE
)

ACTION_RE = re.compile(r'\*[^*]{2,80}\*')


def text_features(text):
    if not text:
        return None

    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    n_chars = len(text)
    n_words = len(words)
    n_sent = max(len(sentences), 1)
    avg_word_len = sum(len(w) for w in words) / max(n_words, 1)
    avg_sent_len = n_words / n_sent

    caps_words = sum(1 for w in words if w and w[0].isupper())
    all_lower = sum(1 for w in words if w.islower() and len(w) > 1)
    caps_rate = caps_words / max(n_words, 1)
    lower_rate = all_lower / max(n_words, 1)

    n_periods = text.count('.')
    n_commas = text.count(',')
    n_exclaim = text.count('!')
    n_quest = text.count('?')
    n_dashes = text.count('') + text.count(' - ')
    n_ellipsis = text.count('...')
    period_rate = n_periods / max(n_sent, 1)

    n_emoji = len(EMOJI_RE.findall(text))
    n_actions = len(ACTION_RE.findall(text))
    emoji_rate = n_emoji / max(n_words, 1) * 100                   
    action_rate = n_actions / max(n_words, 1) * 100

    return {
        "n_words": n_words,
        "avg_word_len": avg_word_len,
        "avg_sent_len": avg_sent_len,
        "caps_rate": caps_rate,
        "lower_rate": lower_rate,
        "period_rate": period_rate,
        "comma_per_100w": n_commas / max(n_words, 1) * 100,
        "exclaim_per_100w": n_exclaim / max(n_words, 1) * 100,
        "quest_per_100w": n_quest / max(n_words, 1) * 100,
        "ellipsis_per_100w": n_ellipsis / max(n_words, 1) * 100,
        "emoji_per_100w": emoji_rate,
        "action_per_100w": action_rate,
    }


def aggregate_features(features_list):
    if not features_list:
        return {}
    keys = features_list[0].keys()
    agg = {}
    for k in keys:
        vals = [f[k] for f in features_list if f.get(k) is not None]
        if not vals:
            continue
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_p25"] = float(np.percentile(vals, 25))
        agg[f"{k}_p50"] = float(np.percentile(vals, 50))
        agg[f"{k}_p75"] = float(np.percentile(vals, 75))
    return agg


def js_divergence(p, q, eps=1e-10):
    p = np.array(p) + eps
    q = np.array(q) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


def length_distribution_match(refs, preds, bins=20):
    ref_lens = [len(r.split()) for r in refs]
    pred_lens = [len(p.split()) for p in preds]
    if not ref_lens or not pred_lens:
        return None

    max_len = max(max(ref_lens), max(pred_lens))
    edges = np.linspace(0, max_len, bins + 1)
    ref_hist, _ = np.histogram(ref_lens, bins=edges)
    pred_hist, _ = np.histogram(pred_lens, bins=edges)
    return js_divergence(ref_hist, pred_hist)


def vocabulary_overlap(refs, preds, top_n=200):
    ref_words = []
    pred_words = []
    for r in refs:
        ref_words.extend(re.findall(r'\w+', r.lower()))
    for p in preds:
        pred_words.extend(re.findall(r'\w+', p.lower()))
    if not ref_words:
        return None
    ref_top = set(w for w, _ in Counter(ref_words).most_common(top_n))
    pred_set = set(pred_words)
    overlap = len(ref_top & pred_set) / len(ref_top)
    return overlap


def type_token_ratio(texts):
    all_words = []
    for t in texts:
        all_words.extend(re.findall(r'\w+', t.lower()))
    if not all_words:
        return 0
    return len(set(all_words)) / len(all_words)


def self_repetition_rate(text, n=4):
    words = text.split()
    if len(words) < n + 1:
        return 0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def distinct_n(texts, n=2):
    all_ngrams = []
    for t in texts:
        words = t.split()
        if len(words) >= n:
            all_ngrams.extend(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    if not all_ngrams:
        return 0
    return len(set(all_ngrams)) / len(all_ngrams)


def cross_sample_diversity(model, tokenizer, prompts, n_per=N_PER_PROMPT):
    print(f"  Cross-sample diversity ({len(prompts)} промптов  {n_per} samples)...")
    FastLanguageModel.for_inference(model)
    all_results = []

    for p in prompts:
        msgs = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"]},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        responses = []
        for _ in range(n_per):
            try:
                resp = gen(model, tokenizer, prompt)
                if resp:
                    responses.append(resp)
            except Exception:
                continue

        if not responses:
            continue

        all_results.append({
            "prompt": p,
            "responses": responses,
            "distinct_2": distinct_n(responses, 2),
            "distinct_3": distinct_n(responses, 3),
            "self_rep": np.mean([self_repetition_rate(r) for r in responses]),
            "len_std": float(np.std([len(r.split()) for r in responses])),
        })

    return all_results


def style_match_score(ref_features, pred_features, length_js):
    """
    Композитный балл: насколько модель пишет в стиле reference.
    Считаем нормированную разницу по ключевым стилевым фичам.
    """
    if not ref_features or not pred_features:
        return None

    keys = [
        "avg_word_len_mean", "avg_sent_len_mean",
        "comma_per_100w_mean", "exclaim_per_100w_mean",
        "quest_per_100w_mean", "ellipsis_per_100w_mean",
        "emoji_per_100w_mean", "action_per_100w_mean",
        "caps_rate_mean",
    ]

    diffs = []
    for k in keys:
        ref_v = ref_features.get(k)
        pred_v = pred_features.get(k)
        if ref_v is None or pred_v is None:
            continue
        denom = max(abs(ref_v), abs(pred_v), 1e-6)
        diff = abs(ref_v - pred_v) / denom
        diffs.append(min(diff, 1.0))

    if not diffs:
        return None
    style_diff = np.mean(diffs)
    style_score = 1.0 - style_diff

    if length_js is not None:
        style_score *= (1.0 - min(length_js, 0.5))

    return float(max(0, style_score))


def generate_html(results):
    def m(key, default=""):
        if "." in key:
            v = results
            for part in key.split("."):
                if isinstance(v, dict):
                    v = v.get(part)
                else:
                    v = None
                    break
        else:
            v = results.get(key)
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"
        return str(v) if v is not None else default

    sm = results.get("style_match_score")
    sm_color = "#1D9E75" if sm and sm > 0.7 else "#D85A30" if sm and sm > 0.5 else "#E24B4A"

    ppl = results.get("perplexity_overall", {}).get("perplexity")
    ppl_color = "#1D9E75" if ppl and ppl < 12 else "#D85A30" if ppl and ppl < 20 else "#E24B4A"
    ppl_cell = f"{float(ppl):.2f}" if ppl is not None else ""

    bert = results.get("reference_metrics", {}).get("bertscore_f1")
    bert_color = "#1D9E75" if bert and bert > 0.85 else "#D85A30" if bert and bert > 0.75 else "#E24B4A"

    metrics_table = f"""
    <h3>Group 1  Loss-based</h3>
    <table>
      <tr><th>Метрика</th><th>Значение</th><th>Цель</th></tr>
      <tr><td>Perplexity (all)</td><td><strong style="color:{ppl_color}">{ppl_cell}</strong></td><td>5-12</td></tr>
      <tr><td>Perplexity (EN)</td><td>{results.get('perplexity_en', {}).get('perplexity', '')}</td><td>5-12</td></tr>
      <tr><td>Perplexity (RU)</td><td>{results.get('perplexity_ru', {}).get('perplexity', '')}</td><td>5-15</td></tr>
      <tr><td>Token accuracy</td><td>{m('token_accuracy')}</td><td>&gt; 0.55</td></tr>
    </table>

    <h3>Group 2  Reference-based</h3>
    <table>
      <tr><th>Метрика</th><th>Значение</th><th>Цель</th></tr>
      <tr><td>BLEU-4</td><td>{m('reference_metrics.bleu')}</td><td>&gt; 5</td></tr>
      <tr><td>ROUGE-L</td><td>{m('reference_metrics.rouge_l')}</td><td>&gt; 0.20</td></tr>
      <tr><td>BERTScore F1</td><td><strong style="color:{bert_color}">{m('reference_metrics.bertscore_f1')}</strong></td><td>&gt; 0.85</td></tr>
      <tr><td>chrF++</td><td>{m('reference_metrics.chrf++')}</td><td>&gt; 25</td></tr>
    </table>

    <h3>Group 3  Style match</h3>
    <table>
      <tr><th>Метрика</th><th>Reference</th><th>Model</th><th>Diff</th></tr>
    """

    ref_f = results.get("reference_features", {})
    pred_f = results.get("model_features", {})
    style_keys = [
        ("avg_word_len_mean", "Avg word length"),
        ("avg_sent_len_mean", "Avg sentence length"),
        ("comma_per_100w_mean", "Commas / 100w"),
        ("exclaim_per_100w_mean", "! / 100w"),
        ("quest_per_100w_mean", "? / 100w"),
        ("ellipsis_per_100w_mean", "... / 100w"),
        ("emoji_per_100w_mean", "Emoji / 100w"),
        ("action_per_100w_mean", "*action* / 100w"),
        ("caps_rate_mean", "Capitalized rate"),
    ]
    for k, label in style_keys:
        rv = ref_f.get(k, 0)
        pv = pred_f.get(k, 0)
        d = abs(rv - pv) / max(abs(rv), abs(pv), 1e-6)
        metrics_table += f"<tr><td>{label}</td><td>{rv:.3f}</td><td>{pv:.3f}</td><td>{d:.2f}</td></tr>"

    metrics_table += f"""
    </table>

    <h3>Group 4  RP-specific</h3>
    <table>
      <tr><th>Метрика</th><th>Значение</th><th>Цель</th></tr>
      <tr><td>Vocabulary overlap</td><td>{m('vocab_overlap')}</td><td>&gt; 0.55</td></tr>
      <tr><td>Length JS-divergence</td><td>{m('length_js')}</td><td>&lt; 0.10</td></tr>
      <tr><td>TTR (model)</td><td>{m('ttr_model')}</td><td>0.4-0.6</td></tr>
      <tr><td>Avg distinct-2 (cross-sample)</td><td>{m('avg_distinct_2')}</td><td>&gt; 0.7</td></tr>
      <tr><td>Avg self-repetition</td><td>{m('avg_self_rep')}</td><td>&lt; 0.05</td></tr>
    </table>

    <h3>Group 5  Composite</h3>
    <table>
      <tr><th>Метрика</th><th>Значение</th><th>Цель</th></tr>
      <tr><td><strong>Style Match Score</strong></td>
          <td><strong style="color:{sm_color}">{m('style_match_score')}</strong></td>
          <td>&gt; 0.7</td></tr>
    </table>
    """

    examples_html = ""
    for r in results.get("rp_samples", []):
        responses_html = "".join(
            f"<div class='resp'>{resp[:400]}{'...' if len(resp) > 400 else ''}</div>"
            for resp in r["responses"][:3]
        )
        examples_html += f"""
        <div class="example">
            <div class="meta">{r['prompt']['lang'].upper()} | distinct-2: {r['distinct_2']:.3f}</div>
            <div class="user">User: {r['prompt']['user']}</div>
            {responses_html}
        </div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Evaluation Report</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, sans-serif; background:#faf9f6; color:#2c2c2a;
       padding:2rem; line-height:1.6; max-width:1100px; margin:0 auto; }}
h1 {{ font-size:28px; font-weight:600; }}
h2 {{ font-size:20px; margin:2rem 0 1rem; border-bottom:1px solid #e5e3dd; padding-bottom:8px; }}
h3 {{ font-size:16px; margin:1.5rem 0 0.5rem; }}
.meta {{ color:#888; font-size:14px; }}
table {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:14px; }}
th {{ background:#f1efe8; padding:8px; text-align:left; font-weight:500; font-size:12px;
     text-transform:uppercase; }}
td {{ padding:8px; border-bottom:0.5px solid #e5e3dd; }}
.example {{ background:#fff; border-radius:8px; padding:1rem; margin:1rem 0;
           border:0.5px solid #e5e3dd; }}
.user {{ background:#f0eee8; padding:8px 12px; border-radius:6px; margin:6px 0; font-size:14px; }}
.resp {{ background:#e1f5ee; padding:8px 12px; border-radius:6px; margin:6px 0; font-size:14px; }}
</style></head><body>

<h1>Evaluation Report  {results['model_name']}</h1>
<div class="meta">{results['timestamp']}</div>

<h2>Метрики</h2>
{metrics_table}

<h2>Примеры RP-генераций (cross-sample diversity)</h2>
{examples_html}

</body></html>"""

    with open(RESULTS_DIR / "eval_report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n HTML: {RESULTS_DIR / 'eval_report.html'}")


def flatten(d, parent="", sep="."):
    items = {}
    for k, v in d.items():
        new = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.update(flatten(v, new, sep))
        else:
            items[new] = v
    return items


def main():
    print("  ПОЛНАЯ ОЦЕНКА")

            
    print(f"\nЗагружаем {LORA_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_PATH, max_seq_length=4096, load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    eval_ds = load_from_disk("data/eval")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model_name": LORA_PATH,
    }

                                     
    print("\n[1/5] Loss-based metrics")
    results["perplexity_overall"] = perplexity(model, tokenizer, eval_ds, max_n=300) or {}
    print(f"  PPL all: {results['perplexity_overall'].get('perplexity', '')}")
    results["perplexity_en"] = perplexity(model, tokenizer, eval_ds, max_n=200, lang_filter="en") or {}
    print(f"  PPL EN:  {results['perplexity_en'].get('perplexity', '')}")
    results["perplexity_ru"] = perplexity(model, tokenizer, eval_ds, max_n=200, lang_filter="ru") or {}
    print(f"  PPL RU:  {results['perplexity_ru'].get('perplexity', '')}")
    results["token_accuracy"] = token_accuracy(model, tokenizer, eval_ds, max_n=50)
    print(f"  Token accuracy: {results['token_accuracy']:.4f}")

                                          
    print("\n[2/5] Reference-based metrics")
    ref_metrics, refs, preds = reference_metrics(model, tokenizer, eval_ds, n=N_GEN_SAMPLES)
    results["reference_metrics"] = ref_metrics
    print(f"  BLEU: {ref_metrics.get('bleu', '')}")
    print(f"  ROUGE-L: {ref_metrics.get('rouge_l', '')}")
    print(f"  BERTScore F1: {ref_metrics.get('bertscore_f1', '')}")
    print(f"  chrF++: {ref_metrics.get('chrf++', '')}")

                                         
    print("\n[3/5] Style features")
    ref_features_list = [text_features(r) for r in refs if text_features(r)]
    pred_features_list = [text_features(p) for p in preds if text_features(p)]
    results["reference_features"] = aggregate_features(ref_features_list)
    results["model_features"] = aggregate_features(pred_features_list)
    print(f"  Ref avg sent len: {results['reference_features'].get('avg_sent_len_mean', 0):.1f}")
    print(f"  Model avg sent len: {results['model_features'].get('avg_sent_len_mean', 0):.1f}")
    print(f"  Ref *action* rate: {results['reference_features'].get('action_per_100w_mean', 0):.2f}")
    print(f"  Model *action* rate: {results['model_features'].get('action_per_100w_mean', 0):.2f}")

                                      
    print("\n[4/5] RP-specific")
    results["length_js"] = length_distribution_match(refs, preds)
    print(f"  Length JS-div: {results['length_js']:.4f}")
    results["vocab_overlap"] = vocabulary_overlap(refs, preds)
    print(f"  Vocab overlap: {results['vocab_overlap']:.4f}")
    results["ttr_reference"] = type_token_ratio(refs)
    results["ttr_model"] = type_token_ratio(preds)
    print(f"  TTR ref: {results['ttr_reference']:.4f}, model: {results['ttr_model']:.4f}")

                                           
    rp_samples = cross_sample_diversity(model, tokenizer, RP, n_per=N_PER_PROMPT)
    results["rp_samples"] = rp_samples
    if rp_samples:
        results["avg_distinct_2"] = float(np.mean([s["distinct_2"] for s in rp_samples]))
        results["avg_distinct_3"] = float(np.mean([s["distinct_3"] for s in rp_samples]))
        results["avg_self_rep"] = float(np.mean([s["self_rep"] for s in rp_samples]))
        print(f"  Distinct-2: {results['avg_distinct_2']:.4f}")
        print(f"  Self-repetition: {results['avg_self_rep']:.4f}")

                                    
    print("\n[5/5] Composite Style Match Score")
    results["style_match_score"] = style_match_score(
        results["reference_features"],
        results["model_features"],
        results["length_js"],
    )
    print(f"  Style Match Score: {results['style_match_score']:.4f}")

                           
    with open(RESULTS_DIR / "eval_full.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

                                     
    flat = flatten(results)
    flat = {k: v for k, v in flat.items()
            if isinstance(v, (int, float, str)) and not k.startswith("rp_samples")}
    with open(RESULTS_DIR / "eval_flat.json", "w") as f:
        json.dump(flat, f, indent=2, default=str)

    generate_html(results)

          
    print()
    print("  ИТОГИ")
    print(f"  Perplexity:      {results['perplexity_overall'].get('perplexity', ''):.2f}")
    print(f"  BERTScore F1:    {results['reference_metrics'].get('bertscore_f1', 0):.4f}")
    print(f"  Style Match:     {results['style_match_score']:.4f}")
    print(f"  Distinct-2:      {results.get('avg_distinct_2', 0):.4f}")
    print(f"\nДеталь: results/eval_report.html")


if __name__ == "__main__":
    main()
