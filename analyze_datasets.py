"""
Небольшая аналитика по датасетам (в разрезе источников).

Выход:
- results/dataset_analytics.json
- results/dataset_analytics.html
"""

import json
import re
import random
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
import numpy as np

from hf_dataset_utils import load_hf_train, load_pippa_train_split

random.seed(42)
SAMPLE_SIZE = 5000              # сколько примеров берём из каждого датасета для анализа
TOKENIZER_NAME = "Qwen/Qwen2.5-32B-Instruct"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Загружаем токенизатор...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001F600-\U0001F64F]+",
    flags=re.UNICODE
)
ACTION_RE = re.compile(r'\*[^*]{2,80}\*')
NSFW_RE = re.compile(
    r'\b(cock|pussy|cum|fuck|sex|penis|vagina|nipple|orgasm|naked|nude|anal)\b',
    re.IGNORECASE
)


def detect_lang(text):
    """Простой детектор: cyrillic → ru, иначе → en."""
    if not text:
        return "unknown"
    cyr = sum(1 for c in text if 'а' <= c.lower() <= 'я')
    lat = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    total = cyr + lat
    if total < 20:
        return "short"
    if cyr / total > 0.3:
        return "ru"
    return "en"


def text_stats(text):
    """Возвращает stats по одному тексту."""
    if not text:
        return None
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sent = max(len(sentences), 1)

    return {
        "n_words": n_words,
        "n_chars": n_chars,
        "avg_word_len": sum(len(w) for w in words) / max(n_words, 1),
        "avg_sent_len": n_words / n_sent,
        "n_emoji": len(EMOJI_RE.findall(text)),
        "n_actions": len(ACTION_RE.findall(text)),
        "n_periods": text.count('.'),
        "n_commas": text.count(','),
        "n_exclaim": text.count('!'),
        "n_quest": text.count('?'),
        "n_ellipsis": text.count('...'),
        "n_caps_words": sum(1 for w in words if w and w[0].isupper()),
        "n_lower_words": sum(1 for w in words if w.islower() and len(w) > 1),
        "n_nsfw": len(NSFW_RE.findall(text)),
        "lang": detect_lang(text),
    }


def aggregate_stats(stats_list):
    """Агрегирует список stats в распределения."""
    if not stats_list:
        return {}

    numeric_keys = [k for k in stats_list[0] if isinstance(stats_list[0][k], (int, float))]
    agg = {}
    for k in numeric_keys:
        vals = [s[k] for s in stats_list if s.get(k) is not None]
        if not vals:
            continue
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(np.max(vals)),
        }

    langs = Counter(s["lang"] for s in stats_list)
    total = sum(langs.values())
    agg["lang_distribution"] = {k: v / total for k, v in langs.items()}

    return agg


def vocab_stats(texts, top_n=50):
    """Vocabulary stats по списку текстов."""
    all_words = []
    for t in texts:
        all_words.extend(re.findall(r'\w+', t.lower()))
    if not all_words:
        return {}
    counter = Counter(all_words)
    return {
        "total_words": len(all_words),
        "unique_words": len(counter),
        "ttr": len(counter) / len(all_words),
        "top_50": counter.most_common(top_n),
    }


def parse_pippa(n_samples):
    print(f"  PIPPA: загружаем...")
    ds = load_pippa_train_split()
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    dialogues = []
    for i in indices:
        ex = ds[i]
        persona = ex.get("bot_description", "") or ex.get("bot_persona", "")
        conv = ex.get("conversation", [])
        if not conv:
            continue
        msgs = []
        if persona:
            msgs.append({"role": "system", "content": persona})
        for turn in conv:
            content = turn.get("message", "").strip()
            if content:
                msgs.append({
                    "role": "user" if turn.get("is_human") else "assistant",
                    "content": content,
                })
        if len(msgs) >= 2:
            dialogues.append(msgs)
    return dialogues


def parse_limarp(n_samples):
    print(f"  LimaRP: загружаем...")
    try:
        ds = load_hf_train("lemonilia/LimaRP")
    except Exception as e:
        print(f"    ⚠ LimaRP недоступен: {e}")
        return []

    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    dialogues = []
    for i in indices:
        ex = ds[i]
        convs = ex.get("conversations") or ex.get("messages") or []
        if not convs:
            continue
        msgs = []
        for c in convs:
            role = c.get("from") or c.get("role")
            content = (c.get("value") or c.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                msgs.append({"role": "system", "content": content})
            elif role in ("human", "user"):
                msgs.append({"role": "user", "content": content})
            elif role in ("gpt", "assistant"):
                msgs.append({"role": "assistant", "content": content})
        if len(msgs) >= 2:
            dialogues.append(msgs)
    return dialogues


def parse_saiga(n_samples):
    print(f"  Saiga: загружаем...")
    ds = load_hf_train("IlyaGusev/saiga_scored")
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    dialogues = []
    for i in indices:
        ex = ds[i]
        msgs_raw = ex.get("messages", [])
        if not msgs_raw:
            continue
        msgs = [{"role": m.get("role"), "content": m.get("content", "").strip()}
                for m in msgs_raw if m.get("content")]
        if len(msgs) >= 2:
            dialogues.append(msgs)
    return dialogues


def parse_claude(n_samples):
    print(f"  Claude multiround: загружаем...")
    ds = load_hf_train("Norquinal/claude_multiround_chat_30k")
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    role_map = {"human": "user", "gpt": "assistant"}
    dialogues = []
    for i in indices:
        ex = ds[i]
        convs = ex.get("conversations", [])
        if not convs:
            continue
        msgs = []
        for c in convs:
            role = role_map.get(c.get("from"), c.get("from"))
            content = c.get("value", "").strip()
            if role and content:
                msgs.append({"role": role, "content": content})
        if len(msgs) >= 2:
            dialogues.append(msgs)
    return dialogues


def analyze_dataset(name, dialogues):
    """Считает все метрики для одного датасета."""
    print(f"\n  Анализ {name} ({len(dialogues)} диалогов)...")

    n_turns_list = []
    n_tokens_total = []
    system_prompt_lens = []
    user_messages = []
    assistant_messages = []
    all_messages = []

    for dlg in dialogues:
        non_sys = [m for m in dlg if m["role"] in ("user", "assistant")]
        n_turns_list.append(len(non_sys))

        full_text = " ".join(m["content"] for m in dlg)
        n_tokens_total.append(
            len(tokenizer(full_text, truncation=False, add_special_tokens=False)["input_ids"])
        )

        sys_msgs = [m for m in dlg if m["role"] == "system"]
        if sys_msgs:
            system_prompt_lens.append(len(sys_msgs[0]["content"].split()))

        for m in dlg:
            if m["role"] == "user":
                user_messages.append(m["content"])
            elif m["role"] == "assistant":
                assistant_messages.append(m["content"])
            all_messages.append(m["content"])

    user_stats = [text_stats(t) for t in user_messages if text_stats(t)]
    assistant_stats = [text_stats(t) for t in assistant_messages if text_stats(t)]

    return {
        "name": name,
        "n_dialogues": len(dialogues),
        "n_user_messages": len(user_messages),
        "n_assistant_messages": len(assistant_messages),
        "tokens_per_dialogue": {
            "mean": float(np.mean(n_tokens_total)),
            "median": float(np.median(n_tokens_total)),
            "p25": float(np.percentile(n_tokens_total, 25)),
            "p75": float(np.percentile(n_tokens_total, 75)),
            "max": int(max(n_tokens_total)),
        },
        "turns_per_dialogue": {
            "mean": float(np.mean(n_turns_list)),
            "median": float(np.median(n_turns_list)),
            "p25": float(np.percentile(n_turns_list, 25)),
            "p75": float(np.percentile(n_turns_list, 75)),
            "max": int(max(n_turns_list)),
        },
        "system_prompt_length_words": {
            "mean": float(np.mean(system_prompt_lens)) if system_prompt_lens else 0,
            "median": float(np.median(system_prompt_lens)) if system_prompt_lens else 0,
        },
        "user_message_stats": aggregate_stats(user_stats),
        "assistant_message_stats": aggregate_stats(assistant_stats),
        "vocabulary_user": vocab_stats(user_messages),
        "vocabulary_assistant": vocab_stats(assistant_messages),
    }


def fmt(v, prec=2):
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def make_comparison_table(results, metric_path, label, prec=2):
    """Сравнительная таблица: одна метрика по всем датасетам."""
    rows = ""
    for r in results:
        v = r
        for k in metric_path.split("."):
            v = v.get(k, "—") if isinstance(v, dict) else "—"
        rows += f"<tr><td>{r['name']}</td><td>{fmt(v, prec) if v != '—' else '—'}</td></tr>"
    return f"""<h4>{label}</h4>
<table><tr><th>Dataset</th><th>{label}</th></tr>{rows}</table>"""


def make_dataset_card(r):
    """Карточка одного датасета."""
    user_msg = r["user_message_stats"]
    asst_msg = r["assistant_message_stats"]
    user_words = user_msg.get("n_words", {}).get("mean", 0)
    asst_words = asst_msg.get("n_words", {}).get("mean", 0)
    asst_actions = asst_msg.get("n_actions", {}).get("mean", 0)
    asst_emoji = asst_msg.get("n_emoji", {}).get("mean", 0)
    asst_nsfw = asst_msg.get("n_nsfw", {}).get("mean", 0)
    nsfw_color = "#E24B4A" if asst_nsfw > 0.5 else "#1D9E75"

    lang_dist = asst_msg.get("lang_distribution", {})
    lang_str = ", ".join(f"{k}: {v*100:.0f}%" for k, v in lang_dist.items() if v > 0.01)

    top_words = r.get("vocabulary_assistant", {}).get("top_50", [])[:20]
    top_str = ", ".join(f"<code>{w}</code>" for w, _ in top_words)

    tokens = r["tokens_per_dialogue"]
    turns = r["turns_per_dialogue"]

    return f"""
<div class="dataset-card">
  <h3>{r['name']}</h3>
  <div class="meta">{r['n_dialogues']} диалогов · {r['n_user_messages']} user / {r['n_assistant_messages']} assistant</div>
  
  <div class="grid">
    <div class="stat-block">
      <div class="stat-label">Tokens/dialogue</div>
      <div class="stat-value">{tokens['median']:.0f}</div>
      <div class="stat-detail">p25={tokens['p25']:.0f} · p75={tokens['p75']:.0f} · max={tokens['max']}</div>
    </div>
    <div class="stat-block">
      <div class="stat-label">Turns/dialogue</div>
      <div class="stat-value">{turns['median']:.0f}</div>
      <div class="stat-detail">p25={turns['p25']:.0f} · p75={turns['p75']:.0f}</div>
    </div>
    <div class="stat-block">
      <div class="stat-label">Avg words / message</div>
      <div class="stat-value">{asst_words:.0f}</div>
      <div class="stat-detail">user: {user_words:.0f} · asst: {asst_words:.0f}</div>
    </div>
    <div class="stat-block">
      <div class="stat-label">*Action* RP markers (avg/msg)</div>
      <div class="stat-value">{asst_actions:.2f}</div>
    </div>
    <div class="stat-block">
      <div class="stat-label">Emoji (avg/msg)</div>
      <div class="stat-value">{asst_emoji:.2f}</div>
    </div>
    <div class="stat-block">
      <div class="stat-label">NSFW words (avg/msg)</div>
      <div class="stat-value" style="color:{nsfw_color}">{asst_nsfw:.2f}</div>
    </div>
  </div>
  
  <div class="lang">Languages: {lang_str}</div>
  <div class="vocab"><strong>Top assistant words:</strong> {top_str}</div>
  <div class="ttr">TTR (richness): user={r['vocabulary_user'].get('ttr', 0):.3f} · assistant={r['vocabulary_assistant'].get('ttr', 0):.3f}</div>
</div>"""


def generate_html(results):
    cards_html = "".join(make_dataset_card(r) for r in results)

    summary_table = """
    <table>
      <tr><th>Dataset</th><th>Dialogues</th><th>Tokens (med)</th><th>Turns (med)</th>
          <th>Asst words/msg</th><th>*Action*</th><th>Emoji</th><th>NSFW</th><th>Languages</th></tr>
    """
    for r in results:
        ums = r["user_message_stats"]
        ams = r["assistant_message_stats"]
        nsfw = ams.get("n_nsfw", {}).get("mean", 0)
        nsfw_color = "color:#E24B4A" if nsfw > 0.5 else ""
        langs = ams.get("lang_distribution", {})
        lang_str = ", ".join(f"{k}: {v*100:.0f}%" for k, v in langs.items() if v > 0.05)
        summary_table += f"""
        <tr>
          <td><strong>{r['name']}</strong></td>
          <td>{r['n_dialogues']}</td>
          <td>{r['tokens_per_dialogue']['median']:.0f}</td>
          <td>{r['turns_per_dialogue']['median']:.0f}</td>
          <td>{ams.get('n_words', {}).get('mean', 0):.0f}</td>
          <td>{ams.get('n_actions', {}).get('mean', 0):.2f}</td>
          <td>{ams.get('n_emoji', {}).get('mean', 0):.2f}</td>
          <td style="{nsfw_color}">{nsfw:.2f}</td>
          <td>{lang_str}</td>
        </tr>"""
    summary_table += "</table>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Dataset Analytics</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, sans-serif; background:#faf9f6; color:#2c2c2a;
       padding:2rem; line-height:1.5; max-width:1200px; margin:0 auto; }}
h1 {{ font-size:28px; font-weight:600; margin-bottom:8px; }}
h2 {{ font-size:20px; margin:2rem 0 1rem; padding-bottom:8px; border-bottom:1px solid #e5e3dd; }}
h3 {{ font-size:18px; margin-bottom:6px; }}
h4 {{ font-size:14px; margin:1rem 0 0.4rem; color:#5f5e5a; }}
.meta {{ font-size:13px; color:#888; margin-bottom:1rem; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; margin:0.5rem 0; }}
th {{ background:#f1efe8; padding:8px; text-align:left; font-weight:500; font-size:11px;
     text-transform:uppercase; }}
td {{ padding:8px; border-bottom:0.5px solid #e5e3dd; }}
.dataset-card {{ background:#fff; border-radius:10px; padding:1.5rem; margin:1rem 0;
                 border:0.5px solid #e5e3dd; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
         gap:12px; margin:1rem 0; }}
.stat-block {{ background:#f8f7f4; padding:12px; border-radius:8px; }}
.stat-label {{ font-size:11px; color:#888; text-transform:uppercase; letter-spacing:0.5px; }}
.stat-value {{ font-size:22px; font-weight:500; margin:4px 0; }}
.stat-detail {{ font-size:11px; color:#888; }}
.lang {{ font-size:13px; margin:8px 0; padding:6px 10px; background:#e8f0fb;
        border-radius:6px; display:inline-block; }}
.vocab {{ font-size:13px; margin:8px 0; }}
.vocab code {{ background:#f0eee8; padding:1px 5px; border-radius:3px;
              margin-right:3px; font-size:12px; }}
.ttr {{ font-size:13px; margin:8px 0; color:#5f5e5a; }}
.warn {{ background:#fff4e0; padding:10px 14px; border-radius:8px; margin:1rem 0;
        border-left:3px solid #BA7517; font-size:13px; }}
</style></head><body>

<h1>Dataset Analytics</h1>
<div class="meta">Анализ {sum(r['n_dialogues'] for r in results)} диалогов из {len(results)} датасетов</div>

<div class="warn">
<strong>Зачем это:</strong> понять что в каждом датасете — длины, словарь, стиль, язык, NSFW.
Это нужно для (1) обоснования пропорций микса, (2) ablation studies (исключающего исследования),
(3) предсказания на какой стиль будет смещаться модель.
</div>

<h2>Сравнительная таблица</h2>
{summary_table}

<h2>Детально по каждому датасету</h2>
{cards_html}

<h2>Что делать с этими данными</h2>
<div class="warn">
<strong>Признаки которые надо учесть:</strong>
<ul style="margin-left:1.2rem;margin-top:6px">
  <li>Если в датасете много NSFW при SFW-цели — добавь жёсткий фильтр</li>
  <li>Если средняя длина сильно различается — после микса модель будет копировать самый длинный</li>
  <li>Если язык не совпадает с целевым — поменяй пропорции</li>
  <li>Высокий *action* rate в assistant = narrative RP стиль (не casual)</li>
</ul>
</div>

</body></html>"""

    with open(OUTPUT_DIR / "dataset_analytics.html", "w", encoding="utf-8") as f:
        f.write(html)


def main():
    print("═" * 60)
    print(f"  DATASET ANALYTICS (sample {SAMPLE_SIZE} from each)")
    print("═" * 60)

    parsers = {
        "PIPPA": parse_pippa,
        "LimaRP": parse_limarp,
        "saiga": parse_saiga,
        "Claude_multiround": parse_claude,
    }

    results = []
    for name, parser in parsers.items():
        print(f"\n[{name}]")
        try:
            dialogues = parser(SAMPLE_SIZE)
            if not dialogues:
                print(f"  ⚠ Пропускаем — нет данных")
                continue
            stats = analyze_dataset(name, dialogues)
            results.append(stats)
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            continue

    with open(OUTPUT_DIR / "dataset_analytics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    generate_html(results)

    print("\n" + "═" * 60)
    print("  СВОДКА")
    print("═" * 60)
    print(f"{'Dataset':<20} {'Tokens':>8} {'Turns':>6} {'Words':>6} {'NSFW':>6}")
    print("─" * 60)
    for r in results:
        ams = r["assistant_message_stats"]
        print(f"{r['name']:<20} "
              f"{r['tokens_per_dialogue']['median']:>8.0f} "
              f"{r['turns_per_dialogue']['median']:>6.0f} "
              f"{ams.get('n_words', {}).get('mean', 0):>6.0f} "
              f"{ams.get('n_nsfw', {}).get('mean', 0):>6.2f}")

    print(f"\n✓ JSON: results/dataset_analytics.json")
    print(f"✓ HTML: results/dataset_analytics.html")


if __name__ == "__main__":
    main()
