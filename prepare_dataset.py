                      
"""
Подготовка SFW RP-датасета.
Микс: 35% PIPPA + 25% LimaRP + 25% saiga (RU) + 15% claude_multiround
Контекст: до 6144 токенов
"""

import json
import random
import re
from pathlib import Path
from collections import Counter
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from hf_dataset_utils import load_hf_train, load_pippa_train_split

random.seed(42)

TARGET_SAMPLES = 50000
MIX_RATIOS = {"pippa": 0.35, "limarp": 0.25, "saiga": 0.25, "claude": 0.15}
MIN_TURNS, MAX_TURNS = 4, 30
MIN_TOKENS, MAX_TOKENS = 100, 6144
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
TOKENIZER_NAME = "Qwen/Qwen2.5-32B-Instruct"

print(f"Загружаем токенизатор {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)


def to_chatml(messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def n_tokens(text):
    return len(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"])

def n_turns(messages):
    return sum(1 for m in messages if m.get("role") in ("user", "assistant"))

def clean(text):
    if not text:
        return ""
    text = re.sub(r'\(\(\s*OOC[:\s].*?\)\)', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\[\s*OOC[:\s].*?\]', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\bBUMP\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

NSFW_RE = re.compile(
    r'\b(cock|pussy|cum|fuck|sex|penis|vagina|nipple|orgasm|masturbat|naked|nude|anal|blowjob)\b',
    re.IGNORECASE
)
def is_sfw(text, max_matches=2):
    return len(NSFW_RE.findall(text or "")) <= max_matches


def load_pippa(target_n):
    print(f"\n[1/4] PIPPA SFW (target: {target_n})")
    ds = load_pippa_train_split()
    print(f"  Сырых: {len(ds)}")

    samples, nsfw_skipped = [], 0
    for ex in ds:
        persona = ex.get("bot_description", "") or ex.get("bot_persona", "")
        conv = ex.get("conversation", [])
        if not persona or not conv:
            continue

        full = " ".join([persona] + [t.get("message", "") for t in conv])
        if not is_sfw(full):
            nsfw_skipped += 1
            continue

        bot_name = ex.get("bot_name", "Character")
        SYS = (f"You are {bot_name}. {persona}\n\n"
               "Stay in character. Respond naturally, match the user's energy.")
        msgs = [{"role": "system", "content": SYS}]
        for turn in conv:
            content = clean(turn.get("message", ""))
            if not content or len(content) < 20:
                continue
            msgs.append({"role": "user" if turn.get("is_human") else "assistant",
                         "content": content})

        if not (MIN_TURNS <= n_turns(msgs) <= MAX_TURNS):
            continue
        text = to_chatml(msgs)
        nt = n_tokens(text)
        if not (MIN_TOKENS <= nt <= MAX_TOKENS):
            continue
        samples.append({"text": text, "source": "pippa", "tokens": nt})
        if len(samples) >= target_n * 2:
            break

    random.shuffle(samples)
    samples = samples[:target_n]
    print(f"  NSFW отфильтровано: {nsfw_skipped}, финал: {len(samples)}")
    return samples


def load_limarp(target_n):
    print(f"\n[2/4] LimaRP (target: {target_n})")
    try:
        ds = load_hf_train("lemonilia/LimaRP")
    except Exception as e:
        print(f"   Недоступен: {e}. Берём больше из PIPPA")
        return load_pippa(target_n)
    print(f"  Сырых: {len(ds)}")

    samples = []
    for ex in ds:
        convs = ex.get("conversations") or ex.get("messages") or []
        if not convs:
            continue

        msgs = []
        for c in convs:
            role = c.get("from") or c.get("role")
            content = clean(c.get("value") or c.get("content", ""))
            if not content:
                continue
            if role == "system":
                msgs.append({"role": "system", "content": content})
            elif role in ("human", "user"):
                msgs.append({"role": "user", "content": content})
            elif role in ("gpt", "assistant"):
                msgs.append({"role": "assistant", "content": content})

        if not msgs or msgs[0]["role"] != "system":
            msgs = [{"role": "system",
                     "content": "You are a creative roleplay partner. Stay in character."}] + msgs

        if not (MIN_TURNS <= n_turns(msgs) <= MAX_TURNS):
            continue
        text = to_chatml(msgs)
        nt = n_tokens(text)
        if not (MIN_TOKENS <= nt <= MAX_TOKENS):
            continue
        if not is_sfw(text):
            continue
        samples.append({"text": text, "source": "limarp", "tokens": nt})
        if len(samples) >= target_n * 2:
            break

    random.shuffle(samples)
    samples = samples[:target_n]
    print(f"  Финал: {len(samples)}")
    return samples


def load_saiga(target_n):
    print(f"\n[3/4] Saiga RU (target: {target_n})")
    ds = load_hf_train("IlyaGusev/saiga_scored")
    print(f"  Сырых: {len(ds)}")

    SYS = ("Ты дружелюбный AI-собеседник. Отвечай естественно, "
           "с эмоциями и юмором. Будь искренним.")
    samples = []
    for ex in ds:
        score = ex.get("opus_score") or ex.get("score") or 0
        if isinstance(score, (int, float)) and score < 7:
            continue
        msgs_raw = ex.get("messages", [])
        if not msgs_raw:
            continue
        msgs = []
        if msgs_raw[0].get("role") != "system":
            msgs.append({"role": "system", "content": SYS})
        for m in msgs_raw:
            role = m.get("role")
            content = m.get("content", "").strip()
            if role in ("system", "user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        if not (MIN_TURNS <= n_turns(msgs) <= MAX_TURNS):
            continue
        text = to_chatml(msgs)
        nt = n_tokens(text)
        if not (MIN_TOKENS <= nt <= MAX_TOKENS):
            continue
        samples.append({"text": text, "source": "saiga", "tokens": nt})
        if len(samples) >= target_n * 2:
            break

    random.shuffle(samples)
    samples = samples[:target_n]
    print(f"  Финал: {len(samples)}")
    return samples


def load_claude(target_n):
    print(f"\n[4/4] Claude multiround (target: {target_n})")
    ds = load_hf_train("Norquinal/claude_multiround_chat_30k")
    print(f"  Сырых: {len(ds)}")

    SYS = "You are a helpful, knowledgeable, friendly assistant."
    role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

    samples = []
    for ex in ds:
        convs = ex.get("conversations", [])
        if not convs:
            continue
        msgs = [{"role": "system", "content": SYS}]
        for c in convs:
            role = role_map.get(c.get("from", ""))
            content = c.get("value", "").strip()
            if role and content:
                msgs.append({"role": role, "content": content})
        if not (MIN_TURNS <= n_turns(msgs) <= MAX_TURNS):
            continue
        text = to_chatml(msgs)
        nt = n_tokens(text)
        if not (MIN_TOKENS <= nt <= MAX_TOKENS):
            continue
        samples.append({"text": text, "source": "claude", "tokens": nt})
        if len(samples) >= target_n * 2:
            break

    random.shuffle(samples)
    samples = samples[:target_n]
    print(f"  Финал: {len(samples)}")
    return samples


def main():
    print("" * 60)
    print(f"  ПОДГОТОВКА SFW RP ДАТАСЕТА (target: {TARGET_SAMPLES})")
    print("" * 60)

    targets = {k: int(TARGET_SAMPLES * v) for k, v in MIX_RATIOS.items()}
    for k, n in targets.items():
        print(f"  {k:8s}: {n}")

    all_samples = []
    all_samples.extend(load_pippa(targets["pippa"]))
    all_samples.extend(load_limarp(targets["limarp"]))
    all_samples.extend(load_saiga(targets["saiga"]))
    all_samples.extend(load_claude(targets["claude"]))

    random.shuffle(all_samples)
    print(f"\nИтого: {len(all_samples)}")

    counts = Counter(s["source"] for s in all_samples)
    print("\nПо источникам:")
    for src, n in counts.most_common():
        print(f"  {src:8s}: {n} ({100*n/len(all_samples):.1f}%)")

    tokens = [s["tokens"] for s in all_samples]
    print(f"\nТокены: avg={sum(tokens)/len(tokens):.0f}, max={max(tokens)}")

    n_eval = max(500, int(len(all_samples) * 0.05))
    eval_samples = all_samples[:n_eval]
    train_samples = all_samples[n_eval:]
    print(f"\nTrain: {len(train_samples)}, Eval: {len(eval_samples)}")

    Dataset.from_list([{"text": s["text"]} for s in train_samples]).save_to_disk(str(OUTPUT_DIR / "train"))
    Dataset.from_list([{"text": s["text"]} for s in eval_samples]).save_to_disk(str(OUTPUT_DIR / "eval"))

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump({
            "total": len(all_samples),
            "train": len(train_samples),
            "eval": len(eval_samples),
            "sources": dict(counts),
            "avg_tokens": sum(tokens) / len(tokens),
            "max_tokens": max(tokens),
            "tokenizer": TOKENIZER_NAME,
        }, f, indent=2)

    print("\n Готово")


if __name__ == "__main__":
    main()
