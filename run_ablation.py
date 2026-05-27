"""
Исключающее исследование: 5 экспериментов на Qwen2.5-7B, каждый раз
исключая один источник данных. После прогона — `results/ablation_report.html`.
"""

import os
import sys
import json
import math
import time
import gc
import torch
from pathlib import Path
from datetime import datetime, timedelta
from datasets import load_from_disk, Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback


SMALL_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ = 2048
SAMPLES_PER_DATASET = 8000
EPOCHS = 1
SAFE_MAX_SEQ = 1536

ABLATIONS = [
    {"name": "exp_full",      "exclude": None,       "desc": "Полный микс (baseline)"},
    {"name": "exp_no_pippa",  "exclude": "pippa",    "desc": "Без PIPPA"},
    {"name": "exp_no_limarp", "exclude": "limarp",   "desc": "Без LimaRP"},
    {"name": "exp_no_saiga",  "exclude": "saiga",    "desc": "Без saiga (русский)"},
    {"name": "exp_no_claude", "exclude": "claude",   "desc": "Без claude_multiround"},
]


def _is_recoverable_runtime_error(exc: Exception) -> bool:
    """Определяем ошибки, которые стоит перезапустить в safe-режиме."""
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cudnn frontend error" in msg
        or "no execution plans support the graph" in msg
    )


def prepare_data_with_sources():
    """Готовит датасет, помеченный источником каждого примера."""
    sources_path = Path("data/train_with_sources")
    if sources_path.exists():
        print(" Данные с метками уже готовы")
        return

    print("Готовим данные с метками источников...")
    sys.path.insert(0, str(Path(__file__).parent))
    from prepare_dataset import (
        load_pippa, load_limarp, load_saiga, load_claude,
        OUTPUT_DIR, MIX_RATIOS,
    )

    targets = {
        "pippa":  SAMPLES_PER_DATASET,
        "limarp": SAMPLES_PER_DATASET,
        "saiga":  SAMPLES_PER_DATASET,
        "claude": SAMPLES_PER_DATASET,
    }

    all_samples = []
    all_samples.extend(load_pippa(targets["pippa"]))
    all_samples.extend(load_limarp(targets["limarp"]))
    all_samples.extend(load_saiga(targets["saiga"]))
    all_samples.extend(load_claude(targets["claude"]))

    import random
    random.seed(42)
    random.shuffle(all_samples)

    ds = Dataset.from_list(all_samples)
    ds.save_to_disk(str(sources_path))
    print(f"  Сохранено: {len(ds)} примеров с метками источника")


def run_one(ablation):
    name = ablation["name"]
    exclude = ablation["exclude"]
    desc = ablation["desc"]

    exp_dir = Path("ablation") / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "name": name,
        "description": desc,
        "excluded": exclude,
        "status": "failed",
    }

    start = time.time()

    model = tokenizer = trainer = None
    try:
        full_ds = load_from_disk("data/train_with_sources")
        if exclude:
            full_ds = full_ds.filter(lambda x: x["source"] != exclude)

        splits = full_ds.train_test_split(test_size=0.05, seed=42)
        train_ds = splits["train"]
        eval_ds = splits["test"]
        result["train_samples"] = len(train_ds)
        result["eval_samples"] = len(eval_ds)
        print(f"  Данные: {len(train_ds)} train / {len(eval_ds)} eval")

        attempts = [
            {
                "label": "default",
                "max_seq": MAX_SEQ,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "packing": False,
                "dataloader_num_workers": 2,
            },
            {
                "label": "safe-fallback",
                "max_seq": SAFE_MAX_SEQ,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "packing": False,
                "dataloader_num_workers": 0,
            },
        ]

        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        last_error = None
        for idx, cfg in enumerate(attempts, start=1):
            try:
                print(
                    f"\n  Попытка {idx}/{len(attempts)}: {cfg['label']} "
                    f"(seq={cfg['max_seq']}, batch={cfg['per_device_train_batch_size']}, "
                    f"accum={cfg['gradient_accumulation_steps']}, packing={cfg['packing']})"
                )
                print(f"  Загружаем {SMALL_MODEL}...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=SMALL_MODEL,
                    max_seq_length=cfg["max_seq"],
                    load_in_4bit=True,
                )

                model = FastLanguageModel.get_peft_model(
                    model, r=32, lora_alpha=64, lora_dropout=0.05,
                    bias="none",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"],
                    use_gradient_checkpointing="unsloth",
                    random_state=42,
                )
                tokenizer = get_chat_template(tokenizer, chat_template="chatml")

                eff_batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
                steps_per_epoch = math.ceil(len(train_ds) / eff_batch)

                trainer = SFTTrainer(
                    model=model, tokenizer=tokenizer,
                    train_dataset=train_ds, eval_dataset=eval_ds,
                    dataset_text_field="text",
                    max_seq_length=cfg["max_seq"],
                    packing=cfg["packing"],
                    args=SFTConfig(
                        per_device_train_batch_size=cfg["per_device_train_batch_size"],
                        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
                        num_train_epochs=EPOCHS,
                        learning_rate=2e-4,
                        warmup_steps=max(10, int(steps_per_epoch * 0.05)),
                        weight_decay=0.01,
                        lr_scheduler_type="cosine",
                        optim="adamw_8bit",
                        bf16=bf16_ok,
                        fp16=not bf16_ok,
                        tf32=True,
                        logging_steps=20,
                        logging_dir=str(exp_dir / "logs"),
                        report_to="tensorboard",
                        eval_strategy="steps",
                        eval_steps=200,
                        save_strategy="steps",
                        save_steps=400,
                        save_total_limit=1,
                        load_best_model_at_end=True,
                        metric_for_best_model="eval_loss",
                        greater_is_better=False,
                        output_dir=str(exp_dir / "ckpt"),
                        seed=42,
                        dataloader_num_workers=cfg["dataloader_num_workers"],
                    ),
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                )

                stats = trainer.train()
                result["train_loss"] = stats.training_loss
                result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9

                eval_result = trainer.evaluate()
                result["eval_loss"] = eval_result["eval_loss"]
                result["perplexity"] = math.exp(eval_result["eval_loss"])

                model.save_pretrained(str(exp_dir / "lora"))
                tokenizer.save_pretrained(str(exp_dir / "lora"))

                log_history = trainer.state.log_history
                result["train_loss_history"] = [
                    {"step": e["step"], "loss": e["loss"]}
                    for e in log_history if "loss" in e
                ]
                result["eval_loss_history"] = [
                    {"step": e["step"], "eval_loss": e["eval_loss"]}
                    for e in log_history if "eval_loss" in e
                ]
                result["run_mode"] = cfg["label"]
                result["status"] = "success"
                result["duration_hours"] = (time.time() - start) / 3600

                print(f"   {name}: PPL={result['perplexity']:.2f}, "
                      f"VRAM={result['peak_vram_gb']:.1f}GB, "
                      f"time={result['duration_hours']:.1f}h")
                break
            except Exception as inner_e:
                last_error = inner_e
                print(f"   Попытка {cfg['label']} не удалась: {inner_e}")
                should_retry = idx < len(attempts) and _is_recoverable_runtime_error(inner_e)
                try:
                    del trainer
                except Exception:
                    pass
                try:
                    del model, tokenizer
                except Exception:
                    pass
                trainer = model = tokenizer = None
                gc.collect()
                torch.cuda.empty_cache()
                if should_retry:
                    time.sleep(10)
                    continue
                raise

        if result["status"] != "success":
            raise last_error if last_error is not None else RuntimeError("Training failed")

    except Exception as e:
        result["error"] = str(e)[:200]
        result["duration_hours"] = (time.time() - start) / 3600
        print(f"   {name}: {e}")

    finally:
        try:
            del model, tokenizer, trainer
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    with open(exp_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def generate_report(all_results):
    successful = [r for r in all_results if r["status"] == "success"]

    if not successful:
        print("Нет успешных экспериментов")
        return

    baseline = next((r for r in successful if r["name"] == "exp_full"), None)

    rows = ""
    for r in all_results:
        if r["status"] == "success":
            ppl = r["perplexity"]
            if baseline and r["name"] != "exp_full":
                delta = ppl - baseline["perplexity"]
                delta_pct = 100 * delta / baseline["perplexity"]
                if delta > 0.5:
                    impact = f'<span style="color:#1D9E75">PPL ^ {delta:+.2f} ({delta_pct:+.1f}%) — датасет ВАЖЕН</span>'
                elif delta < -0.5:
                    impact = f'<span style="color:#D85A30">PPL v {delta:+.2f} ({delta_pct:+.1f}%) — датасет МЕШАЕТ</span>'
                else:
                    impact = f'<span style="color:#888">PPL {delta:+.2f} ({delta_pct:+.1f}%) — нейтрально</span>'
            else:
                impact = "—"

            rows += f"""
            <tr>
                <td><strong>{r['name']}</strong><br><small>{r['description']}</small></td>
                <td>{r.get('train_samples', '—')}</td>
                <td>{r.get('train_loss', 0):.4f}</td>
                <td>{r.get('eval_loss', 0):.4f}</td>
                <td><strong>{r.get('perplexity', 0):.2f}</strong></td>
                <td>{r.get('peak_vram_gb', 0):.1f} GB</td>
                <td>{r.get('duration_hours', 0):.1f} ч</td>
                <td>{impact}</td>
            </tr>"""
        else:
            rows += f"""
            <tr style="opacity:0.6">
                <td><strong>{r['name']}</strong><br><small>{r['description']}</small></td>
                <td colspan="6">FAILED: {r.get('error', 'unknown')}</td>
                <td>—</td>
            </tr>"""

    chart_data = {}
    for r in successful:
        chart_data[r["name"]] = {
            "train": [(e["step"], e["loss"]) for e in r.get("train_loss_history", [])],
            "eval": [(e["step"], e["eval_loss"]) for e in r.get("eval_loss_history", [])],
        }

    if baseline:
        recs = []
        for r in successful:
            if r["name"] == "exp_full":
                continue
            delta_pct = 100 * (r["perplexity"] - baseline["perplexity"]) / baseline["perplexity"]
            ds_name = r["excluded"]
            if delta_pct > 3:
                recs.append(f"<li><strong>{ds_name}</strong> важен: без него PPL ухудшается на {delta_pct:.1f}%</li>")
            elif delta_pct < -3:
                recs.append(f"<li><strong>{ds_name}</strong> возможно вреден: без него PPL УЛУЧШАЕТСЯ на {-delta_pct:.1f}%. Рассмотри фильтрацию или удаление.</li>")
            else:
                recs.append(f"<li><strong>{ds_name}</strong> малозначим (дельта {delta_pct:+.1f}%): можно оставить или убрать без последствий</li>")
        recs_html = "<ul>" + "".join(recs) + "</ul>"
    else:
        recs_html = "Baseline (exp_full) не отработал — выводов нет."

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Ablation Study</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, sans-serif; background:#faf9f6; color:#2c2c2a;
       padding:2rem; line-height:1.6; max-width:1100px; margin:0 auto; }}
h1 {{ font-size:28px; font-weight:600; }}
h2 {{ font-size:20px; margin:2rem 0 1rem; padding-bottom:8px; border-bottom:1px solid #e5e3dd; }}
.meta {{ color:#888; margin-bottom:1.5rem; }}
table {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:13px; }}
th {{ background:#f1efe8; padding:10px; text-align:left; font-weight:500;
     font-size:12px; text-transform:uppercase; }}
td {{ padding:10px; border-bottom:0.5px solid #e5e3dd; vertical-align:top; }}
.warn {{ background:#fff4e0; padding:14px; border-radius:8px;
        border-left:3px solid #BA7517; margin:1rem 0; font-size:14px; }}
.chart {{ background:#fff; border-radius:10px; padding:1.5rem;
         border:0.5px solid #e5e3dd; margin:1rem 0; }}
ul {{ margin-left:1.2rem; }}
li {{ margin:0.4rem 0; }}
</style></head><body>

<h1>Ablation Study — исключающее исследование</h1>
<div class="meta">
  Базовая модель: Qwen2.5-7B (4-bit QLoRA, r=32) · 1 эпоха · seq=2048<br>
  Датасет: микс 4 источников по {SAMPLES_PER_DATASET} примеров<br>
  Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="warn">
exp_full — baseline.
</div>

<h2>Сравнительная таблица</h2>
<table>
  <tr>
    <th>Эксперимент</th><th>Train</th><th>Train Loss</th><th>Eval Loss</th>
    <th>PPL</th><th>VRAM</th><th>Время</th><th>Влияние датасета</th>
  </tr>
  {rows}
</table>

<h2>Кривые обучения</h2>
<div class="chart">
  <canvas id="lossChart" style="height:380px"></canvas>
</div>

<h2>Выводы и рекомендации</h2>
<div class="warn">
{recs_html}
</div>

<script>
const data = {json.dumps(chart_data, default=str)};
const colors = ['#534AB7','#1D9E75','#D85A30','#3266ad','#D4537E'];
const datasets = [];
let i = 0;
for (const [name, d] of Object.entries(data)) {{
  const c = colors[i % colors.length];
  if (d.train.length) datasets.push({{
    label: name + ' (train)', data: d.train.map(([s, l]) => ({{x: s, y: l}})),
    borderColor: c, borderDash: [4, 2], borderWidth: 1.5, pointRadius: 0, fill: false,
  }});
  if (d.eval.length) datasets.push({{
    label: name + ' (eval)', data: d.eval.map(([s, l]) => ({{x: s, y: l}})),
    borderColor: c, borderWidth: 2.5, pointRadius: 4, fill: false,
  }});
  i++;
}}
new Chart(document.getElementById('lossChart'), {{
  type: 'scatter',
  data: {{ datasets }},
  options: {{
    responsive: true, maintainAspectRatio: false, showLine: true,
    scales: {{ x: {{ title: {{display: true, text: 'Step'}} }},
              y: {{ title: {{display: true, text: 'Loss'}} }} }},
    plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 11 }} }} }} }}
  }}
}});
</script>

</body></html>"""

    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_report.html", "w", encoding="utf-8") as f:
        f.write(html)


def main():
    start = datetime.now()
    print("  ABLATION STUDY")
    print(f"  Старт: {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Экспериментов: {len(ABLATIONS)}")
    print(f"  Ожидаемое время: ~{4*len(ABLATIONS)}-{5*len(ABLATIONS)} часов")

    prepare_data_with_sources()

    all_results = []
    for i, ab in enumerate(ABLATIONS):
        print()
        print(f"[{i+1}/{len(ABLATIONS)}] {ab['name']} — {ab['desc']}")
        r = run_one(ab)
        all_results.append(r)

        Path("results").mkdir(exist_ok=True)
        with open("results/ablation_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        generate_report(all_results)

        elapsed = datetime.now() - start
        remaining = len(ABLATIONS) - i - 1
        print(f"\nПрошло: {elapsed}, осталось ~{remaining}")

    end = datetime.now()
    total = end - start
    successful = [r for r in all_results if r["status"] == "success"]
    print()
    print(f"  ABLATION ЗАВЕРШЕН")
    print(f"  Время: {total}")
    print(f"  Успешно: {len(successful)}/{len(ABLATIONS)}")
    print(f"  Отчёт: results/ablation_report.html")

    with open("ABLATION_DONE.txt", "w") as f:
        f.write(f"Done: {end}\nDuration: {total}\nSuccess: {len(successful)}/{len(ABLATIONS)}\n")


if __name__ == "__main__":
    main()
