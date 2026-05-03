                      
"""

  TRAINING  Qwen2.5-32B-Instruct, SFW RP


Конфиг под 60GB VRAM лимит (20GB запас):
   Контекст:  6144 (покрывает 99% RP-диалогов)
   Батч:      2  8 grad_accum = 16 effective
   LoRA:      r=64, alpha=128
   Эпохи:     3 + early stopping
   LR:        1e-4 cosine

Запуск:
    python train.py                    # с нуля
    python train.py --resume           # продолжить
    nohup python train.py > train.log 2>&1 &
"""

import os
import sys
import json
import math
import time
import argparse
import torch
from pathlib import Path
from datetime import datetime, timedelta
from datasets import load_from_disk
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback, TrainerCallback


class Config:
                                                  
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    max_seq_length = 6144
    load_in_4bit = True

                          
    lora_r = 64
    lora_alpha = 128
    lora_dropout = 0.05
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

                                      
    batch_size = 2
    gradient_accumulation = 8                           
    num_epochs = 3
    learning_rate = 1e-4
    warmup_ratio = 0.03
    weight_decay = 0.01
    lr_scheduler = "cosine"
    optim = "adamw_8bit"
    max_grad_norm = 1.0

                   
    # Disk-safe defaults for smaller VM volumes.
    eval_steps = 500
    save_steps = 1000
    save_total_limit = 2
    early_stopping_patience = 4
    early_stopping_threshold = 0.001

                                  
    vram_warn_gb = 60.0                                 
    vram_abort_gb = 75.0                                 

             
    logging_steps = 10
    logging_dir = "logs"
    report_to = "tensorboard"

          
    output_dir = "checkpoints"
    final_lora = "model_final"

            
    seed = 42
    packing = True
    gradient_checkpointing = "unsloth"


cfg = Config()


                               

class MetricsLogger(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
        self.data = {"train": [], "eval": []}
        self.path.parent.mkdir(exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.data["train"].append({
                "step": step, "loss": logs["loss"],
                "lr": logs.get("learning_rate", 0),
                "epoch": logs.get("epoch", 0),
            })
        if "eval_loss" in logs:
            self.data["eval"].append({
                "step": step, "loss": logs["eval_loss"],
                "perplexity": math.exp(min(logs["eval_loss"], 20)),
                "epoch": logs.get("epoch", 0),
            })
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


class VRAMGuard(TrainerCallback):
    """Защита от выхода за лимит VRAM."""
    def __init__(self, warn_gb, abort_gb):
        self.warn_gb = warn_gb
        self.abort_gb = abort_gb
        self.peak_gb = 0
        self.warned = False
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        peak = torch.cuda.max_memory_allocated() / 1e9
        self.peak_gb = max(self.peak_gb, peak)
        self.step_count += 1

        if peak > self.abort_gb:
            print(f"\n VRAM {peak:.1f}GB > abort {self.abort_gb}GB  ОСТАНОВКА")
            control.should_training_stop = True

        if peak > self.warn_gb and not self.warned:
            print(f"\n VRAM {peak:.1f}GB > warn {self.warn_gb}GB  близко к лимиту")
            self.warned = True

        if self.step_count % 100 == 0:
            current = torch.cuda.memory_allocated() / 1e9
            print(f"    [VRAM] now: {current:.1f}GB, peak: {self.peak_gb:.1f}GB")


class ETACallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 50 == 0:
            elapsed = time.time() - self.start_time
            done = state.global_step
            total = state.max_steps
            if total > 0:
                rate = elapsed / done
                eta = rate * (total - done)
                pct = 100 * done / total
                print(f"    [ETA] {done}/{total} ({pct:.1f}%)  осталось ~{eta/3600:.1f}ч")


                                   

def find_latest_checkpoint(output_dir):
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1]),
    )
    return str(ckpts[-1]) if ckpts else None


                          

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Старт: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Модель: {cfg.model_name}")
    print(f"  Контекст: {cfg.max_seq_length}, batch: {cfg.batch_size}{cfg.gradient_accumulation}")
    print(f"  VRAM лимит: warn={cfg.vram_warn_gb}GB, abort={cfg.vram_abort_gb}GB")
    print("=" * 60)

               
    print(f"\n[1/5] Загрузка модели...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

             
    print(f"\n[2/5] LoRA r={cfg.lora_r}...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=cfg.target_modules,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
        random_state=cfg.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

               
    print(f"\n[3/5] Загружаем датасет...")
    if not Path("data/train").exists():
        print("   data/train не найден. Запусти: python prepare_dataset.py")
        sys.exit(1)
    train_ds = load_from_disk("data/train")
    eval_ds = load_from_disk("data/eval")
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    eff_batch = cfg.batch_size * cfg.gradient_accumulation
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    print(f"  Effective batch: {eff_batch}, steps/epoch: {steps_per_epoch}, total: {total_steps}")

               
    print(f"\n[4/5] Тренер...")
    metrics_cb = MetricsLogger("results/metrics.json")
    vram_cb = VRAMGuard(cfg.vram_warn_gb, cfg.vram_abort_gb)
    eta_cb = ETACallback()

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        packing=cfg.packing,
        args=SFTConfig(
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            num_train_epochs=cfg.num_epochs,
            learning_rate=cfg.learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_scheduler,
            optim=cfg.optim,
            max_grad_norm=cfg.max_grad_norm,
            bf16=True, fp16=False, tf32=True,
            logging_steps=cfg.logging_steps,
            logging_dir=cfg.logging_dir,
            report_to=cfg.report_to,
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            output_dir=cfg.output_dir,
            seed=cfg.seed,
            data_seed=cfg.seed,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.early_stopping_patience,
                early_stopping_threshold=cfg.early_stopping_threshold,
            ),
            metrics_cb, vram_cb, eta_cb,
        ],
    )

              
    print(f"\n[5/5] СТАРТ")
    print(f"  TensorBoard: tensorboard --logdir {cfg.logging_dir}")
    print("=" * 60)

    resume_from = None
    if args.resume:
        resume_from = find_latest_checkpoint(cfg.output_dir)
        if resume_from:
            print(f"\n Resume from {resume_from}")
        else:
            print("\n Нет чекпоинта, стартуем с нуля")

    start = time.time()
    stats = trainer.train(resume_from_checkpoint=resume_from)
    duration = time.time() - start

           
    print(f"\n{'' * 60}")
    print(f"  Завершено за {timedelta(seconds=int(duration))}")
    print(f"  Train loss: {stats.training_loss:.4f}")
    print(f"  Peak VRAM:  {vram_cb.peak_gb:.1f}GB")
    print(f"{'' * 60}\n")

                    
    print("Сохраняем финальный LoRA...")
    model.save_pretrained(cfg.final_lora)
    tokenizer.save_pretrained(cfg.final_lora)

                    
    print("Финальная оценка...")
    final_eval = trainer.evaluate()
    print(f"  Eval loss: {final_eval['eval_loss']:.4f}")
    print(f"  Perplexity: {math.exp(final_eval['eval_loss']):.2f}")

             
    summary = {
        "model": cfg.model_name,
        "duration_hours": duration / 3600,
        "train_loss": stats.training_loss,
        "eval_loss": final_eval["eval_loss"],
        "perplexity": math.exp(final_eval["eval_loss"]),
        "peak_vram_gb": vram_cb.peak_gb,
        "trainable_params": trainable,
        "completed_at": datetime.now().isoformat(),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open("DONE.txt", "w") as f:
        f.write(f"Completed: {datetime.now()}\n")
        f.write(f"Duration: {timedelta(seconds=int(duration))}\n")
        f.write(f"Eval loss: {final_eval['eval_loss']:.4f}\n")
        f.write(f"PPL: {math.exp(final_eval['eval_loss']):.2f}\n")
        f.write(f"Peak VRAM: {vram_cb.peak_gb:.1f}GB\n")
        f.write(f"\nNext: python evaluate.py\n")

    print("\n Готово. Запусти: python evaluate.py")


if __name__ == "__main__":
    main()
