                      
"""
PRE-FLIGHT CHECK  Qwen2.5-32B-Instruct + лимиты ментора (60GB VRAM, 120GB RAM)

Запуск: python preflight_check.py
"""

import sys
import os
from pathlib import Path

OK = "\033[92m\033[0m"
FAIL = "\033[91m\033[0m"
WARN = "\033[93m\033[0m"


def check(name, ok, details=""):
    icon = OK if ok else FAIL
    print(f"  {icon}  {name}{(': ' + details) if details else ''}")
    return ok


def main():
    print("=" * 60)
    print("  PRE-FLIGHT CHECK")
    print("=" * 60)
    all_ok = True

         
    print("\n[1/6] GPU")
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
        all_ok &= check("CUDA доступна", gpu_ok)
        if gpu_ok:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"     GPU: {name}, VRAM: {mem:.1f}GB")
            check("A100 80GB", "A100" in name and mem > 70)
            check("bf16", torch.cuda.is_bf16_supported())
    except ImportError:
        print(f"  {FAIL}  PyTorch не установлен")
        all_ok = False

         
    print("\n[2/6] Системные ресурсы")
    try:
        import psutil
        ram = psutil.virtual_memory().total / 1e9
        check(f"RAM  120GB (для лимита ментора 120-140GB)",
              ram >= 120, f"{ram:.0f}GB")
        check(f"CPU  16", psutil.cpu_count() >= 16, f"{psutil.cpu_count()} ядер")
    except ImportError:
        pass

    stat = os.statvfs(Path.home())
    free = (stat.f_bavail * stat.f_frsize) / 1e9
    all_ok &= check("Свободно на диске  500GB (модель + кеш + чекпоинты)",
                    free > 500, f"{free:.0f}GB")

        
    print("\n[3/6] HuggingFace")
    try:
        from huggingface_hub import HfApi, whoami
        info = whoami()
        check("Залогинен", True, f"user: {info.get('name', '?')}")

        api = HfApi()
        for model in ["Qwen/Qwen2.5-32B-Instruct"]:
            try:
                api.model_info(model)
                check(f"Модель {model}", True)
            except Exception as e:
                print(f"  {FAIL}  {model}: {str(e)[:60]}")
                all_ok = False

        for ds in [
            "PygmalionAI/PIPPA",
            "lemonilia/LimaRP",
            "IlyaGusev/saiga_scored",
            "Norquinal/claude_multiround_chat_30k",
        ]:
            try:
                api.dataset_info(ds)
                check(f"Датасет {ds}", True)
            except Exception as e:
                msg = str(e)[:60]
                if "gated" in msg.lower():
                    print(f"  {WARN}  {ds}  gated (нужно подтверждение)")
                else:
                    print(f"  {FAIL}  {ds}: {msg}")
                    all_ok = False
    except Exception as e:
        print(f"  {FAIL}  Не залогинен: запусти `huggingface-cli login`")
        all_ok = False

                 
    print("\n[4/6] Зависимости")
    # Unsloth should be imported before transformers/trl for patching.
    deps = ["torch", "unsloth", "transformers", "trl", "peft", "bitsandbytes", "datasets",
            "accelerate", "tensorboard", "sacrebleu", "rouge_score",
            "bert_score", "numpy"]
    for pkg in deps:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            check(pkg, True, ver)
        except ImportError:
            print(f"  {FAIL}  {pkg} не установлен")
            all_ok = False
        except Exception as e:
            print(f"  {FAIL}  {pkg} ошибка импорта: {e}")
            all_ok = False

                 
    print("\n[5/6] Тест загрузки Qwen2.5-32B (~10 минут, скачает ~16GB)")
    try:
        import torch
        from unsloth import FastLanguageModel
        print("     Скачиваем модель...")
        model, tok = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-32B-Instruct",
            max_seq_length=512,
            load_in_4bit=True,
        )
        peak = torch.cuda.max_memory_allocated() / 1e9
        check("Модель загружена", True, f"VRAM: {peak:.1f}GB")
        all_ok &= check("VRAM в лимите 60GB при загрузке", peak < 25,
                        f"{peak:.1f}GB (норма: ~17-20GB при load)")

                       
        FastLanguageModel.for_inference(model)
        prompt = "Привет, как дела?"
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"     Тест RU: '{resp[:60]}'")

        del model, tok
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {FAIL}  Ошибка: {e}")
        if "-lcuda" in str(e) or "cannot find -lcuda" in str(e):
            print("     Подсказка: линкер не видит libcuda.so.")
            print("     Запусти `./setup.sh` заново (он создаёт symlink в venv/lib).")
            print("     Либо вручную добавь путь драйвера в LD_LIBRARY_PATH/LIBRARY_PATH.")
        all_ok = False

           
    print("\n" + "=" * 60)
    if all_ok:
        print(f"  {OK}  Всё готово")
    else:
        print(f"  {FAIL}  Есть проблемы")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
