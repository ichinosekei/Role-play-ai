# Role-play-ai

Fine-tuned LoRA (Qwen2.5-32B, Unsloth) для текстового ролевого моделирования.

Веса и токенизатор на Hugging Face Hub: [svyatsharov/Role-play-ai](https://huggingface.co/svyatsharov/Role-play-ai).

## Пример использования модели

Нужны GPU и окружение с **Unsloth**, **torch** и зависимостями проекта (как при обучении).

Подставьте идентификатор репозитория на Hub:

```python
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MODEL_REF = "svyatsharov/Role-play-ai"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_REF,
    max_seq_length=4096,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "Ты помощник для текстового ролевого моделирования."},
    {"role": "user", "content": "Привет! Ответь коротко, как настрой ролевой сессии."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(
    "cuda"
)

with torch.no_grad():
    out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.05,
        do_sample=True,
    )

reply = tokenizer.decode(
    out[0][inputs["input_ids"].shape[1] :],
    skip_special_tokens=True,
).split("<|im_end|>")[0].strip()

print(reply)
```

Базовая модель адаптера: `unsloth/qwen2.5-32b-instruct-bnb-4bit`.

## Абляция (Qwen2.5-7B)

Исключающее исследование по источникам данных (скрипт `run_ablation.py`): пять QLoRA-адаптеров на базе `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`. Метрики и HTML-отчёт после прогона — `results/ablation_results.json`, `results/ablation_report.html`.

Репозитории на Hugging Face Hub:

| Эксперимент | Репозиторий |
|-------------|-------------|
| Полный микс (baseline) | [svyatsharov/role-play-ai-ablation-7b-full](https://huggingface.co/svyatsharov/role-play-ai-ablation-7b-full) |
| Без PIPPA | [svyatsharov/role-play-ai-ablation-7b-no-pippa](https://huggingface.co/svyatsharov/role-play-ai-ablation-7b-no-pippa) |
| Без LimaRP | [svyatsharov/role-play-ai-ablation-7b-no-limarp](https://huggingface.co/svyatsharov/role-play-ai-ablation-7b-no-limarp) |
| Без saiga | [svyatsharov/role-play-ai-ablation-7b-no-saiga](https://huggingface.co/svyatsharov/role-play-ai-ablation-7b-no-saiga) |
| Без claude_multiround | [svyatsharov/role-play-ai-ablation-7b-no-claude](https://huggingface.co/svyatsharov/role-play-ai-ablation-7b-no-claude) |

### Пример: база 7B + адаптер с Hub

Подставьте нужный `ADAPTER_REF` из таблицы выше. Длина контекста при обучении абляции была до 2048 (при необходимости уменьшите, если не хватает VRAM).

```python
import torch
from peft import PeftModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

BASE_REF = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
ADAPTER_REF = "svyatsharov/role-play-ai-ablation-7b-full"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_REF,
    max_seq_length=2048,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, ADAPTER_REF)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "Ты помощник для текстового ролевого моделирования."},
    {"role": "user", "content": "Привет! Ответь коротко, как настрой ролевой сессии."},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

with torch.no_grad():
    out = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.05,
        do_sample=True,
    )

reply = tokenizer.decode(
    out[0][inputs["input_ids"].shape[1] :],
    skip_special_tokens=True,
).split("<|im_end|>")[0].strip()

print(reply)
```
