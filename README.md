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
