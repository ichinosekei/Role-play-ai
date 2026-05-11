"""
Общая загрузка датасетов с Hugging Face при datasets>=4 (скрипты отключены).

Используется в prepare_dataset.py и analyze_datasets.py.
"""

from __future__ import annotations

from datasets import load_dataset
from huggingface_hub import HfApi

hf_api = HfApi()


def _recoverable_hf_load_error(msg: str) -> bool:
    m = msg.lower()
    return (
        "dataset scripts are no longer supported" in m
        or "no (supported) data files found" in m
        or "couldn't find any data file" in m
    )


def load_hf_train(repo_id: str):
    """
    Train split; при ошибках скрипта или отсутствия data files — parquet/json с Hub.
    """
    try:
        return load_dataset(repo_id, split="train")
    except Exception as e:
        if not _recoverable_hf_load_error(str(e)):
            raise

        repo_files = hf_api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        parquet_files = sorted(p for p in repo_files if p.endswith(".parquet"))
        if parquet_files:
            print(f"    HF fallback: parquet ({len(parquet_files)} files)")
            parquet_urls = [f"hf://datasets/{repo_id}/{p}" for p in parquet_files]
            return load_dataset("parquet", data_files={"train": parquet_urls}, split="train")

        json_files = sorted(
            p for p in repo_files if p.endswith(".json") or p.endswith(".jsonl")
        )
        if json_files:
            print(f"    HF fallback: json/jsonl ({len(json_files)} files)")
            json_urls = [f"hf://datasets/{repo_id}/{p}" for p in json_files]
            return load_dataset("json", data_files={"train": json_urls}, split="train")

        raise RuntimeError(
            f"{repo_id}: не удалось загрузить train — на Hub нет parquet/json/jsonl "
            "(или нужна авторизация для gated-датасета)."
        ) from e


def load_pippa_train_split():
    """
    PIPPA публикуется со скриптом + jsonl; для datasets>=4 — только явные jsonl,
    как в prepare_dataset.load_pippa (совместимая схема).
    """
    try:
        return load_dataset("PygmalionAI/PIPPA", split="train")
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" not in str(e):
            raise
        pippa_files = [
            "hf://datasets/PygmalionAI/PIPPA/pippa_deduped.jsonl",
            "hf://datasets/PygmalionAI/PIPPA/pippa.jsonl",
        ]
        last_err = e
        for file_url in pippa_files:
            try:
                ds = load_dataset("json", data_files={"train": file_url}, split="train")
                print(f"    PIPPA: scripts unsupported -> json ({file_url.rsplit('/', 1)[-1]})")
                return ds
            except Exception as err:
                last_err = err
                continue
        raise RuntimeError(
            "PIPPA: json fallback не удалось (файлы pippa_deduped.jsonl / pippa.jsonl)."
        ) from last_err
