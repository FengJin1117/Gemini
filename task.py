# task.py

import json
import os

from gemini_client import classify_audio_with_gemini
from openai_client import classify_audio_with_openai


def classify_audio(
    backend: str,
    client,
    model_name: str,
    prompt: str,
    wav_path: str,
):
    """
    Unified audio classification interface.
    """

    if backend == "gemini":
        return classify_audio_with_gemini(
            client=client,
            model_name=model_name,
            prompt=prompt,
            wav_path=wav_path,
        )

    elif backend == "openai":
        return classify_audio_with_openai(
            client=client,
            model_name=model_name,
            prompt=prompt,
            wav_path=wav_path,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

# 评估任务
def run_audio_task(
    client,
    model_name: str,
    prompt: str,
    wav_path: str,
    output_jsonl: str | None = None,
    true_label: str | None = None,
    task_type: str = "classification",  # "classification" or "score"
    backend: str = "gemini",
) -> dict:
    """
    Run a single audio task (genre classification or vocal-style scoring).
    """

    key = os.path.splitext(os.path.basename(wav_path))[0]

    raw_output = classify_audio(
        backend=backend,
        client=client,
        model_name=model_name,
        prompt=prompt,
        wav_path=wav_path,
    )

    # -------- classification task --------
    if task_type == "classification":
        result = {
            "key": key,
            "true": true_label,
            "pred": raw_output,
        }

    # -------- score task --------
    elif task_type == "score":
        try:
            score = int(raw_output)
        except Exception:
            score = -1

        result = {
            "key": key,
            "score": score,
        }

        if true_label is not None:
            result["true"] = true_label

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # -------- optional write --------
    if output_jsonl is not None:
        with open(output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result
