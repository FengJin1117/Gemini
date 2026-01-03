# task.py

import json
import os
from gemini_client import classify_audio_with_gemini

def classify_genre_task(
    client,
    model_name: str,
    prompt: str,
    wav_path: str,
    true_genre: str,
    output_jsonl: str,
) -> dict:
    key = os.path.splitext(os.path.basename(wav_path))[0]

    pred = classify_audio_with_gemini(
        client=client,
        model_name=model_name,
        prompt=prompt,
        wav_path=wav_path,
    )

    result = {
        "key": key,
        "true": true_genre,
        "pred": pred,
    }

    with open(output_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result

def run_audio_task(
    client,
    model_name: str,
    prompt: str,
    wav_path: str,
    output_jsonl: str | None = None,
    true_label: str | None = None,
    task_type: str = "classification",  # "classification" or "score"
) -> dict:
    """
    Run a single audio task (genre classification or vocal-style scoring).

    Args:
        client: Gemini client
        model_name: Gemini model name
        prompt: task-specific prompt
        wav_path: path to audio file
        output_jsonl: if provided, append result to this jsonl file
        true_label: ground-truth label (optional, mainly for classification)
        task_type: "classification" or "score"

    Returns:
        result dict
    """

    key = os.path.splitext(os.path.basename(wav_path))[0]

    raw_output = classify_audio_with_gemini(
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

    # -------- score task (e.g., vocal style) --------
    elif task_type == "score":
        try:
            score = int(raw_output)
        except Exception:
            score = -1  # invalid / parse error

        result = {
            "key": key,
            "score": score,
        }

        if true_label is not None:
            result["true"] = true_label

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # -------- optional write to jsonl --------
    if output_jsonl is not None:
        with open(output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result