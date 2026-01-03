# evaluate.py

import os
import json
from tqdm import tqdm
from task import classify_genre_task, run_audio_task


def load_existing_results(jsonl_path: str) -> dict:
    results = {}
    if not os.path.exists(jsonl_path):
        return results

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                results[data["key"]] = data
            except Exception:
                continue
    return results


def evaluate_folder(
    wav_dir: str,
    client,
    model_name: str,
    prompt: str,
    output_jsonl: str,
) -> float:
    existing_results = load_existing_results(output_jsonl)

    tasks = []
    for genre_folder in os.listdir(wav_dir):
        genre_path = os.path.join(wav_dir, genre_folder)
        if not os.path.isdir(genre_path):
            continue

        true_genre = genre_folder.lower()
        for file in os.listdir(genre_path):
            if not file.endswith((".wav", ".mp3")):
                continue

            key = os.path.splitext(file)[0]
            if key in existing_results:
                continue

            wav_path = os.path.join(genre_path, file)
            tasks.append((wav_path, true_genre))

    correct = 0
    for wav_path, true_genre in tqdm(tasks, desc="Classifying"):
        res = classify_genre_task(
            client=client,
            model_name=model_name,
            prompt=prompt,
            wav_path=wav_path,
            true_genre=true_genre,
            output_jsonl=output_jsonl,
        )
        if res["pred"] == res["true"]:
            correct += 1

    total_preds = len(existing_results) + len(tasks)
    total_correct = (
        sum(1 for r in existing_results.values() if r["pred"] == r["true"])
        + correct
    )

    acc = total_correct / total_preds if total_preds > 0 else 0.0
    print(f"\nTotal: {total_preds}, Correct: {total_correct}, Accuracy: {acc:.2%}")

    return acc

'''
jsonl 数据格式（隐含约定）

为了保证上面代码工作正常，run_audio_task 写入的 jsonl 行应至少包含：

{
  "key": "rock_alternative-rock_suno_000_07",
  "score": 4
}
'''
def evaluate_style_score_folder(
    wav_dir: str,
    client,
    model_name: str,
    prompt: str,
    output_jsonl: str,
) -> float:
    """
    Evaluate vocal-style score for all wav files in a folder.
    Supports resume from existing jsonl results.

    Returns:
        mean_score (float)
    """

    # 1. 读取已有结果（断点存续）
    existing_results = load_existing_results(output_jsonl)

    # 2. 构建待评测任务
    tasks = []
    for fname in sorted(os.listdir(wav_dir)):
        if not fname.endswith((".wav", ".mp3")):
            continue

        key = os.path.splitext(fname)[0]
        if key in existing_results:
            continue

        wav_path = os.path.join(wav_dir, fname)
        tasks.append((key, wav_path))

    # 3. 执行新的评测
    new_scores = []
    for key, wav_path in tqdm(tasks, desc="Scoring"):
        res = run_audio_task(
            client=client,
            model_name=model_name,
            prompt=prompt,
            wav_path=wav_path,
            output_jsonl=output_jsonl,
            task_type="score",
        )

        score = res.get("score", -1)
        if score > 0:
            new_scores.append(score)

    # 4. 统计所有（已有 + 新算）的 score
    all_scores = []

    for r in existing_results.values():
        s = r.get("score", -1)
        if isinstance(s, (int, float)) and s > 0:
            all_scores.append(s)

    all_scores.extend(new_scores)

    if len(all_scores) == 0:
        mean_score = 0.0
    else:
        mean_score = sum(all_scores) / len(all_scores)

    print(
        f"\nTotal files: {len(all_scores)}, "
        f"Mean vocal-style score: {mean_score:.2f}"
    )

    return mean_score