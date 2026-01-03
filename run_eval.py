# run_eval.py

import os
from gemini_client import init_gemini_client
from prompt import build_genre_prompt, build_vocal_style_prompt
from pathlib import Path
from evaluate import evaluate_folder, evaluate_style_score_folder
from task import run_audio_task

MODEL_NAME = "gemini-2.5-flash"
# MODEL_NAME = "gemini-2.5-pro"

def main():
    wav_folder = "gemini_classification_super_clips"

    output_jsonl = os.path.join(
        wav_folder, f"genre_predictions_by_{MODEL_NAME}.jsonl"
    )

    client = init_gemini_client()
    prompt = build_genre_prompt()

    evaluate_folder(
        wav_dir=wav_folder,
        client=client,
        model_name=MODEL_NAME,
        prompt=prompt,
        output_jsonl=output_jsonl,
    )

def run_single_evaluation():
    # 常见bug: windows的\路径分隔符在Python中是转义符，导致文件找不到！！
    wav_path = "samples_for_gemini/suno_visinger2/rock/rock_alternative-rock_suno_000_07.wav"

    client = init_gemini_client()
    prompt = build_vocal_style_prompt(genre="rock")

    res = run_audio_task(
        client=client,
        model_name=MODEL_NAME,
        prompt=prompt,
        wav_path=wav_path,
        output_jsonl=None,
        task_type="score",
    )

    print(res["score"])

def run_style_score_folder(
    wav_dir: str,
    genre: str = "rock",
):
    """
    High-level API:
    - wav_dir: e.g. samples_for_gemini/suno_visinger2/rock
    - output jsonl name is derived HERE
    """

    wav_dir = Path(wav_dir)
    assert wav_dir.exists(), f"Folder not found: {wav_dir}"

    # e.g. suno_visinger2_rock.jsonl
    output_jsonl = (
        wav_dir.parent / f"{wav_dir.parent.name}_{wav_dir.name}.jsonl"
    )
    print("输出jsonl：", output_jsonl)

    client = init_gemini_client()
    prompt = build_vocal_style_prompt(genre=genre)

    mean_score = evaluate_style_score_folder(
        wav_dir=str(wav_dir),
        client=client,
        model_name=MODEL_NAME,
        prompt=prompt,
        output_jsonl=str(output_jsonl),
    )

    return mean_score


if __name__ == "__main__":
    # main()
    # run_single_evaluation()

    # 文件夹评测
    wav_dir = "samples_for_gemini/suno_visinger2/rock"
    genre = os.path.basename(wav_dir)
    print("评测：", genre)
    run_style_score_folder(
        wav_dir=wav_dir,
        genre=genre,
    )
