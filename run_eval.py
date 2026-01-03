# run_eval.py

import os
from gemini_client import init_gemini_client
from prompt import build_genre_prompt, build_vocal_style_prompt
from pathlib import Path
from evaluate import evaluate_folder, evaluate_style_score_folder
from task import run_audio_task
from prompt_loader import load_extra_genre_prompts, get_extra_genre_prompt

# MODEL_NAME = "gemini-2.5-flash"
MODEL_NAME = "gemini-2.5-flash-lite"
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
    # wav_path = "samples_for_gemini/suno_visinger2/rock/rock_alternative-rock_suno_000_07.wav"
    # wav_path = "samples_rock_train/test/wav/rock_punk-rock_suno_000_04.wav"
    # pop
    wav_path = "samples_for_gemini/suno_visinger2/pop/pop_ballad-pop_suno_010_02.wav"

    GENRE = os.path.basename(os.path.dirname(wav_path))
    print("评测GENRE：", GENRE)

    client = init_gemini_client()
    # prompt = build_vocal_style_prompt(genre="rock")
    # 附加参数
    # extra_genre_prompt=(
    #     "The target rock style mainly refers to punk rock and alternative rock. "
    #     "Vocals may sound aggressive, strained, raspy, noisy, or intentionally distorted. "
    #     "Such characteristics are typical and should be considered positively when scoring."
    # )
    extra_prompt_map = load_extra_genre_prompts(
        "genre_extra_prompts.json"
    )

    extra_genre_prompt = get_extra_genre_prompt(
        GENRE, extra_prompt_map
    )

    prompt = build_vocal_style_prompt(genre=GENRE, extra_genre_prompt=extra_genre_prompt)
    print("Prompt:\n", prompt)

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
    extra_genre_prompt: str | None = None,
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
    prompt = build_vocal_style_prompt(genre=genre, extra_genre_prompt=extra_genre_prompt)
    print("Prompt:\n", prompt)

    mean_score = evaluate_style_score_folder(
        wav_dir=str(wav_dir),
        client=client,
        model_name=MODEL_NAME,
        prompt=prompt,
        output_jsonl=str(output_jsonl),
    )

    return mean_score

def eval_genre_folder():
    # 文件夹评测
    genre = "pop"
    print("评测：", genre)

    wav_dir = f"samples_for_gemini/suno_visinger2/{genre}"
    # wav_dir = "samples_rock_train/suno_visinger2/rock"

    extra_prompt_map = load_extra_genre_prompts(
        "genre_extra_prompts.json"
    )

    extra_genre_prompt = get_extra_genre_prompt(
        genre, extra_prompt_map
    )

    prompt = build_vocal_style_prompt(genre=genre, extra_genre_prompt=extra_genre_prompt)
    print("Prompt:\n", prompt)

    run_style_score_folder(
        wav_dir=wav_dir,
        genre=genre,
        extra_genre_prompt=extra_genre_prompt,
    )

if __name__ == "__main__":
    # main()
    # run_single_evaluation()

    eval_genre_folder()