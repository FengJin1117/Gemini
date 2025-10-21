import os
import json
import time
from tqdm import tqdm
from google import genai
from google.genai.errors import ServerError
import random

# ====== 配置 ======
os.environ["GEMINI_API_KEY"] = "AIzaSyBQwR8zcbYmhPvulYfmEp6b_zbLPn_Bz7Q"  # 改成你的 API Key

MODEL_NAME = "gemini-2.5-flash"  # 可自定义模型
# MODEL_NAME = "gemini-2.5-pro"  # 可自定义模型

GENRES = [
    "Schlager",
    "Skiffle (Revival)",
    "Brill Building Pop & Crooners",
    "Bubblegum & Teenybop",
    "Country Pop & Country Rock",
    "Singer/Songwriter",
    "(Early) Pop Rock & Power Pop",
    "Soft Rock / Adult Contemporary (A.C.)",
    "Synthpop & New Romantics",
    "Hi-NRG / Eurodisco"
]
GENRE_LIST = "\n".join([f"{i+1}. {genre}" for i, genre in enumerate(GENRES)])

PROMPT = (
    "You are a music genre classifier working within the Pop (Popular Music) domain. "
    "You will be given a 10-second audio clip.\n\n"
    "Classify it into one of the following 10 subgenres of Pop music:\n"
    f"{GENRE_LIST}\n\n"
    "Output only the number (1–10) corresponding to the most likely genre. "
    "Do not output any text, explanation, or punctuation—only the number."
)

# ====== 初始化 Gemini 客户端 ======
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
print(f"🎵 Using Gemini model: {MODEL_NAME}\n")


# ====== 单条预测函数（带重试） ======
def classify_genre_task(audio_path, true_genre_id, output_jsonl, retries=5, wait_sec=30):
    key = os.path.basename(audio_path)

    for attempt in range(retries):
        try:
            uploaded = client.files.upload(file=audio_path)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[PROMPT, uploaded]
            )
            pred = response.text.strip()

            result = {"music": key, "true": str(true_genre_id), "pred": pred, "model": MODEL_NAME}

            # 写入 JSONL 文件（追加模式）
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            return result

        except ServerError as e:
            print(f"⚠️ Server error (attempt {attempt+1}/{retries}): {e}. Retrying in {wait_sec}s...")
            time.sleep(wait_sec)

        except Exception as e:
            print(f"❌ Error processing {key}: {e}")
            time.sleep(wait_sec)

    # 超过重试次数仍失败，记录为 error
    result = {"music": key, "true": str(true_genre_id), "pred": "error", "model": MODEL_NAME}
    with open(output_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"❌ Failed after {retries} attempts: {key}")
    return result


# ====== 主评测函数（支持断点续传 + 剩余进度条） ======
def evaluate_from_jsonl(pop_jsonl, audio_folder, output_jsonl="gemini_predictions.jsonl"):
    # 读取测试集信息
    with open(pop_jsonl, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # 读取已完成预测，避免重复
    done_keys = set()
    correct_keys = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    done_keys.add(result["music"])
                    if result["pred"] == result["true"]:
                        correct_keys.add(result["music"])
                except:
                    continue

    # 筛选剩余未处理的文件
    remaining_data = [item for item in data if item["music"] not in done_keys]

    total, correct = len(remaining_data), len(correct_keys)

    # 遍历剩余文件
    for item in tqdm(remaining_data, desc="Classifying", unit="file"):
        music = item["music"]
        genre_id = item["genre_id"]

        audio_path = os.path.join(audio_folder, music)
        if not os.path.exists(audio_path):
            print(f"⚠️ Missing file: {audio_path}")
            continue

        res = classify_genre_task(audio_path, genre_id, output_jsonl, wait_sec=30)

        if res["pred"] == str(genre_id):
            correct += 1

        # 避免过快调用 API
        time.sleep(5 + random.random() * 2)

    acc = correct / len(data) if len(data) > 0 else 0.0
    print(f"\n✅ Evaluation complete.")
    print(f"Total: {len(data)}, Correct: {correct}, Accuracy: {acc:.2%}")

    return acc


if __name__ == "__main__":
    base_dir = r"D:\Music_projects\Gemini"
    pop_jsonl = os.path.join(base_dir, "pop_test.jsonl")
    audio_folder = os.path.join(base_dir, "pop_test")
    output_path = os.path.join(base_dir, f"pop_predictions_by_{MODEL_NAME}.jsonl")

    evaluate_from_jsonl(pop_jsonl, audio_folder, output_path)
