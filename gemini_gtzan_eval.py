import os
import json
from google import genai
from tqdm import tqdm
import time
from google.genai.errors import ServerError

# ====== 配置 ======
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"

# ====== 初始化 Gemini 客户端 ======
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = (
    "Classify the given audio into one of these 10 music genres: "
    "blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock. "
    "Only output the single most likely genre name, nothing else. "
    "Output format: return only the genre tag as a single lowercase word."
)

# ====== 分类任务函数 ======
# def classify_genre_task(wav_path, true_genre, output_jsonl):
#     key = os.path.splitext(os.path.basename(wav_path))[0]

#     # 上传音频并调用 Gemini API
#     uploaded = client.files.upload(file=wav_path)
#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=[PROMPT, uploaded]
#     )
#     pred = response.text.strip().lower()

#     result = {"key": key, "true": true_genre, "pred": pred}

#     # 将结果立即写入 JSONL
#     with open(output_jsonl, "a", encoding="utf-8") as f:
#         f.write(json.dumps(result, ensure_ascii=False) + "\n")

#     return result

# 加入失败重试机制
def classify_genre_task(wav_path, true_genre, output_jsonl, retries=3, wait_sec=30):
    key = os.path.splitext(os.path.basename(wav_path))[0]

    for attempt in range(retries):
        try:
            uploaded = client.files.upload(file=wav_path)
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[PROMPT, uploaded]
            )
            pred = response.text.strip().lower()
            result = {"key": key, "true": true_genre, "pred": pred}

            # 写入 JSONL
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result

        except ServerError as e:
            print(f"Server overloaded (attempt {attempt+1}/{retries}). Retrying in {wait_sec}s...")
            time.sleep(wait_sec)

    # 超过重试次数仍失败
    print(f"Failed to classify {key} after {retries} attempts.")
    return {"key": key, "true": true_genre, "pred": "error"}

# ====== 主评估函数 ======
def evaluate_folder(wav_dir, output_jsonl="predictions.jsonl"):
    # 读取已有预测结果
    existing_results = {}
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_results[data["key"]] = data
                except:
                    continue

    # 构建任务列表，跳过已存在的 key
    tasks = []
    for file in os.listdir(wav_dir):
        if not file.endswith(".wav"):
            continue
        key = os.path.splitext(file)[0]
        if key in existing_results:
            continue
        true_genre = key.split(".")[0]
        wav_path = os.path.join(wav_dir, file)
        tasks.append((wav_path, true_genre))

    # 顺序调用
    correct = 0
    for wav_path, true_genre in tqdm(tasks, desc="Classifying"):
        res = classify_genre_task(wav_path, true_genre, output_jsonl)
        if res["pred"] == res["true"]:
            correct += 1

    # 统计总准确率（包括已有结果）
    total_preds = len(existing_results) + len(tasks)
    total_correct = sum(1 for r in existing_results.values() if r["pred"] == r["true"]) + correct

    acc = total_correct / total_preds if total_preds > 0 else 0.0
    print(f"\nTotal: {total_preds}, Correct: {total_correct}, Accuracy: {acc:.2%}")

    return acc

if __name__ == "__main__":
    wav_folder = "./gtzan_test"  # 你的测试集目录
    output_path = "gemini_pro_predictions.jsonl"
    evaluate_folder(wav_folder, output_path)
