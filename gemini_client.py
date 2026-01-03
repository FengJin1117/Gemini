# gemini_client.py

import os
import time
from google import genai
from google.genai.errors import ServerError


def init_gemini_client(api_key: str | None = None):
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    return genai.Client(api_key=api_key)


def safe_upload(client, file_path: str, retries: int = 3, sleep_sec: int = 3):
    for attempt in range(retries):
        try:
            return client.files.upload(file=file_path)
        except Exception as e:
            print(f"[Upload] failed ({attempt + 1}/{retries}): {e}")
            time.sleep(sleep_sec)
    raise RuntimeError("File upload failed after retries")


def classify_audio_with_gemini(
    client,
    model_name: str,
    prompt: str,
    wav_path: str,
    retries: int = 3,
    wait_sec: int = 30,
) -> str:
    """
    输入 wav 路径，返回预测 genre（lowercase string）
    """
    for attempt in range(retries):
        try:
            uploaded = safe_upload(client, wav_path)
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, uploaded],
            )
            return response.text.strip().lower()

        except ServerError:
            print(
                f"[Gemini] server overloaded "
                f"({attempt + 1}/{retries}), retrying in {wait_sec}s..."
            )
            time.sleep(wait_sec)

    return "error"
