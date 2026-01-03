# openai_client.py

import base64
import os
import time
from openai import OpenAI


def init_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
):
    """
    Initialize OpenAI-compatible client (for Gemini proxy / relay).
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY / GOOGLE_API_KEY is not set")

    if base_url is None:
        base_url = os.environ.get("OPENAI_BASE_URL")

    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is not set")

    return OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
    )


def classify_audio_with_openai(
    client,
    model_name: str,
    prompt: str,
    wav_path: str,
    retries: int = 3,
    wait_sec: int = 10,
) -> str:
    """
    Input wav path, return model output text.
    (Works for both classification and score tasks.)
    """
    # read audio
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": "wav",
                                },
                            },
                        ],
                    }
                ],
            )

            return response.choices[0].message.content.strip().lower()

        except Exception as e:
            print(
                f"[OpenAI Proxy] request failed "
                f"({attempt + 1}/{retries}): {e}"
            )
            time.sleep(wait_sec)

    return "error"
