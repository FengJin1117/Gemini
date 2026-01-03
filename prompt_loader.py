import json
import os

def load_extra_genre_prompts(path: str) -> dict:
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_extra_genre_prompt(
    genre: str,
    prompt_dict: dict,
) -> str:
    return prompt_dict.get(genre.lower(), "")
