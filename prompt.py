# prompt.py

GENRE_DESCRIPTIONS = {
    "blues": "emotional guitar-based music with slow rhythm and soulful vocals",
    "classical": "orchestral or instrumental music with structured composition and no modern beats",
    "country": "acoustic instruments like guitar or banjo, storytelling vocals, often with rural themes",
    "electronic": "synthesized sounds, repetitive beats, and minimal or robotic vocals",
    "folk": "acoustic, narrative songs rooted in traditional culture, often simple and melodic",

    "hiphop": "rhythmic beats, rap vocals, and strong emphasis on rhythm and rhyme",
    "jazz": "improvisational music with swing rhythms, saxophone or trumpet, and complex harmonies",
    "metal": "loud, distorted guitars, aggressive drums, and powerful vocals",
    "pop": "catchy melodies, simple lyrics, and modern production, aiming for mass appeal",
    "rnb": "smooth, soulful vocals with strong groove, often blending rhythm, blues, and pop elements",

    "rock": "electric guitars, strong backbeat, and energetic or emotional singing",
    "world": (
        "music styles from different cultures, such as Arab, African, "
        "Chinese Traditional, Indian, and Latin, often using traditional "
        "instruments and regional rhythms"
    ),
    "other": "non-typical tracks such as speech, recitation, or undefined music styles",
}


def build_genre_prompt() -> str:
    genre_lines = "\n".join(
        f"- {genre}: {desc}" for genre, desc in GENRE_DESCRIPTIONS.items()
    )

    prompt = (
        "You are an expert in music genre classification. "
        "Listen to the given audio and classify it into one of these genres "
        "based on its musical style:\n"
        f"{genre_lines}\n\n"
        "Task: Classify the audio clip into the single most likely genre. "
        f"Output only the genre tag (one of: {', '.join(GENRE_DESCRIPTIONS.keys())}) "
        "in lowercase, with no explanation."
    )
    return prompt

def build_vocal_style_prompt(
    genre: str = "rock",
    extra_genre_prompt: str | None = None,
) -> str:
    genre = genre.lower()
    genre_upper = genre.upper()

    extra_block = ""
    if extra_genre_prompt:
        extra_block = extra_genre_prompt.strip() + "\n\n"

    return (
        "You are an expert in music genre analysis, specializing in vocal style.\n\n"
        "The given audio is a dry vocal recording (singing voice only), with no accompaniment.\n"
        "Do NOT consider instrumentation, rhythm section, or production elements.\n"
        "Focus ONLY on vocal style, including:\n"
        "- vocal timbre and tone\n"
        "- singing intensity and energy\n"
        "- articulation and expression\n"
        f"- stylistic traits typical of {genre} vocals\n\n"
        f"{extra_block}"
        "Task:\n"
        f"Rate how well this vocal performance matches the {genre_upper} vocal style.\n\n"
        "Scoring rules:\n"
        "1: Not matching at all\n"
        "2: Slightly matching\n"
        "3: Moderately matching\n"
        "4: Clearly matching\n"
        "5: Strongly and prototypically matching\n\n"
        "Note:\n"
        "When the vocal characteristics are highly typical of the target genre, "
        "do not hesitate to assign a higher score.\n\n"
        "Output ONLY a single integer from 1 to 5.\n"
        "Do not include explanations or extra text."
    )
