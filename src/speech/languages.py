"""Single source of truth for the 8 EchoFlow-Q v2 languages.

Voice catalogues verified 2026-11-22:
  * Gnani timbre-v3.0 (docs.gnani.ai/api/TTS/available-voices): every voice speaks all 10 Indic languages;
    below are the page's recommended voices per language, split by gender.
  * edge-tts (edge_tts.list_voices()): pa-IN has NO neural voice, so Punjabi relies on Gnani.
"""
from __future__ import annotations

LANGUAGES: dict[str, dict] = {
    "kn": {"name": "Kannada", "bcp47": "kn-IN", "indicvoices": "kannada",
           "gnani": {"F": ["Shruthi", "Sneha", "Sunaina"], "M": ["Prakash", "Shubham"]},
           "edge": {"F": "kn-IN-SapnaNeural", "M": "kn-IN-GaganNeural"}},
    "en": {"name": "English", "bcp47": "en-IN", "librispeech": True,
           "gnani": {"F": ["Sunaina", "Priyasharm", "Sohini"], "M": ["Arko", "Mahendra"]},
           "edge": {"F": "en-IN-NeerjaNeural", "M": "en-IN-PrabhatNeural"}},
    "hi": {"name": "Hindi", "bcp47": "hi-IN", "indicvoices": "hindi",
           "gnani": {"F": ["Kalpana", "Lakshmi", "Swathi"], "M": ["Arko", "Avinash", "Devansh"]},
           "edge": {"F": "hi-IN-SwaraNeural", "M": "hi-IN-MadhurNeural"}},
    "ta": {"name": "Tamil", "bcp47": "ta-IN", "indicvoices": "tamil",
           "gnani": {"F": ["Kalpana", "Riyaa", "Tarika"], "M": ["Prakash", "Avinash", "Shubham"]},
           "edge": {"F": "ta-IN-PallaviNeural", "M": "ta-IN-ValluvarNeural"}},
    "te": {"name": "Telugu", "bcp47": "te-IN", "indicvoices": "telugu",
           "gnani": {"F": ["Lakshmi", "Swathi", "Vasudha"], "M": ["Rajeev", "Rakesh", "Vinod"]},
           "edge": {"F": "te-IN-ShrutiNeural", "M": "te-IN-MohanNeural"}},
    "ml": {"name": "Malayalam", "bcp47": "ml-IN", "indicvoices": "malayalam",
           "gnani": {"F": ["Rajni", "Tarika", "Sunaina"], "M": ["Shubham", "Nitin"]},
           "edge": {"F": "ml-IN-SobhanaNeural", "M": "ml-IN-MidhunNeural"}},
    "bn": {"name": "Bengali", "bcp47": "bn-IN", "indicvoices": "bengali",
           "gnani": {"F": ["Swathi", "Sohini", "Kalpana"], "M": ["Santosh", "Rakesh", "Arko"]},
           "edge": {"F": "bn-IN-TanishaaNeural", "M": "bn-IN-BashkarNeural"}},
    "pa": {"name": "Punjabi", "bcp47": "pa-IN", "indicvoices": "punjabi",
           "gnani": {"F": ["Neelam", "Kalpana", "Sneha"], "M": ["Vinod", "Rakesh", "Devansh"]},
           "edge": {}},
}
CODES = list(LANGUAGES)                                   # kn en hi ta te ml bn pa

# Full timbre-v3.0 catalogue (every voice speaks all 10 languages) — used after a language's
# recommended voices are exhausted, so up to 18 F / 16 M speakers each keep a distinct voice.
GNANI_ALL = {
    "F": ["Anitasingh", "Chetana", "Kalpana", "Lakshmi", "Lakshmidevi", "Neelam", "Priyasharm", "Rajni", "Rekha",
          "Riyaa", "Shivani", "Shruthi", "Sneha", "Sohini", "Sunaina", "Swathi", "Tarika", "Vasudha"],
    "M": ["Arko", "Avinash", "Devansh", "Harshad", "Mahendra", "Manjunath", "Nitin", "Prakash", "Rahulmehta",
          "Rajeev", "Rakesh", "Santosh", "Shubham", "Vikramdasu", "Vinod", "Yashraj"],
}


def gnani_voices(code: str, gender: str) -> list[str]:
    """Recommended voices for the language first, then the rest of the catalogue (no duplicates)."""
    rec = LANGUAGES[code]["gnani"][gender]
    return rec + [v for v in GNANI_ALL[gender] if v not in rec]
NAME_TO_CODE = {v["name"].lower(): k for k, v in LANGUAGES.items()}
BCP47_TO_CODE = {v["bcp47"].lower(): k for k, v in LANGUAGES.items()}
AUTO = "indic-auto"


def code_of(lang: str | None) -> str | None:
    """'Kannada' | 'kn' | 'kn-IN' -> 'kn'; None/'auto' -> None."""
    if not lang:
        return None
    s = lang.strip().lower()
    if s in ("auto", "auto-detect", AUTO) or s.startswith("auto-detect"):
        return None
    return s if s in LANGUAGES else NAME_TO_CODE.get(s) or BCP47_TO_CODE.get(s) or BCP47_TO_CODE.get(s.split("_")[0])


def name_of(code: str) -> str:
    return LANGUAGES[code]["name"]


def asr_code(source_language: str | None) -> str:
    """Source-language selector -> Gnani Prisma language_code ('indic-auto' when auto/unknown)."""
    c = code_of(source_language)
    return LANGUAGES[c]["bcp47"] if c else AUTO
