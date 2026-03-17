from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .storage import load_json

BAD_FRAGMENTS = [
    "подписывайтесь", "ставьте лайк", "ссылка в описании", "спонсор", "реклама",
    "мы рады вас приветствовать", "итоги недели", "добро пожаловать", "в новом выпуске",
    "обо всем этом расскажу", "традиционно ответим", "разыграем подарки", "поехали",
    "погнали", "приятного просмотра", "[музыка]", "музыка"
]

FILLER_PATTERNS = [
    r"\bну\b", r"\bвот\b", r"\bсобственно\b", r"\bкак бы\b", r"\bполучается\b",
    r"\bтак сказать\b", r"\bкороче\b", r"\bскажем так\b", r"\bв целом\b",
    r"\bто есть\b", r"\bпо сути\b", r"\bв принципе\b", r"\bда\b", r"\bэ\b", r"\bээ\b"
]

SLANG_WORDS = ["нахрен", "хрен", "фиг", "блин"]

TITLE_STOPWORDS = {
    "что", "как", "для", "или", "это", "все", "всё", "когда", "после", "будет",
    "могут", "может", "почему", "какие", "какой", "дальше", "россии", "рубля",
    "акции", "рост", "цен", "рынок", "вообще", "чему", "куда", "где", "ключевой",
    "ставкой", "ставка", "геополитика", "влияет", "прибыль"
}


class Normalizer:
    def __init__(self, rules_path: Path):
        self.rules = load_json(rules_path, {"replacements": {}, "regex_replacements": [], "entity_aliases": {}})

    def apply_rules(self, text: str) -> str:
        for src, dst in self.rules.get("replacements", {}).items():
            text = re.sub(rf"\b{re.escape(src)}\b", dst, text, flags=re.IGNORECASE)
        for item in self.rules.get("regex_replacements", []):
            text = re.sub(item["pattern"], item["replacement"], text, flags=re.IGNORECASE)
        return text

    def clean_segment_text(self, text: str) -> str:
        text = self.apply_rules(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zа-яё0-9\s%$₽€-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+", text)
        out = []
        for s in parts:
            s = s.strip(" -—\n\t")
            if len(s) < 40 or len(s.split()) < 6:
                continue
            out.append(s)
        return out

    def remove_fillers(self, s: str) -> str:
        out = " " + s + " "
        for p in FILLER_PATTERNS:
            out = re.sub(rf"[, ]*{p}[, ]*", " ", out, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", out).strip()

    def normalize_fact_text(self, s: str, max_bullet_len: int) -> str | None:
        s = self.remove_fillers(s)
        s = re.sub(r"^(я считаю|мне кажется|на мой взгляд|по моему мнению|как мне кажется)\s*,?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^(смотрите|короче|в общем|получается)\s*,?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^(мы хотели бы начать с одной интереснейшей новости[^.]*\.\s*)", "", s, flags=re.IGNORECASE)
        for w in SLANG_WORDS:
            s = re.sub(rf"\b{w}\b", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([А-Яа-яЁёA-Za-z0-9]+)(\s+\1\b)+", r"\1", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip(" ,;:-")
        s = self.apply_rules(s)
        if not s:
            return None
        if len(s) > max_bullet_len:
            parts = [p.strip(" ,;:-") for p in s.split(",") if p.strip()]
            short = ""
            for part in parts:
                candidate = part if not short else short + ", " + part
                if len(candidate) > max_bullet_len:
                    break
                short = candidate
            s = short.rstrip(" ,;:-") if short else s[:max_bullet_len].rstrip(" ,;:-")
        if len(s) < 35:
            return None
        return s[0].upper() + s[1:]

    def extract_title_keywords(self, title: str) -> list[str]:
        words = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", title.lower())
        words = [w for w in words if len(w) > 3 and w not in TITLE_STOPWORDS]
        return list(dict.fromkeys(words))

    def detect_entities(self, text: str, title_keywords: list[str]) -> list[str]:
        low = self.normalize_text(text)
        found = []
        for canonical, aliases in self.rules.get("entity_aliases", {}).items():
            for alias in aliases:
                if alias.lower() in low:
                    found.append(canonical)
                    break
        for word in title_keywords:
            if word in low:
                found.append(word)
        proper_nouns = re.findall(r"\b[А-ЯA-Z][а-яa-zA-Z-]{2,}\b", text)
        for pn in proper_nouns:
            if pn.lower() not in {"я", "мы", "что", "как"}:
                found.append(pn)
        return list(dict.fromkeys(found))

    def detect_main_entity(self, text: str, title_keywords: list[str]) -> str | None:
        entities = self.detect_entities(text, title_keywords)
        if not entities:
            return None
        return Counter(entities).most_common(1)[0][0]
