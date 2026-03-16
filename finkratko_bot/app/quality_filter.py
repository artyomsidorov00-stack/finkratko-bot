from __future__ import annotations

import re

from .normalizer import BAD_FRAGMENTS

BAD_STARTS = [
    "это", "она", "он", "они", "его", "ее", "её", "их", "там", "тут", "ну", "вот",
    "а", "и", "но", "если", "получается", "собственно", "потому", "поэтому"
]

FIRST_PERSON_NOISE = [
    "я зашел", "я вошел", "я вышел", "я зашортил", "я шортил", "я лонговал",
    "я купил", "я продал", "у меня открыт терминал", "моя позиция", "мой терминал",
    "я думаю", "мне кажется"
]

TRADER_NOISE_WORDS = [
    "шорт", "лонг", "лонговый", "зашортил", "зашаркил", "терминал", "позиция",
    "стоп", "тейк", "свеча", "трейд"
]

COMPANY_WORDS = [
    "сбер", "сбербанк", "втб", "татнефть", "татнефтью", "ростелеком", "headhunter",
    "hh", "хэдхантер", "хедхантер", "мгтс", "мтс", "новатэк", "совкомфлот",
    "фосагро", "русал", "норникель", "роснефть", "лукойл", "яндекс", "x5",
    "транснефть", "аэрофлот", "делимобиль", "сургутнефтегаз", "новобев", "втб лизинг"
]

MACRO_WORDS = [
    "бюджет", "инфляц", "ставк", "рубл", "доллар", "юан", "курс", "нефт", "газ",
    "экономик", "экспорт", "импорт", "дефицит", "доход", "расход", "цб",
    "центральный банк", "брент", "brent", "urals", "баррел", "баррель", "пхг",
    "газохранилищ", "спг", "офз"
]

METRIC_WORDS = [
    "дивиденд", "дивдоход", "дивидендная доходность", "купон", "целевая цена",
    "выручк", "чистая прибыль", "прибыл", "свободный денежный поток", "денежный поток",
    "fcf", "ebitda", "рентабельност", "долг", "чистый долг", "доходность", "баррель",
    "кубометр", "оборот", "прибыль на акцию", "цена за акцию", "рублей за акцию", "трлн",
    "млрд", "%"
]

FINANCE_WORDS = [
    "акци", "облигац", "дивид", "ставк", "инфляц", "нефт", "газ", "доллар", "рубл",
    "курс", "бюджет", "дефицит", "доход", "расход", "сбер", "втб", "офз", "ипотек",
    "недвиж", "рынок", "прибыл", "выручк", "цб", "банк", "экспорт", "импорт",
    "санкц", "дивиденд", "кредит", "жиль", "вторич", "первич", "капитал", "налог",
    "металл", "золото", "валют", "эмитент", "купон", "доходност", "марж", "прибыль",
    "рентабель", "роснефт", "лукойл", "новатэк", "газпром", "совкомфлот", "фосагро",
    "русал", "норникель", "яндекс", "мтс", "магнит", "долг", "ликвидност", "brent",
    "urals", "баррел", "баррель", "кубометр", "спг", "ростелеком", "хэдхантер", "hh",
    "мгтс", "дивдоходность", "депозит", "headhunter", "татнефть", "аэрофлот", "делимобиль",
    "транснефть"
]

PREDICATE_WORDS = [
    "составляет", "обсуждается", "ожидается", "может", "будет", "остается", "снижается",
    "растет", "вырос", "упал", "падает", "сокращается", "давит", "поддерживает",
    "укрепляется", "ослабевает", "сохраняется", "ускоряется", "замедляется", "увеличивается",
    "предполагает", "прогнозирует", "считает", "ждет", "выглядит", "переоценивается",
    "интересен", "обсуждают", "урезают", "выигрывает", "страдает", "зависит", "помогает",
    "ухудшается", "улучшается", "влияет", "компенсируется", "опережает", "отстает",
    "торгуется", "превысил", "сократился", "снизился", "выросла", "упала"
]

THESIS_WORDS = [
    "выраст", "упад", "сниз", "повыс", "жд", "ожида", "может", "прогноз", "интересн",
    "фаворит", "давление", "дисконт", "покупать", "продавать", "сокращен", "секвест",
    "сильн", "слабо", "лучше", "хуже", "сохран", "остает", "рис", "выгляд", "паден",
    "рост", "замедл", "ускор", "укреп", "ослаб", "сократ", "увелич", "снизит", "повысит",
    "поддерж", "ухудш", "улучш", "переоцен", "недооцен"
]

AMBIGUOUS_PATTERNS = [
    r"^это\b", r"^эти\b", r"^эта\b", r"^этот\b", r"^такой\b", r"^такая\b", r"^такие\b",
    r"^они\b", r"^она\b", r"^он\b", r"^если это\b", r"^если такой\b", r"^у каждой из этих\b",
    r"^в таком случае\b", r"^поэтому\b"
]


def starts_ambiguous(text: str, normalize_text) -> bool:
    low = normalize_text(text)
    return any(re.search(p, low) for p in AMBIGUOUS_PATTERNS)


def contains_trader_noise(text: str, normalize_text) -> bool:
    low = normalize_text(text)
    if any(p in low for p in FIRST_PERSON_NOISE):
        return True
    return any(w in low for w in TRADER_NOISE_WORDS) and any(x in low for x in ["я ", "у меня", "моя", "мой"])


def has_explicit_subject(text: str, title_keywords: list[str], normalizer) -> bool:
    low = normalizer.normalize_text(text)
    finance_hits = sum(1 for w in FINANCE_WORDS if w in low)
    macro_hits = sum(1 for w in MACRO_WORDS if w in low)
    entities = normalizer.detect_entities(text, title_keywords)
    return (finance_hits + macro_hits + len(entities)) >= 1


def has_predicate(text: str, normalize_text) -> bool:
    low = normalize_text(text)
    if re.search(r"\d", low):
        return True
    if any(x in low for x in ["%", "руб", "₽", "$", "€", "доллар", "евро"]):
        return True
    if any(w in low for w in PREDICATE_WORDS):
        return True
    if any(w in low for w in THESIS_WORDS):
        return True
    return False


def looks_incomplete(text: str, normalize_text) -> bool:
    low = normalize_text(text)
    if low.startswith("потому что"):
        return True
    if text.endswith(",") or text.endswith("-") or text.endswith("—"):
        return True
    if "если это произойдет" in low or "если это произойдёт" in low:
        return True
    if "эти бумаги" in low or "этих бумаг" in low:
        return True
    if "такой вариант" in low or "эта история" in low or "эта ситуация" in low:
        return True
    if "..." in text or "…" in text:
        return True
    return False


def has_metric_without_subject(text: str, title_keywords: list[str], normalizer) -> bool:
    low = normalizer.normalize_text(text)
    has_metric = any(w in low for w in METRIC_WORDS) or bool(re.search(r"\d", low))
    if not has_metric:
        return False

    entities = normalizer.detect_entities(text, title_keywords)
    macro_hits = sum(1 for w in MACRO_WORDS if w in low)
    if macro_hits >= 1 and any(w in low for w in ["ставк", "инфляц", "курс", "доллар", "рубл", "нефт", "газ", "бюджет"]):
        return False
    if entities:
        return False
    return any(w in low for w in ["дивид", "купон", "целевая цена", "выручк", "прибыл", "долг", "доходност", "рублей за акцию", "акцию"])


def has_orphan_number(text: str, title_keywords: list[str], normalizer) -> bool:
    low = normalizer.normalize_text(text)
    if not re.search(r"\d", low):
        return False
    if normalizer.detect_entities(text, title_keywords):
        return False
    if any(w in low for w in MACRO_WORDS):
        return False
    if any(w in low for w in METRIC_WORDS):
        return False
    return True


def fact_score(text: str, title_keywords: list[str], normalizer) -> float:
    low = normalizer.normalize_text(text)
    score = 0.0
    if any(x in low for x in BAD_FRAGMENTS):
        return -100
    if contains_trader_noise(text, normalizer.normalize_text):
        score -= 6
    if re.search(r"\d", low):
        score += 3.0
    if any(x in low for x in ["%", "руб", "₽", "$", "€", "доллар", "евро"]):
        score += 3.0
    for w in FINANCE_WORDS:
        if w in low:
            score += 1.0
    for w in THESIS_WORDS:
        if w in low:
            score += 0.9
    for w in title_keywords:
        if w in low:
            score += 1.2
    entities = normalizer.detect_entities(text, title_keywords)
    score += min(2.5, 0.8 * len(entities))
    words = normalizer.normalize_text(text).split()
    if words and words[0] in BAD_STARTS:
        score -= 2.5
    if starts_ambiguous(text, normalizer.normalize_text):
        score -= 2.0
    if "?" in text:
        score -= 0.8
    if len(text) > 220:
        score -= 1.0
    if "мы " in low or "у нас" in low:
        score -= 1.2
    return score


def classify_fact_topic(text: str, normalize_text) -> str:
    low = normalize_text(text)
    if any(w in low for w in ["нефт", "газ", "brent", "urals", "спг", "баррел", "кубометр", "золото"]):
        return "commodity"
    if any(w in low for w in ["офз", "облигац", "купон", "доходност", "депозит"]):
        return "bonds"
    if any(w in low for w in ["бюджет", "ставк", "инфляц", "рубл", "доллар", "курс", "цб", "экономик"]):
        return "macro"
    if any(w in low for w in COMPANY_WORDS):
        return "company"
    return "market"


def validate_fact_candidate(text: str, title_keywords: list[str], normalizer):
    reasons = []
    if not text:
        reasons.append("empty")
        return False, reasons
    if any(x in normalizer.normalize_text(text) for x in BAD_FRAGMENTS):
        reasons.append("bad_fragment")
    if contains_trader_noise(text, normalizer.normalize_text):
        reasons.append("trader_noise")
    if looks_incomplete(text, normalizer.normalize_text):
        reasons.append("incomplete")
    if starts_ambiguous(text, normalizer.normalize_text):
        reasons.append("ambiguous_start")
    if not has_explicit_subject(text, title_keywords, normalizer):
        reasons.append("no_subject")
    if not has_predicate(text, normalizer.normalize_text):
        reasons.append("no_predicate")
    if has_metric_without_subject(text, title_keywords, normalizer):
        reasons.append("metric_without_subject")
    if has_orphan_number(text, title_keywords, normalizer):
        reasons.append("orphan_number")
    return len(reasons) == 0, reasons
