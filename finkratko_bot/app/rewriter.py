from __future__ import annotations

import torch

from .quality_filter import validate_fact_candidate, fact_score, starts_ambiguous, looks_incomplete, contains_trader_noise


class Rewriter:
    def __init__(self, model_name: str, device: str, normalizer):
        self.normalizer = normalizer
        self.device = device
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def paraphrase_ru(self, text: str) -> str:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        max_len = min(256, int(enc["input_ids"].shape[1] * 1.5 + 12))
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                encoder_no_repeat_ngram_size=4,
                num_beams=5,
                max_length=max_len,
                do_sample=False,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @staticmethod
    def lowercase_first_after_prefix(text: str) -> str:
        return text[0].lower() + text[1:] if text else text

    def ensure_entity_prefix(self, text: str, title_keywords: list[str]) -> str:
        entity = self.normalizer.detect_main_entity(text, title_keywords)
        if not entity:
            return text
        low = self.normalizer.normalize_text(text)
        if low.startswith(self.normalizer.normalize_text(entity)):
            return text
        if entity in {"HeadHunter", "Ростелеком", "Татнефть", "Сбер", "ВТБ", "МТС", "МГТС", "Аэрофлот", "Делимобиль", "Транснефть", "Сургутнефтегаз", "Brent", "Urals", "ОФЗ", "ЦБ"}:
            return f"{entity}: {self.lowercase_first_after_prefix(text)}"
        return text

    def looks_awkward(self, s: str) -> bool:
        low = self.normalizer.normalize_text(s)
        if starts_ambiguous(s, self.normalizer.normalize_text):
            return True
        if looks_incomplete(s, self.normalizer.normalize_text):
            return True
        if low.count(",") >= 3:
            return True
        if "что по" in low or "потому что" in low or "у каждой из этих" in low:
            return True
        if contains_trader_noise(s, self.normalizer.normalize_text):
            return True
        return False

    def rewrite_fact(self, text: str, title_keywords: list[str], max_bullet_len: int) -> str | None:
        base = self.normalizer.normalize_fact_text(text, max_bullet_len)
        if not base:
            return None
        base = self.ensure_entity_prefix(base, title_keywords)
        is_ok, _ = validate_fact_candidate(base, title_keywords, self.normalizer)
        if is_ok and not self.looks_awkward(base):
            return base
        try:
            para = self.paraphrase_ru(base)
            para = self.normalizer.normalize_fact_text(para, max_bullet_len)
            if para:
                para = self.ensure_entity_prefix(para, title_keywords)
            if not para:
                return base if is_ok else None
            para_ok, _ = validate_fact_candidate(para, title_keywords, self.normalizer)
            if para_ok and fact_score(para, title_keywords, self.normalizer) >= fact_score(base, title_keywords, self.normalizer) - 1.2:
                return para
            return base if is_ok else None
        except Exception:
            return base if is_ok else None
