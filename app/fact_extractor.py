from __future__ import annotations

from .quality_filter import fact_score, validate_fact_candidate, classify_fact_topic


class FactExtractor:
    def __init__(self, normalizer, settings):
        self.normalizer = normalizer
        self.settings = settings

    def build_blocks_from_segments(self, segments: list[dict]) -> list[dict]:
        if not segments:
            return []
        blocks = []
        current = []
        current_start = segments[0]["start"]
        current_end = segments[0]["end"]
        current_chars = 0
        for seg in segments:
            seg_text = seg["text"]
            seg_len = len(seg_text)
            if not current:
                current = [seg]
                current_start = seg["start"]
                current_end = seg["end"]
                current_chars = seg_len
                continue
            duration_if_add = seg["end"] - current_start
            chars_if_add = current_chars + seg_len
            if duration_if_add > self.settings.block_seconds or chars_if_add > self.settings.block_max_chars:
                block_text = " ".join(x["text"] for x in current).strip()
                if block_text:
                    blocks.append({"start": current_start, "end": current_end, "text": block_text})
                current = [seg]
                current_start = seg["start"]
                current_end = seg["end"]
                current_chars = seg_len
            else:
                current.append(seg)
                current_end = seg["end"]
                current_chars += seg_len
        if current:
            block_text = " ".join(x["text"] for x in current).strip()
            if block_text:
                blocks.append({"start": current_start, "end": current_end, "text": block_text})
        return blocks

    def build_candidate_units(self, sentences: list[str]) -> list[dict]:
        units = []
        for i, s in enumerate(sentences):
            units.append({"text": s, "src": [i]})
            if i > 0:
                prev = sentences[i - 1]
                merged = prev.rstrip(".!?") + ". " + s
                if len(merged) <= 300:
                    low = self.normalizer.normalize_text(s)
                    if self._starts_ambiguous(s) or any(ch.isdigit() for ch in low):
                        units.append({"text": merged, "src": [i - 1, i]})
            if i > 0 and i + 1 < len(sentences):
                prev = sentences[i - 1]
                nxt = sentences[i + 1]
                if self._starts_ambiguous(s) and len(prev) + len(s) + len(nxt) < 360:
                    units.append({"text": prev.rstrip(".!?") + ". " + s + " " + nxt, "src": [i - 1, i, i + 1]})
        return units

    def _starts_ambiguous(self, s: str) -> bool:
        from .quality_filter import starts_ambiguous
        return starts_ambiguous(s, self.normalizer.normalize_text)

    def extract_facts_from_block(self, block_text: str, block_idx: int, title_keywords: list[str], reject_log: list[dict]) -> list[dict]:
        sentences = self.normalizer.split_sentences(block_text)
        if not sentences:
            return []
        units = self.build_candidate_units(sentences)
        candidates = []
        for unit in units:
            raw = unit["text"]
            cleaned = self.normalizer.normalize_fact_text(raw, self.settings.max_bullet_len)
            if not cleaned:
                reject_log.append({"block_idx": block_idx, "text": raw, "reason": ["empty_after_clean"]})
                continue
            is_ok, reasons = validate_fact_candidate(cleaned, title_keywords, self.normalizer)
            score = fact_score(cleaned, title_keywords, self.normalizer)
            if (not is_ok) or score < 4.8:
                if score < 4.8:
                    reasons = reasons + ["low_score"]
                reject_log.append({"block_idx": block_idx, "text": cleaned, "reason": reasons, "score": score})
                continue
            candidates.append({
                "block_idx": block_idx,
                "text": cleaned,
                "score": score,
                "topic": classify_fact_topic(cleaned, self.normalizer.normalize_text),
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = []
        used_topics = set()
        for c in candidates:
            if any(self._too_similar(c["text"], old["text"]) for old in selected):
                continue
            if c["topic"] not in used_topics or not selected:
                selected.append(c)
                used_topics.add(c["topic"])
            if len(selected) >= self.settings.facts_per_block:
                break
        if len(selected) < self.settings.facts_per_block:
            for c in candidates:
                if any(self._too_similar(c["text"], old["text"]) for old in selected):
                    continue
                if c in selected:
                    continue
                selected.append(c)
                if len(selected) >= self.settings.facts_per_block:
                    break
        return selected

    def _too_similar(self, a: str, b: str) -> bool:
        sa = set(self.normalizer.normalize_text(a).split())
        sb = set(self.normalizer.normalize_text(b).split())
        if not sa or not sb:
            return False
        overlap = len(sa & sb) / max(1, min(len(sa), len(sb)))
        return overlap >= 0.74

    def collect_all_facts(self, blocks: list[dict], title_keywords: list[str], reject_log: list[dict]) -> list[dict]:
        all_facts = []
        for idx, block in enumerate(blocks):
            all_facts.extend(self.extract_facts_from_block(block["text"], idx, title_keywords, reject_log))
        return all_facts

    def select_final_facts(self, all_facts: list[dict]) -> list[dict]:
        if not all_facts:
            return []
        result = []
        best_by_block = {}
        for fact in all_facts:
            block_idx = fact["block_idx"]
            if block_idx not in best_by_block or fact["score"] > best_by_block[block_idx]["score"]:
                best_by_block[block_idx] = fact
        coverage_facts = sorted(best_by_block.values(), key=lambda x: x["score"], reverse=True)
        for fact in coverage_facts:
            if any(self._too_similar(fact["text"], old["text"]) for old in result):
                continue
            result.append(fact)
            if len(result) >= self.settings.max_bullets:
                return result

        by_topic = {}
        for fact in all_facts:
            by_topic.setdefault(fact["topic"], []).append(fact)
        for topic, facts in by_topic.items():
            facts.sort(key=lambda x: x["score"], reverse=True)
            for fact in facts:
                if any(self._too_similar(fact["text"], old["text"]) for old in result):
                    continue
                result.append(fact)
                break
            if len(result) >= self.settings.max_bullets:
                return result

        for fact in sorted(all_facts, key=lambda x: x["score"], reverse=True):
            if any(self._too_similar(fact["text"], old["text"]) for old in result):
                continue
            result.append(fact)
            if len(result) >= self.settings.max_bullets:
                break
        return result[: self.settings.max_bullets]
