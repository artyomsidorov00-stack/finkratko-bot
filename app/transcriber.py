from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi


@dataclass
class TranscriptResult:
    status: str  # ok | bad_quality | missing
    segments: list[dict]
    reason: str
    metrics: dict


class TranscriptFetcher:
    def __init__(self, settings):
        self.settings = settings

    def _normalize_items(self, raw_items: list[Any]) -> list[dict]:
        items = []
        for item in raw_items:
            if hasattr(item, 'text'):
                text = getattr(item, 'text', '')
                start = float(getattr(item, 'start', 0.0) or 0.0)
                duration = float(getattr(item, 'duration', 0.0) or 0.0)
            else:
                text = item.get('text', '')
                start = float(item.get('start', 0.0) or 0.0)
                duration = float(item.get('duration', 0.0) or 0.0)
            items.append({'text': text, 'start': start, 'end': start + duration, 'duration': duration})
        return items

    def _fetch_raw(self, video_id: str) -> list[dict] | None:
        errors = []

        # Newer API style
        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=['ru'])
            return self._normalize_items(list(fetched))
        except Exception as e:
            errors.append(e)

        # Older API style
        try:
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=['ru'])
            return self._normalize_items(data)
        except Exception as e:
            errors.append(e)

        # Try manual transcripts list as fallback
        try:
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            for transcript in transcript_list:
                if getattr(transcript, 'language_code', '') == 'ru':
                    fetched = transcript.fetch()
                    return self._normalize_items(list(fetched))
        except Exception as e:
            errors.append(e)

        return None

    @staticmethod
    def _transcript_metrics(segments: list[dict], video_duration_sec: int) -> dict:
        text = ' '.join(seg['text'] for seg in segments)
        total_chars = len(text)
        total_alpha = len(re.findall(r'[A-Za-zА-Яа-яЁё]', text))
        total_cyr = len(re.findall(r'[А-Яа-яЁё]', text))
        cyrillic_ratio = (total_cyr / total_alpha) if total_alpha else 0.0
        covered_seconds = sum(max(0.0, seg.get('duration', seg['end'] - seg['start'])) for seg in segments)
        coverage_ratio = (covered_seconds / video_duration_sec) if video_duration_sec else 0.0
        avg_words_per_segment = (sum(len(seg['text'].split()) for seg in segments) / len(segments)) if segments else 0.0
        music_count = len(re.findall(r'\[(?:музыка|music)\]', text, flags=re.IGNORECASE))
        duplicate_segments = 0
        seen = set()
        for seg in segments:
            norm = re.sub(r'\s+', ' ', seg['text'].strip().lower())
            if not norm:
                continue
            if norm in seen:
                duplicate_segments += 1
            seen.add(norm)
        duplicate_ratio = (duplicate_segments / len(segments)) if segments else 0.0
        return {
            'total_chars': total_chars,
            'coverage_ratio': round(coverage_ratio, 3),
            'cyrillic_ratio': round(cyrillic_ratio, 3),
            'avg_words_per_segment': round(avg_words_per_segment, 2),
            'music_count': music_count,
            'duplicate_ratio': round(duplicate_ratio, 3),
            'segment_count': len(segments),
        }

    def _is_quality_good(self, metrics: dict) -> tuple[bool, str]:
        if metrics['total_chars'] < self.settings.min_transcript_chars:
            return False, 'too_short'
        if metrics['coverage_ratio'] < self.settings.min_coverage_ratio:
            return False, 'low_coverage'
        if metrics['cyrillic_ratio'] < self.settings.min_cyrillic_ratio:
            return False, 'low_cyrillic_ratio'
        if metrics['avg_words_per_segment'] < self.settings.min_avg_words_per_segment:
            return False, 'segments_too_short'
        if metrics['duplicate_ratio'] > 0.20:
            return False, 'too_many_duplicates'
        return True, 'ok'

    def fetch_transcript(self, video_id: str, video_duration_sec: int, clean_fn) -> TranscriptResult:
        raw_segments = self._fetch_raw(video_id)
        if not raw_segments:
            return TranscriptResult(status='missing', segments=[], reason='transcript_not_found', metrics={})

        segments = []
        for seg in raw_segments:
            text = clean_fn(seg['text'])
            if not text:
                continue
            segments.append({
                'start': float(seg['start']),
                'end': float(seg['end']),
                'text': text,
                'duration': float(seg.get('duration', seg['end'] - seg['start'])),
            })

        metrics = self._transcript_metrics(segments, video_duration_sec)
        ok, reason = self._is_quality_good(metrics)
        return TranscriptResult(status='ok' if ok else 'bad_quality', segments=segments, reason=reason, metrics=metrics)
