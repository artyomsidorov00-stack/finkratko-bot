from __future__ import annotations

import os
from pathlib import Path

import yt_dlp
from faster_whisper import WhisperModel


class AudioTranscriber:
    def __init__(self, model_name: str, device: str, compute_type: str, download_dir: Path, delete_after_run: bool = True):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.download_dir = download_dir
        self.delete_after_run = delete_after_run
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _build_ydl_opts(self, video_id: str, format_string: str) -> dict:
        outtmpl = os.path.join(str(self.download_dir), f"{video_id}.%(ext)s")

        ydl_opts = {
            "format": format_string,
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
        }

        cookie_file = os.environ.get("YTDLP_COOKIE_FILE", "").strip()
        if cookie_file and os.path.exists(cookie_file):
            ydl_opts["cookiefile"] = cookie_file

        return ydl_opts

    def download_audio(self, video_id: str) -> str:
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Пробуем несколько вариантов форматов по очереди.
        format_candidates = [
            "bestaudio*/bestaudio/best",
            "bestaudio/best",
            "best",
        ]

        last_error = None

        for format_string in format_candidates:
            try:
                ydl_opts = self._build_ydl_opts(video_id, format_string)

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return ydl.prepare_filename(info)

            except Exception as e:
                last_error = e
                print(f"Не удалось скачать формат {format_string}: {e}")

        raise last_error

    def transcribe(self, video_id: str, clean_fn) -> list[dict]:
        audio_path = self.download_audio(video_id)
        try:
            segments, _info = self.model.transcribe(
                audio_path,
                language="ru",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )

            result = []
            for seg in segments:
                text = clean_fn(seg.text)
                if not text or len(text.split()) < 3:
                    continue
                result.append({
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                })

            return result

        finally:
            if self.delete_after_run and os.path.exists(audio_path):
                os.remove(audio_path)
