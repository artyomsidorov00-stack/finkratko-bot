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

    def _base_ydl_opts(self, video_id: str) -> dict:
        outtmpl = os.path.join(str(self.download_dir), f"{video_id}.%(ext)s")

        ydl_opts = {
            "outtmpl": outtmpl,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "web"]
                }
            },
        }

        cookie_file = os.environ.get("YTDLP_COOKIE_FILE", "").strip()
        if cookie_file and os.path.exists(cookie_file):
            ydl_opts["cookiefile"] = cookie_file

        return ydl_opts

    def _find_downloaded_file(self, video_id: str) -> str | None:
        possible_exts = ["mp3", "m4a", "webm", "mp4", "opus", "wav"]
        for ext in possible_exts:
            candidate = os.path.join(str(self.download_dir), f"{video_id}.{ext}")
            if os.path.exists(candidate):
                return candidate
        return None

    def download_audio(self, video_id: str) -> str:
        url = f"https://www.youtube.com/watch?v={video_id}"

        download_variants = [
            {
                "format": "ba/b",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            },
            {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            },
            {
                "format": "best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
            },
        ]

        last_error = None

        for variant in download_variants:
            try:
                ydl_opts = self._base_ydl_opts(video_id)
                ydl_opts.update(variant)

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(url, download=True)

                downloaded = self._find_downloaded_file(video_id)
                if downloaded:
                    return downloaded

            except Exception as e:
                last_error = e
                print(f"Не удалось скачать вариант {variant['format']}: {e}")

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
