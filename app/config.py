from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo


MSK = ZoneInfo("Europe/Moscow")


@dataclass
class Settings:
    youtube_api_key: str = os.environ.get("YOUTUBE_API_KEY", "")
    telegram_bot_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    own_channel_url: str = os.environ.get("OWN_CHANNEL_URL", "https://t.me/fincratko")

    channel_handles: list[str] = field(default_factory=lambda: [
        h.strip()
        for h in os.environ.get(
            "CHANNEL_HANDLES",
            "@finam_invest,@brokerru,@rbcinvest,@dengi_ne_spyat",
        ).split(",")
        if h.strip()
    ])

    force_repost: bool = os.environ.get("FORCE_REPOST", "false").lower() == "true"
    max_bullets: int = int(os.environ.get("MAX_BULLETS", "8"))
    facts_per_block: int = int(os.environ.get("FACTS_PER_BLOCK", "2"))
    min_video_minutes: int = int(os.environ.get("MIN_VIDEO_MINUTES", "6"))
    max_bullet_len: int = int(os.environ.get("MAX_BULLET_LEN", "190"))

    block_seconds: int = int(os.environ.get("BLOCK_SECONDS", "210"))
    block_max_chars: int = int(os.environ.get("BLOCK_MAX_CHARS", "2600"))

    publish_hour_start: int = int(os.environ.get("PUBLISH_HOUR_START_MSK", "8"))
    publish_hour_end: int = int(os.environ.get("PUBLISH_HOUR_END_MSK", "21"))

    # transcript quality thresholds for free mode
    min_transcript_chars: int = int(os.environ.get("MIN_TRANSCRIPT_CHARS", "800"))
    min_coverage_ratio: float = float(os.environ.get("MIN_COVERAGE_RATIO", "0.75"))
    min_cyrillic_ratio: float = float(os.environ.get("MIN_CYRILLIC_RATIO", "0.60"))
    min_avg_words_per_segment: float = float(os.environ.get("MIN_AVG_WORDS_PER_SEGMENT", "4.0"))

    base_dir: Path = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def debug_dir(self) -> Path:
        return self.base_dir / "debug_reports"

    @property
    def state_file(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def pending_file(self) -> Path:
        return self.data_dir / "pending_queue.json"

    @property
    def needs_review_file(self) -> Path:
        return self.data_dir / "needs_review.json"

    @property
    def normalization_file(self) -> Path:
        return self.data_dir / "normalization_rules.json"

    @property
    def feedback_file(self) -> Path:
        return self.data_dir / "feedback.jsonl"

    def validate(self) -> None:
        missing = []
        if not self.youtube_api_key:
            missing.append("YOUTUBE_API_KEY")
        if not self.telegram_bot_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.telegram_chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.debug_dir.mkdir(parents=True, exist_ok=True)
    return settings
