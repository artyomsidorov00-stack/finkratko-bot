from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


MSK = ZoneInfo("Europe/Moscow")


def now_msk() -> datetime:
    return datetime.now(MSK)


def is_publish_window_open(current: datetime, start_hour: int, end_hour: int) -> bool:
    return start_hour <= current.hour < end_hour


def should_publish_today(video_published_at: datetime, current: datetime, start_hour: int, end_hour: int) -> bool:
    if not is_publish_window_open(current, start_hour, end_hour):
        return False
    if video_published_at.astimezone(MSK).hour >= end_hour:
        return False
    return True
