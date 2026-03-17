from __future__ import annotations

from googleapiclient.discovery import build


class YouTubeSource:
    def __init__(self, api_key: str):
        self.youtube = build("youtube", "v3", developerKey=api_key)

    def get_channel_info(self, handle: str) -> dict | None:
        res = self.youtube.channels().list(
            part="id,contentDetails,snippet",
            forHandle=handle.replace("@", ""),
        ).execute()
        items = res.get("items", [])
        if not items:
            return None
        item = items[0]
        return {
            "handle": handle,
            "channel_id": item["id"],
            "channel_title": item["snippet"]["title"],
            "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
        }

    @staticmethod
    def iso_duration_to_seconds(value: str) -> int:
        import re

        m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", value)
        if not m:
            return 0
        hours = int(m.group(1) or 0)
        minutes = int(m.group(2) or 0)
        seconds = int(m.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    def get_latest_video_from_uploads(self, uploads_playlist_id: str, min_video_minutes: int) -> dict | None:
        res = self.youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=8,
        ).execute()
        items = res.get("items", [])
        if not items:
            return None

        video_ids = []
        base_items = []
        for item in items:
            title = item["snippet"]["title"]
            if title in {"Deleted video", "Private video"}:
                continue
            vid = item["contentDetails"]["videoId"]
            video_ids.append(vid)
            base_items.append(
                {
                    "id": vid,
                    "title": title,
                    "published_at": item["contentDetails"].get("videoPublishedAt", ""),
                }
            )

        if not video_ids:
            return None

        details = self.youtube.videos().list(
            part="contentDetails,snippet",
            id=",".join(video_ids),
        ).execute()
        details_map = {item["id"]: item for item in details.get("items", [])}

        fallback = None
        for base in base_items:
            detail = details_map.get(base["id"])
            if not detail:
                continue
            duration_sec = self.iso_duration_to_seconds(detail["contentDetails"]["duration"])
            current = {
                "id": base["id"],
                "title": detail["snippet"]["title"],
                "published_at": base["published_at"],
                "duration_sec": duration_sec,
                "url": f"https://www.youtube.com/watch?v={base['id']}",
            }
            if fallback is None:
                fallback = current
            if duration_sec >= min_video_minutes * 60:
                return current

        return fallback
