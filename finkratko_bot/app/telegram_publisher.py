from __future__ import annotations

import html
import requests


def build_message(channel_name: str, title: str, video_url: str, bullets: list[str], own_channel_url: str) -> str:
    lines = [
        f"<b>Канал — {html.escape(channel_name)}</b>",
        "",
        f'<a href="{html.escape(video_url)}"><b>{html.escape(title)}</b></a>',
        "",
    ]
    for bullet in bullets:
        lines.append(f"👉 {html.escape(bullet)}")
    lines.extend(["", video_url, "", f'<a href="{html.escape(own_channel_url)}">ФинКратко</a>'])
    return "\n".join(lines)


def send_telegram(bot_token: str, chat_id: str, text: str) -> dict:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        },
        timeout=30,
    )
    print("Telegram status:", response.status_code)
    print("Telegram response:", response.text)
    response.raise_for_status()
    return response.json()
