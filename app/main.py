from __future__ import annotations

from datetime import datetime
import json

from .config import get_settings, MSK
from .youtube_source import YouTubeSource
from .transcriber import TranscriptFetcher
from .normalizer import Normalizer
from .fact_extractor import FactExtractor
from .rewriter import Rewriter
from .telegram_publisher import build_message, send_telegram
from .scheduler_rules import now_msk, is_publish_window_open
from .storage import load_json, save_json


def parse_published_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except Exception:
        return None


def save_debug_report(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _has_video(items: list[dict], video_id: str) -> bool:
    return any(x.get('video_id') == video_id for x in items)


def _queue_needs_review(needs_review: dict, channel_name: str, video: dict, reason: str, metrics: dict):
    items = needs_review.get('items', [])
    if not _has_video(items, video['id']):
        items.append({
            'video_id': video['id'],
            'channel_name': channel_name,
            'title': video['title'],
            'video_url': video['url'],
            'published_at': video.get('published_at', ''),
            'reason': reason,
            'metrics': metrics,
        })
    needs_review['items'] = items


def main() -> None:
    settings = get_settings()
    settings.validate()

    yt = YouTubeSource(settings.youtube_api_key)
    normalizer = Normalizer(settings.normalization_file)
    transcript_fetcher = TranscriptFetcher(settings)
    extractor = FactExtractor(normalizer, settings)
    rewriter = Rewriter('cointegrated/rut5-base-paraphraser', 'cpu', normalizer)

    state = load_json(settings.state_file, {'processed_video_ids': []})
    processed = set(state.get('processed_video_ids', []))
    pending = load_json(settings.pending_file, {'items': []})
    needs_review = load_json(settings.needs_review_file, {'items': []})

    current_time = now_msk()

    # 1. публикуем отложенные посты, если окно открыто
    if is_publish_window_open(current_time, settings.publish_hour_start, settings.publish_hour_end):
        still_pending = []
        for item in pending.get('items', []):
            try:
                send_telegram(settings.telegram_bot_token, settings.telegram_chat_id, item['message'])
                processed.add(item['video_id'])
            except Exception:
                still_pending.append(item)
        pending['items'] = still_pending

    for handle in settings.channel_handles:
        print('=' * 60)
        print('Канал:', handle)
        channel = yt.get_channel_info(handle)
        if not channel:
            continue

        video = yt.get_latest_video_from_uploads(channel['uploads_playlist_id'], settings.min_video_minutes)
        if not video:
            print('Видео не найдено')
            continue

        print('Видео:', video['title'])
        print('Video ID:', video['id'])

        if (not settings.force_repost) and video['id'] in processed:
            print('Это видео уже публиковали, пропускаем')
            continue
        if (not settings.force_repost) and _has_video(needs_review.get('items', []), video['id']):
            print('Это видео уже отправлено в очередь needs_review, пропускаем')
            continue

        published_at = parse_published_at(video.get('published_at', ''))
        if published_at is None:
            published_at = current_time

        transcript = transcript_fetcher.fetch_transcript(video['id'], video.get('duration_sec', 0), normalizer.clean_segment_text)
        print('Transcript status:', transcript.status)
        print('Transcript reason:', transcript.reason)
        if transcript.metrics:
            print('Transcript metrics:', transcript.metrics)

        if transcript.status != 'ok':
            _queue_needs_review(
                needs_review,
                channel['channel_title'],
                video,
                transcript.reason,
                transcript.metrics,
            )
            print('Видео отправлено в очередь needs_review')
            debug_payload = {
                'video_id': video['id'],
                'title': video['title'],
                'channel_name': channel['channel_title'],
                'transcript_status': transcript.status,
                'transcript_reason': transcript.reason,
                'transcript_metrics': transcript.metrics,
                'accepted_facts': [],
                'rejected_candidates': [],
            }
            save_debug_report(settings.debug_dir / f"{video['id']}.json", debug_payload)
            continue

        title_keywords = normalizer.extract_title_keywords(video['title'])
        blocks = extractor.build_blocks_from_segments(transcript.segments)
        reject_log = []
        all_facts = extractor.collect_all_facts(blocks, title_keywords, reject_log)
        if not all_facts:
            bullets = ['Не удалось выделить понятные ключевые мысли из выпуска.']
            accepted_debug = []
        else:
            selected = extractor.select_final_facts(all_facts)
            bullets = []
            accepted_debug = []
            for fact in selected:
                rewritten = rewriter.rewrite_fact(fact['text'], title_keywords, settings.max_bullet_len)
                if not rewritten:
                    continue
                if any(extractor._too_similar(rewritten, old) for old in bullets):
                    continue
                bullets.append(rewritten)
                accepted_debug.append({
                    'block_idx': fact['block_idx'],
                    'topic': fact['topic'],
                    'score': fact['score'],
                    'raw_text': fact['text'],
                    'final_text': rewritten,
                })

        if not bullets:
            bullets = ['Не удалось собрать качественный конспект автоматически.']

        message = build_message(
            channel_name=channel['channel_title'],
            title=video['title'],
            video_url=video['url'],
            bullets=bullets[: settings.max_bullets],
            own_channel_url=settings.own_channel_url,
        )

        if is_publish_window_open(current_time, settings.publish_hour_start, settings.publish_hour_end) and published_at.astimezone(MSK).hour < settings.publish_hour_end:
            send_telegram(settings.telegram_bot_token, settings.telegram_chat_id, message)
            processed.add(video['id'])
            print('Пост отправлен')
        else:
            pending_items = pending.get('items', [])
            if not _has_video(pending_items, video['id']):
                pending_items.append({
                    'video_id': video['id'],
                    'message': message,
                    'queued_at': current_time.isoformat(),
                    'video_published_at': published_at.isoformat(),
                })
                pending['items'] = pending_items
                print('Видео поставлено в очередь на следующую публикацию')

        debug_payload = {
            'video_id': video['id'],
            'title': video['title'],
            'channel_name': channel['channel_title'],
            'transcript_status': transcript.status,
            'transcript_reason': transcript.reason,
            'transcript_metrics': transcript.metrics,
            'accepted_facts': accepted_debug,
            'rejected_candidates': reject_log,
        }
        save_debug_report(settings.debug_dir / f"{video['id']}.json", debug_payload)

    state['processed_video_ids'] = sorted(processed)
    save_json(settings.state_file, state)
    save_json(settings.pending_file, pending)
    save_json(settings.needs_review_file, needs_review)
    print('=' * 60)
    print('Готово')


if __name__ == '__main__':
    main()
