"""Recuperation de headlines Reddit (format RedditNews.csv)."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

ARCTIC_SHIFT_API = "https://arctic-shift.photon-reddit.com/api/posts/search"
# Subreddits cibles du projet (peuvent etre surcharges par le script/CLI).
DEFAULT_SUBREDDITS = ("stocks", "wallstreetbets", "StockMarket", "investing")
MAX_RETRIES = 5


class RedditFetchError(Exception):
    """Erreur lors de la recuperation Reddit."""


def _utc_date_from_ts(ts: int | float) -> date:
    return datetime.fromtimestamp(ts, tz=timezone.utc).date()


def _day_start_ts(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _day_end_ts(day: date) -> int:
    return _day_start_ts(day + timedelta(days=1))


def _ts_to_iso(ts: int | float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_with_retry(
    session: requests.Session,
    params: dict,
    retries: int = MAX_RETRIES,
) -> list[dict]:
    """Appel API avec retry sur 422/429/5xx."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(ARCTIC_SHIFT_API, params=params, timeout=60)
            if response.status_code in {422, 429, 500, 502, 503, 504}:
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json().get("data") or []
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 30))
    raise RedditFetchError(f"API Arctic Shift indisponible apres {retries} essais: {last_error}")


def fetch_day_arctic(
    subreddit: str,
    day: date,
    session: requests.Session | None = None,
    max_per_day: int | None = None,
    pause_s: float = 0.2,
) -> list[dict]:
    """Recupere les posts d'un subreddit pour un jour (pagination Arctic Shift)."""
    session = session or requests.Session()
    day_start = _day_start_ts(day)
    day_end = _day_end_ts(day)
    cursor_after = day_start
    seen_ids: set[str] = set()
    rows: list[dict] = []

    while cursor_after < day_end:
        params = {
            "subreddit": subreddit,
            "after": _ts_to_iso(cursor_after),
            "before": _ts_to_iso(day_end),
            "limit": 100,
            "sort": "asc",
            "fields": "title,created_utc,subreddit,id",
        }
        batch = _request_with_retry(session, params)
        if not batch:
            break

        new_in_batch = 0
        last_ts = cursor_after
        for post in batch:
            post_id = post.get("id", "")
            ts = post.get("created_utc")
            title = (post.get("title") or "").strip()
            if not ts or not title:
                continue
            if post_id and post_id in seen_ids:
                continue
            post_day = _utc_date_from_ts(ts)
            if post_day != day:
                continue
            if post_id:
                seen_ids.add(post_id)
            rows.append({
                "Date": post_day.isoformat(),
                "News": title,
                "subreddit": post.get("subreddit", subreddit),
            })
            new_in_batch += 1
            last_ts = max(last_ts, int(ts))

        if max_per_day and len(rows) >= max_per_day:
            return rows[:max_per_day]
        if new_in_batch == 0:
            break
        if last_ts <= cursor_after:
            cursor_after += 1
        else:
            cursor_after = last_ts + 1
        if len(batch) < 100:
            break
        time.sleep(pause_s)

    return rows


def fetch_range_arctic(
    start: date,
    end: date,
    subreddits: list[str] | None = None,
    max_per_day: int | None = None,
    max_total: int | None = None,
    on_progress=None,
    partial_output: str | None = None,
) -> pd.DataFrame:
    """Recupere les headlines sur une plage de dates."""
    subreddits = subreddits or list(DEFAULT_SUBREDDITS)
    session = requests.Session()
    all_rows: list[dict] = []
    current = start

    while current <= end:
        day_rows: list[dict] = []
        for sub in subreddits:
            try:
                day_rows.extend(
                    fetch_day_arctic(sub, current, session, max_per_day=max_per_day)
                )
            except RedditFetchError as exc:
                if all_rows and partial_output:
                    _save_partial(all_rows, partial_output)
                    raise RedditFetchError(
                        f"{exc} — {len(all_rows)} lignes partielles sauvees dans {partial_output}"
                    ) from exc
                raise
            time.sleep(0.15)

        all_rows.extend(day_rows)
        if partial_output and day_rows:
            _save_partial(all_rows, partial_output)

        if on_progress:
            on_progress(current, len(day_rows), len(all_rows))

        if max_total and len(all_rows) >= max_total:
            all_rows = all_rows[:max_total]
            break
        current += timedelta(days=1)

    if not all_rows:
        raise RedditFetchError(f"Aucun post trouve entre {start} et {end}.")

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["Date", "News"]).sort_values(["Date", "News"])
    # Conserver le subreddit aide pour l'EDA / analyse, et ne casse pas build_combined
    # (qui utilise seulement Date/News).
    if "subreddit" in df.columns:
        return df[["Date", "News", "subreddit"]]
    return df[["Date", "News"]]


def _save_partial(rows: list[dict], path: str) -> None:
    df = pd.DataFrame(rows).drop_duplicates(subset=["Date", "News"])
    df[["Date", "News"]].to_csv(path, index=False)


def fetch_recent_praw(
    subreddit: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fallback PRAW — posts recents uniquement (pas d'historique complet)."""
    import os

    try:
        import praw
    except ImportError as exc:
        raise RedditFetchError("Installez praw: pip install praw") from exc

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "stock-news-fetcher/1.0")
    if not client_id or not client_secret:
        raise RedditFetchError(
            "REDDIT_CLIENT_ID et REDDIT_CLIENT_SECRET requis pour PRAW."
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    rows = []
    for submission in reddit.subreddit(subreddit).new(limit=limit):
        rows.append({
            "Date": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).date().isoformat(),
            "News": submission.title.strip(),
        })
    return pd.DataFrame(rows).drop_duplicates(subset=["Date", "News"])
