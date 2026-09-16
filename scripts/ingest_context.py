#!/usr/bin/env python3
"""Ingest official FPL status, RSS/news, and optional X context events.

The output is normalized JSONL with publication and expiry timestamps. X API
access is intentionally credential-gated through ``X_BEARER_TOKEN`` (or the
flag below); the command never scrapes a browser session or stores credentials.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpl_lab.context import ContextEvent, ContextStore, bootstrap_news_events, load_context_events, normalize_text


def _fetch_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "fpl-lab-context/1.0", **(headers or {})})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _strip_html(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()


def _rss_records(url: str) -> list[dict[str, Any]]:
    request = Request(url, headers={"User-Agent": "fpl-lab-context/1.0"})
    with urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())
    records = []
    for item in root.findall(".//item") + root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        def child(*names: str) -> str:
            for name in names:
                node = item.find(name)
                if node is None:
                    node = item.find(f"{{http://www.w3.org/2005/Atom}}{name}")
                if node is not None and node.text:
                    return node.text.strip()
            return ""

        title = child("title")
        description = child("description", "summary", "content")
        published = child("pubDate", "published", "updated")
        link = child("link")
        records.append(
            {
                "event_id": link or f"{url}::{published}::{title}",
                "published_at": published,
                "source": url,
                "title": _strip_html(title),
                "body": _strip_html(description),
                "url": link,
            }
        )
    return records


def _x_records(query: str, bearer_token: str, max_results: int) -> list[dict[str, Any]]:
    params = urlencode(
        {
            "query": query,
            "max_results": max(10, min(100, max_results)),
            "tweet.fields": "created_at,public_metrics,author_id,lang",
            "expansions": "author_id",
            "user.fields": "name,username,verified",
        }
    )
    payload = _fetch_json(
        f"https://api.x.com/2/tweets/search/recent?{params}",
        headers={"Authorization": f"Bearer {bearer_token}"},
    )
    users = {str(user["id"]): user for user in payload.get("includes", {}).get("users", [])}
    records = []
    for tweet in payload.get("data", []):
        author = users.get(str(tweet.get("author_id")), {})
        metrics = tweet.get("public_metrics") or {}
        records.append(
            {
                "event_id": f"x-{tweet.get('id')}",
                "published_at": tweet.get("created_at"),
                "source": "X",
                "title": tweet.get("text", ""),
                "body": tweet.get("text", ""),
                "author": author.get("username") or author.get("name", ""),
                "verified": bool(author.get("verified", False)),
                "engagement": sum(int(metrics.get(key, 0) or 0) for key in ("like_count", "retweet_count", "reply_count", "quote_count")),
                "tweet_id": tweet.get("id"),
            }
        )
    return records


def _player_map(bootstrap: dict[str, Any]) -> list[dict[str, str]]:
    output = []
    for player in bootstrap.get("elements", []):
        names = {
            normalize_text(player.get("web_name")),
            normalize_text(f"{player.get('first_name', '')} {player.get('second_name', '')}"),
        }
        output.append(
            {
                "id": str(player.get("id")),
                "team": str(player.get("team")),
                "name": str(player.get("web_name") or ""),
                "names": json.dumps(sorted(name for name in names if name)),
            }
        )
    return output


def _resolve_entities(records: list[dict[str, Any]], bootstrap: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not bootstrap:
        return records
    players = []
    for row in _player_map(bootstrap):
        names = json.loads(row["names"])
        players.append((max((len(name) for name in names), default=0), names, row))
    players.sort(reverse=True, key=lambda item: item[0])
    output = []
    for record in records:
        if record.get("player_id"):
            output.append(record)
            continue
        text = normalize_text(f"{record.get('title', '')} {record.get('body', '')}")
        for _length, names, player in players:
            if any(re.search(rf"\b{re.escape(name)}\b", text) for name in names if name):
                record = dict(record)
                record["player_id"] = player["id"]
                record["player_name"] = player["name"]
                record["team"] = player["team"]
                break
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default=None, help="official bootstrap-static.json snapshot")
    parser.add_argument("--news-file", action="append", default=[], help="local news JSON/JSONL/CSV")
    parser.add_argument("--social-file", action="append", default=[], help="local social JSON/JSONL/CSV")
    parser.add_argument("--rss-url", action="append", default=[], help="RSS/Atom news feed URL")
    parser.add_argument("--x-query", action="append", default=[], help="X recent-search query; requires a bearer token")
    parser.add_argument("--x-bearer-token", default=None, help="X bearer token; prefer X_BEARER_TOKEN")
    parser.add_argument("--x-max-results", type=int, default=50)
    parser.add_argument("--news-out", default="data/context/news.jsonl")
    parser.add_argument("--social-out", default="data/context/social.jsonl")
    parser.add_argument("--manifest-out", default="data/context/manifest.json")
    args = parser.parse_args()

    bootstrap = None
    news_events: list[ContextEvent] = []
    social_events: list[ContextEvent] = []
    if args.bootstrap:
        bootstrap = json.loads(Path(args.bootstrap).read_text(encoding="utf-8"))
        news_events.extend(bootstrap_news_events(bootstrap))
    for path in args.news_file:
        news_events.extend(load_context_events(path, kind="news"))
    for path in args.social_file:
        social_events.extend(load_context_events(path, kind="social"))
    for url in args.rss_url:
        news_events.extend(ContextStore.from_records(_resolve_entities(_rss_records(url), bootstrap), kind="news").events)
    if args.x_query:
        token = args.x_bearer_token
        if not token:
            import os

            token = os.environ.get("X_BEARER_TOKEN")
        if not token:
            raise ValueError("--x-query requires --x-bearer-token or X_BEARER_TOKEN")
        for query in args.x_query:
            records = _resolve_entities(_x_records(query, token, args.x_max_results), bootstrap)
            social_events.extend(ContextStore.from_records(records, kind="social").events)

    news_store = ContextStore(news_events)
    social_store = ContextStore(social_events)
    news_store.write_jsonl(args.news_out)
    social_store.write_jsonl(args.social_out)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "news": news_store.summary(),
        "social": social_store.summary(),
        "sources": {
            "bootstrap": args.bootstrap,
            "rss_urls": args.rss_url,
            "x_queries": args.x_query,
        },
        "temporal_rule": "features include only published_at <= decision cutoff and exclude expired events",
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
