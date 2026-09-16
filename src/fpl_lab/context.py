"""Timestamped injury, team-news, and social-context features.

The context layer is deliberately provider-neutral. It accepts normalized
JSON/JSONL/CSV records from the official FPL bootstrap, RSS/news feeds, or the
X API and only exposes events known by an explicit ``as_of`` timestamp. This
keeps current news useful for live decisions without allowing a later update
to leak into a historical backtest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


UTC = timezone.utc
_WORD_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(value: object) -> str:
    """Normalize names and free text for conservative entity matching."""

    text = str(value or "").strip().lower()
    return _WORD_RE.sub(" ", text).strip()


def parse_timestamp(value: object) -> datetime | None:
    """Parse an ISO/RFC3339 timestamp into timezone-aware UTC."""

    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        # RFC 2822 timestamps are common in RSS and X exports.
        from email.utils import parsedate_to_datetime

        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _infer_event_type(text: str) -> str:
    lowered = normalize_text(text)
    if any(word in lowered for word in ("injury", "injured", "knock", "fitness", "hamstring", "illness")):
        return "injury"
    if any(word in lowered for word in ("suspend", "ban", "red card")):
        return "suspension"
    if any(word in lowered for word in ("doubt", "miss", "ruled out", "out for")):
        return "availability"
    if any(word in lowered for word in ("rotation", "rested", "benched", "bench")):
        return "rotation"
    if any(word in lowered for word in ("starts", "starting", "back in training", "fit to play")):
        return "role_positive"
    if any(word in lowered for word in ("price rise", "price fall", "transfers in", "transfers out")):
        return "price"
    return "general"


def _infer_sentiment(text: str, event_type: str) -> float:
    lowered = normalize_text(text)
    negative = ("injury", "injured", "doubt", "out", "suspend", "ban", "rotation", "benched", "miss")
    positive = ("fit", "starts", "starting", "return", "back", "available", "signed")
    score = sum(lowered.count(word) for word in positive) - sum(lowered.count(word) for word in negative)
    if event_type in {"injury", "suspension", "availability", "rotation"} and score >= 0:
        score = -1
    return _clip(score / 3.0, -1.0, 1.0)


def _default_reliability(kind: str, source: str) -> float:
    source_key = normalize_text(source)
    if "fantasy premier league" in source_key or source_key in {"fpl", "official fpl"}:
        return 1.0
    if "premier league" in source_key:
        return 0.95
    if kind == "social" or source_key in {"x", "twitter"}:
        return 0.35
    return 0.55


@dataclass(frozen=True)
class ContextEvent:
    """One normalized event with an explicit publication and expiry time."""

    event_id: str
    kind: str
    published_at: datetime
    source: str
    title: str
    body: str = ""
    player_id: str | None = None
    player_name: str | None = None
    team: str | None = None
    event_type: str = "general"
    sentiment: float = 0.0
    reliability: float = 0.5
    expires_at: datetime | None = None
    author: str = ""
    engagement: float = 0.0
    verified: bool = False

    def to_record(self) -> dict[str, Any]:
        row = asdict(self)
        row["published_at"] = self.published_at.isoformat()
        row["expires_at"] = self.expires_at.isoformat() if self.expires_at else None
        return row


@dataclass(frozen=True)
class ContextFeatures:
    """As-of aggregate used to modify a player's pre-deadline signals."""

    availability_delta: float = 0.0
    role_security_delta: float = 0.0
    news_risk: float = 0.0
    news_sentiment: float = 0.0
    social_sentiment: float = 0.0
    price_pressure: float = 0.0
    reliability: float = 0.0
    event_count: int = 0
    news_count: int = 0
    social_count: int = 0
    latest_event_at: datetime | None = None

    def to_record(self) -> dict[str, Any]:
        row = asdict(self)
        row["latest_event_at"] = self.latest_event_at.isoformat() if self.latest_event_at else None
        return row


class ContextStore:
    """In-memory, point-in-time event store for news and social signals."""

    def __init__(self, events: Iterable[ContextEvent] = ()):
        self.events = sorted(list(events), key=lambda event: (event.published_at, event.event_id))
        self._by_player_id: dict[str, list[ContextEvent]] = {}
        self._by_player_name: dict[str, list[ContextEvent]] = {}
        self._by_team: dict[str, list[ContextEvent]] = {}
        self._team_level: list[ContextEvent] = []
        for event in self.events:
            if event.player_id:
                self._by_player_id.setdefault(str(event.player_id), []).append(event)
            if event.player_name:
                self._by_player_name.setdefault(normalize_text(event.player_name), []).append(event)
            if event.team:
                self._by_team.setdefault(normalize_text(event.team), []).append(event)
            if not event.player_id and not event.player_name and event.team:
                self._team_level.append(event)

    @classmethod
    def from_paths(
        cls,
        news_path: str | Path | None = None,
        social_path: str | Path | None = None,
    ) -> "ContextStore":
        events: list[ContextEvent] = []
        if news_path:
            events.extend(load_context_events(news_path, kind="news"))
        if social_path:
            events.extend(load_context_events(social_path, kind="social"))
        return cls(events)

    @classmethod
    def from_records(cls, records: Iterable[dict[str, Any]], kind: str = "news") -> "ContextStore":
        return cls(_event_from_record(record, kind=kind, index=index) for index, record in enumerate(records))

    def features_for_player(
        self,
        player_id: str,
        player_name: str,
        team: str,
        as_of: datetime | None,
    ) -> ContextFeatures:
        """Aggregate only events published by the decision cutoff."""

        if as_of is None:
            return ContextFeatures()
        cutoff = parse_timestamp(as_of)
        if cutoff is None:
            return ContextFeatures()
        player_key = normalize_text(player_name)
        team_key = normalize_text(team)
        candidate_events = {
            *self._by_player_id.get(str(player_id), []),
            *self._by_player_name.get(player_key, []),
            *self._by_team.get(team_key, []),
            *self._team_level,
        }
        matched: list[tuple[ContextEvent, float]] = []
        for event in sorted(candidate_events, key=lambda item: (item.published_at, item.event_id)):
            if event.published_at > cutoff:
                continue
            if event.expires_at is not None and event.expires_at < cutoff:
                continue
            event_player_id = str(event.player_id or "")
            event_player_name = normalize_text(event.player_name)
            event_team = normalize_text(event.team)
            direct_match = event_player_id != "" and event_player_id == str(player_id)
            name_match = event_player_name != "" and event_player_name == player_key
            team_match = event_team != "" and event_team == team_key and not event_player_name and not event_player_id
            if not (direct_match or name_match or team_match):
                continue
            age_hours = max(0.0, (cutoff - event.published_at).total_seconds() / 3600.0)
            recency = float(np.exp(-age_hours / 48.0))
            match_weight = recency * _clip(event.reliability, 0.0, 1.0)
            if team_match:
                match_weight *= 0.35
            matched.append((event, match_weight))

        if not matched:
            return ContextFeatures()

        availability_delta = 0.0
        role_delta = 0.0
        news_risk = 0.0
        news_sentiment = 0.0
        social_sentiment = 0.0
        price_pressure = 0.0
        total_weight = 0.0
        latest = max(event.published_at for event, _ in matched)
        for event, weight in matched:
            event_type = event.event_type
            if event_type in {"injury", "suspension"}:
                availability_delta -= 0.45 * weight
                role_delta -= 0.35 * weight
                news_risk += 0.80 * weight
            elif event_type == "availability":
                availability_delta -= 0.35 * weight
                role_delta -= 0.20 * weight
                news_risk += 0.60 * weight
            elif event_type == "rotation":
                availability_delta -= 0.20 * weight
                role_delta -= 0.25 * weight
                news_risk += 0.35 * weight
            elif event_type == "role_positive":
                availability_delta += 0.20 * weight
                role_delta += 0.25 * weight
                news_risk -= 0.15 * weight
            if event_type == "price":
                price_pressure += event.sentiment * weight
            news_sentiment += event.sentiment * weight
            if event.kind == "social":
                # Social chatter is deliberately attenuated relative to
                # official/team reporting and is retained as a separate field
                # for ablation tests.
                social_sentiment += event.sentiment * weight * 0.50
            total_weight += weight

        news_events = sum(event.kind == "news" for event, _ in matched)
        social_events = sum(event.kind == "social" for event, _ in matched)
        return ContextFeatures(
            availability_delta=_clip(availability_delta, -0.75, 0.50),
            role_security_delta=_clip(role_delta, -0.75, 0.50),
            news_risk=_clip(news_risk, 0.0, 1.0),
            news_sentiment=_clip(news_sentiment / max(1.0, total_weight), -1.0, 1.0),
            social_sentiment=_clip(social_sentiment / max(1.0, total_weight), -1.0, 1.0),
            price_pressure=_clip(price_pressure, -1.0, 1.0),
            reliability=_clip(total_weight / max(1, len(matched)), 0.0, 1.0),
            event_count=len(matched),
            news_count=news_events,
            social_count=social_events,
            latest_event_at=latest,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "events": len(self.events),
            "news_events": sum(event.kind == "news" for event in self.events),
            "social_events": sum(event.kind == "social" for event in self.events),
            "sources": sorted({event.source for event in self.events}),
            "first_published_at": self.events[0].published_at.isoformat() if self.events else None,
            "last_published_at": self.events[-1].published_at.isoformat() if self.events else None,
        }

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event.to_record(), ensure_ascii=False) + "\n")


def _event_from_record(record: dict[str, Any], kind: str, index: int) -> ContextEvent:
    title = str(record.get("title") or record.get("text") or record.get("headline") or "").strip()
    body = str(record.get("body") or record.get("description") or record.get("text") or "").strip()
    source = str(record.get("source") or record.get("provider") or ("x" if kind == "social" else "unknown"))
    published = parse_timestamp(
        record.get("published_at")
        or record.get("created_at")
        or record.get("timestamp")
        or record.get("date")
    )
    if published is None:
        raise ValueError(f"context record {index} is missing a valid published_at timestamp")
    event_type = str(record.get("event_type") or "").strip() or _infer_event_type(f"{title} {body}")
    sentiment = _clip(
        _float(record.get("sentiment"), _infer_sentiment(f"{title} {body}", event_type)),
        -1.0,
        1.0,
    )
    reliability = _clip(
        _float(record.get("reliability"), _default_reliability(kind, source)),
        0.0,
        1.0,
    )
    expiry = parse_timestamp(record.get("expires_at"))
    if expiry is None:
        ttl_hours = 72 if event_type in {"injury", "suspension", "availability"} else 48
        expiry = published + timedelta(hours=ttl_hours)
    event_id = str(record.get("event_id") or record.get("id") or "").strip()
    if not event_id:
        digest = hashlib.sha1(
            f"{kind}|{source}|{published.isoformat()}|{title}|{record.get('player_id', '')}".encode("utf-8")
        ).hexdigest()[:16]
        event_id = f"{kind}-{digest}"
    public_metrics = record.get("public_metrics")
    if not isinstance(public_metrics, dict):
        public_metrics = {}
    engagement_value = record.get("engagement") or record.get("like_count") or public_metrics.get("like_count")
    return ContextEvent(
        event_id=event_id,
        kind=kind,
        published_at=published,
        source=source,
        title=title,
        body=body,
        player_id=str(record.get("player_id")) if record.get("player_id") is not None else None,
        player_name=str(record.get("player_name") or record.get("name")) if record.get("player_name") or record.get("name") else None,
        team=str(record.get("team")) if record.get("team") is not None else None,
        event_type=event_type,
        sentiment=sentiment,
        reliability=reliability,
        expires_at=expiry,
        author=str(record.get("author") or record.get("username") or ""),
        engagement=_float(engagement_value),
        verified=bool(record.get("verified", False)),
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("events", "data", "items", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError(f"unsupported context payload in {path}")


def load_context_events(path: str | Path, kind: str = "news") -> list[ContextEvent]:
    """Load normalized or provider-native records from JSON, JSONL, or CSV."""

    source = Path(path)
    records = _read_records(source)
    events: list[ContextEvent] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        try:
            events.append(_event_from_record(record, kind=kind, index=index))
        except ValueError:
            # A malformed provider row should not erase the rest of a feed;
            # the ingestion script reports the dropped-row count separately.
            continue
    return events


def bootstrap_news_events(bootstrap: dict[str, Any], fetched_at: datetime | None = None) -> list[ContextEvent]:
    """Convert official FPL player-news fields into timestamped events."""

    now = parse_timestamp(fetched_at) or datetime.now(UTC)
    events: list[ContextEvent] = []
    for player in bootstrap.get("elements", []):
        news = str(player.get("news") or "").strip()
        status = str(player.get("status") or "").strip().lower()
        chance = player.get("chance_of_playing_next_round")
        chance_value = _float(chance, 100.0) if chance is not None else 100.0
        if not news and status in {"a", "i"} and chance_value >= 100:
            continue
        if not news:
            news = f"Official FPL status: {status or 'unknown'}; chance next round {chance_value:.0f}%"
        published = parse_timestamp(player.get("news_added")) or now
        event_type = "availability" if chance_value < 75 or status in {"i", "s", "u"} else "general"
        if chance_value < 25:
            event_type = "injury"
        sentiment = _clip((chance_value - 50.0) / 50.0, -1.0, 1.0)
        events.append(
            _event_from_record(
                {
                    "event_id": f"fpl-bootstrap-{player.get('id')}-{published.isoformat()}",
                    "published_at": published.isoformat(),
                    "source": "official FPL",
                    "title": news,
                    "body": news,
                    "player_id": player.get("id"),
                    "player_name": player.get("web_name") or f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
                    "team": player.get("team"),
                    "event_type": event_type,
                    "sentiment": sentiment,
                    "reliability": 1.0,
                    "expires_at": (published + timedelta(hours=36)).isoformat(),
                },
                kind="news",
                index=int(player.get("id", 0)),
            )
        )
    return events
