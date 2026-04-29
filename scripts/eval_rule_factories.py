"""Factory functions for building relevance-rule callables used by eval_stats.py."""
from __future__ import annotations

from typing import Callable


def _has_director(name: str) -> Callable:
    return lambda r: name in r.get("directors", "")


def _has_actor(name: str) -> Callable:
    return lambda r: name in r.get("actors", "")


def _has_country(code: str) -> Callable:
    return lambda r: code in r.get("production_countries", "")


def _in_decade(start: int, end: int) -> Callable:
    def rule(r: dict) -> bool:
        y = r.get("year", "")
        return bool(y) and start <= int(y) <= end
    return rule


def _has_genre(genre: str) -> Callable:
    return lambda r: genre.lower() in r.get("genres", "").lower()


def _has_genres(*genres: str) -> Callable:
    return lambda r: all(g.lower() in r.get("genres", "").lower() for g in genres)


def _in_decade_and_genre(start: int, end: int, genre: str) -> Callable:
    decade_rule = _in_decade(start, end)
    genre_rule = _has_genre(genre)
    return lambda r: decade_rule(r) and genre_rule(r)


def _has_media_type(t: str) -> Callable:
    return lambda r: r.get("type", "").upper() == t.upper()


def _has_certification(*certs: str) -> Callable:
    cert_set = {c.upper() for c in certs}
    return lambda r: r.get("age_certification", "").upper() in cert_set


def _runtime_lte(minutes: int) -> Callable:
    def rule(r: dict) -> bool:
        rt = r.get("runtime")
        return bool(rt) and int(rt) <= minutes
    return rule


def _runtime_gte(minutes: int) -> Callable:
    def rule(r: dict) -> bool:
        rt = r.get("runtime")
        return bool(rt) and int(rt) >= minutes
    return rule
