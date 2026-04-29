"""Relevance rules for eval/inputs/retrieval/queries.csv (~41 queries)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_rule_factories import (
    _has_director, _has_actor, _has_country, _in_decade, _has_genre,
    _has_genres, _in_decade_and_genre, _has_media_type, _has_certification,
)

RELEVANCE_RULES = {
    # --- director --------------------------------------------------------------
    "David Fincher movies":                         _has_director("David Fincher"),
    "Christopher Nolan films":                      _has_director("Christopher Nolan"),
    "Bong Joon-ho films":                           _has_director("Bong Joon-ho"),
    "Quentin Tarantino movies":                     _has_director("Quentin Tarantino"),

    # --- actor -----------------------------------------------------------------
    "movies with Tom Hanks":                        _has_actor("Tom Hanks"),
    "something with Cate Blanchett":                _has_actor("Cate Blanchett"),
    "movies with Meryl Streep":                     _has_actor("Meryl Streep"),
    "show with Pedro Pascal":                       _has_actor("Pedro Pascal"),

    # --- country / language ----------------------------------------------------
    "Korean drama":                                 _has_country("KR"),
    "a good French film":                           _has_country("FR"),

    # --- genre (explicit) ------------------------------------------------------
    "romantic comedy":                              _has_genres("comedy", "romance"),

    # --- era -------------------------------------------------------------------
    "classic films from the 80s":                   _in_decade(1980, 1989),
    "good comedies from the 90s":                   _in_decade_and_genre(1990, 1999, "comedy"),

    # --- format / media type ---------------------------------------------------
    "a long series I can binge this weekend":       _has_media_type("SHOW"),

    # --- audience --------------------------------------------------------------
    "something to watch with my 8-year-old":        _has_certification("G", "PG", "TV-G", "TV-Y", "TV-Y7", "TV-PG"),

    # --- mood / genre proxy (imperfect but meaningful) -------------------------
    "something light and funny to cheer me up":     _has_genre("comedy"),
    "I need a good cry, what should I watch?":      _has_genre("drama"),
    "something intense that'll keep me on edge":    lambda r: _has_genre("thriller")(r) or _has_genre("horror")(r),
    "movies about artificial intelligence":         _has_genre("scifi"),
    "documentaries about climate change":           _has_genre("documentation"),
    "spy thriller":                                 _has_genre("thriller"),

    # --- complex ---------------------------------------------------------------
    "spy thriller with action by Martin Scorsese":  _has_director("Martin Scorsese"),
    "Christopher Nolan mind-bending thriller":      _has_director("Christopher Nolan"),
    "Tom Hanks drama about friendship":             _has_actor("Tom Hanks"),
    "Korean crime thriller like Parasite":          _has_country("KR"),
    "80s horror film":                              _in_decade_and_genre(1978, 1992, "horror"),
    "French romantic comedy":                       _has_country("FR"),
    "documentary about nature and wildlife":        _has_genre("documentation"),

    # --- negative preference (genre proxy) -------------------------------------
    "a comedy that isn't crude or raunchy":         _has_genre("comedy"),
    "action movie but nothing too gory":            _has_genre("action"),
}

SUBJECTIVE_QUERIES = {
    # mood: no reliable genre mapping
    "something comforting to watch when I'm sick",
    "I want something mind-bending",
    # topic: too specific for any metadata field
    "a film about grief and loss",
    "survival in the wild",
    "films about the financial crisis",
    # similarity
    "something like Inception but shorter",
    "if I liked Breaking Bad, what should I watch?",
    "movies in the same vein as The Godfather",
    # audience: no single filter captures intent
    "a movie for date night",
    "something I can have on in the background",
    # format: mini-series not distinguishable from series via metadata
    "just a mini-series, nothing too long",
}
