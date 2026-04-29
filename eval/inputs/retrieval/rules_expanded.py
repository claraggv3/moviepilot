"""Relevance rules for eval/inputs/retrieval/queries_expanded.csv (~100 queries)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from eval_rule_factories import (
    _has_director, _has_actor, _has_country, _in_decade, _has_genre,
    _has_genres, _in_decade_and_genre, _has_media_type, _has_certification,
    _runtime_lte, _runtime_gte,
)

RELEVANCE_RULES = {
    # --- director: named -------------------------------------------------------
    "David Fincher movies":                          _has_director("David Fincher"),
    "Christopher Nolan films":                       _has_director("Christopher Nolan"),
    "Bong Joon-ho films":                            _has_director("Bong Joon-ho"),
    "Quentin Tarantino movies":                      _has_director("Quentin Tarantino"),
    "Alfonso Cuarón films":                          _has_director("Alfonso Cuarón"),
    "Spike Lee movies":                              _has_director("Spike Lee"),
    "Wes Anderson films":                            _has_director("Wes Anderson"),
    "Ava DuVernay documentaries":                    _has_director("Ava DuVernay"),

    # --- director: indirect (same relevance, harder query) ---------------------
    "something by the director of Parasite":         _has_director("Bong Joon-ho"),
    "films by the woman who directed Selma":         _has_director("Ava DuVernay"),

    # --- actor: named ----------------------------------------------------------
    "movies with Tom Hanks":                         _has_actor("Tom Hanks"),
    "movies with Meryl Streep":                      _has_actor("Meryl Streep"),
    "show with Pedro Pascal":                        _has_actor("Pedro Pascal"),
    "Leonardo DiCaprio films":                       _has_actor("Leonardo DiCaprio"),
    "movies with Idris Elba":                        _has_actor("Idris Elba"),
    "Viola Davis movies":                            _has_actor("Viola Davis"),
    "Adam Sandler films on Netflix":                 _has_actor("Adam Sandler"),

    # --- actor: informal -------------------------------------------------------
    "something with Cate Blanchett":                 _has_actor("Cate Blanchett"),
    "anything with Scarlett Johansson":              _has_actor("Scarlett Johansson"),
    "show me something with Denzel":                 _has_actor("Denzel Washington"),

    # --- genre: single ---------------------------------------------------------
    "I want a horror movie":                         _has_genre("horror"),
    "recommend me a western":                        _has_genre("western"),
    "a crime drama":                                 _has_genre("crime"),
    "an animated film":                              _has_genre("animation"),
    "a science fiction movie":                       _has_genre("scifi"),
    "a romantic comedy":                             _has_genres("comedy", "romance"),

    # --- genre: compound -------------------------------------------------------
    "a sci-fi horror film":                          _has_genres("scifi", "horror"),
    "a romantic thriller":                           _has_genres("romance", "thriller"),
    "an action comedy":                              _has_genres("action", "comedy"),

    # --- era -------------------------------------------------------------------
    "classic films from the 80s":                    _in_decade(1980, 1989),
    "good films from the 70s":                       _in_decade(1970, 1979),
    "something from the 90s I might have missed":    _in_decade(1990, 1999),
    "80s horror film":                               _in_decade_and_genre(1978, 1992, "horror"),
    "90s crime thriller":                            _in_decade_and_genre(1990, 1999, "thriller"),
    "70s drama":                                     _in_decade_and_genre(1970, 1979, "drama"),
    "a drama from the 2000s":                        _in_decade_and_genre(2000, 2009, "drama"),
    "2000s romantic comedy":                         lambda r: _in_decade(2000, 2009)(r) and _has_genres("comedy", "romance")(r),

    # --- country / language ----------------------------------------------------
    "any good Italian movies on Netflix":            _has_country("IT"),
    "a good Spanish film":                           _has_country("ES"),
    "Indian cinema":                                 _has_country("IN"),
    "German thriller":                               _has_country("DE"),
    "British comedy":                                _has_country("GB"),
    "recommend something in Korean":                 _has_country("KR"),
    "a good Japanese film":                          _has_country("JP"),
    "something in Mandarin":                         _has_country("CN"),
    "I want to watch something French":              _has_country("FR"),
    "show me something Brazilian":                   _has_country("BR"),

    # --- format / media type ---------------------------------------------------
    "just a movie not a series":                     _has_media_type("MOVIE"),
    "a long series I can binge this weekend":        _has_media_type("SHOW"),
    "a good documentary":                            _has_genre("documentation"),
    "documentary about nature and wildlife":         _has_genre("documentation"),
    "a documentary series about nature":             lambda r: _has_genre("documentation")(r) and _has_media_type("SHOW")(r),

    # --- runtime ---------------------------------------------------------------
    "a short film under 90 minutes":                 lambda r: _has_media_type("MOVIE")(r) and _runtime_lte(90)(r),
    "a series with short episodes":                  lambda r: _has_media_type("SHOW")(r)  and _runtime_lte(30)(r),
    "something I can watch all evening":             _runtime_gte(120),
    "a drama with long episodes":                    lambda r: _has_media_type("SHOW")(r)  and _runtime_gte(45)(r),

    # --- certification ---------------------------------------------------------
    "something family-friendly rated for all ages":  _has_certification("G", "PG", "TV-G", "TV-Y", "TV-Y7", "TV-PG"),
    "a children's film":                             _has_certification("G", "PG", "TV-G", "TV-Y", "TV-Y7"),
    "a mature R-rated film":                         _has_certification("R", "TV-MA", "NC-17"),

    # --- combination: 2 params -------------------------------------------------
    "spy thriller with action by Martin Scorsese":   _has_director("Martin Scorsese"),
    "Christopher Nolan mind-bending thriller":       _has_director("Christopher Nolan"),
    "Tom Hanks drama about friendship":              _has_actor("Tom Hanks"),
    "Korean crime thriller like Parasite":           _has_country("KR"),
    "French romantic comedy":                        _has_country("FR"),
    "an 80s Korean horror film":                     lambda r: _has_country("KR")(r) and _in_decade_and_genre(1978, 1992, "horror")(r),
    "a Japanese anime series":                       lambda r: _has_country("JP")(r) and _has_genre("animation")(r) and _has_media_type("SHOW")(r),

    # --- combination: 3 params -------------------------------------------------
    "a short French thriller from the 90s":          lambda r: _has_country("FR")(r) and _has_genre("thriller")(r) and _in_decade(1990, 1999)(r),
    "a 2000s Korean crime drama":                    lambda r: _has_country("KR")(r) and _has_genre("crime")(r) and _in_decade(2000, 2009)(r),
    "an 80s British sci-fi comedy":                  lambda r: _has_country("GB")(r) and _has_genre("scifi")(r) and _in_decade(1980, 1989)(r),
}

SUBJECTIVE_QUERIES = {
    # mood
    "something light and funny to cheer me up",
    "I need a good cry what should I watch",
    "something intense that'll keep me on edge",
    "something comforting to watch when I'm sick",
    "I want something mind-bending",
    # topic / theme
    "movies about artificial intelligence",
    "a film about grief and loss",
    "survival in the wild",
    "something about the financial crisis",
    "documentaries about climate change",
    "films about space exploration",
    "movies about immigration",
    "a film set during World War II",
    # audience / context
    "something to watch with my 8-year-old",
    "a movie for date night",
    "something I can have on in the background",
    "a film to watch with my parents",
    "something the whole family can watch",
    # format / runtime (no objective filter available)
    "just a mini-series nothing too long",
    # genre: niche / subjective
    "a martial arts film",
    "a film noir",
    # era: relative / discovery
    "a hidden gem from the early 2000s",
    # similarity
    "something like Inception",
    "movies in the same vein as The Godfather",
    "something like Narcos",
    "a film like Casablanca",
    "if I liked Breaking Bad what should I watch",
    # quality / rating (IMDB not used as filter)
    "highly rated thrillers on Netflix",
    "best reviewed drama series",
    # negative preference
    "something not too violent I've had a rough day",
    "a comedy that isn't crude or raunchy",
    "action movie but nothing too gory",
}
