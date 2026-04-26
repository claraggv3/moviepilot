# Query Testset — MoviePilot

50+ realistic user queries grouped by the NLP/routing problem they test.
Each category exposes a distinct failure mode or capability requirement.

---

## Category 1 — Named director (tests BM25 recall on proper nouns)
Semantic search is weak here: "Fincher" doesn't appear in plot descriptions.
BM25 catches it if we embed director names in the document text.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 1 | "Show me Christopher Nolan films on Netflix" | netflix | Director name as primary signal |
| 2 | "Any David Fincher movies?" | netflix | Director name, informal phrasing |
| 3 | "I love Bong Joon-ho, what does he have on Netflix?" | netflix | Non-English director name |
| 4 | "Something by the director of Parasite" | netflix | Indirect director reference — requires inference |

---

## Category 2 — Named actor (tests BM25 recall on proper nouns)
Same problem as directors. "Tom Hanks" won't appear in the description of most of his films.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 5 | "Movies with Tom Hanks on Netflix" | netflix | A-list actor, direct mention |
| 6 | "Something with Cate Blanchett" | netflix | Actor name without "Netflix" keyword |
| 7 | "I want a film starring Meryl Streep" | netflix | Formal phrasing |
| 8 | "What has Pedro Pascal been in on Netflix?" | netflix | TV/film crossover actor |

---

## Category 3 — Pure genre, no recency signal (tests routing decision)
These should route to netflix, not trending. No temporal signal present.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 9 | "I want a horror movie" | netflix | Single genre, unambiguous |
| 10 | "Recommend me a western" | netflix | Niche genre |
| 11 | "A good thriller" | netflix | Vague genre, no recency |
| 12 | "Give me a romantic comedy" | netflix | Classic genre combo |

---

## Category 4 — Mood / emotional state (tests abstract semantic retrieval)
The user doesn't say a genre — they describe a feeling. Semantic search must bridge the gap.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 13 | "Something light and funny to cheer me up" | netflix | Mood → genre inference |
| 14 | "I need a good cry, what should I watch?" | netflix | Emotional trigger → drama |
| 15 | "Something intense that'll keep me on edge" | netflix | Tension/suspense without genre word |
| 16 | "I'm feeling nostalgic, any recommendations?" | netflix | Abstract mood, era signal |
| 17 | "Something comforting to watch when I'm sick" | netflix | Comfort viewing — no clear genre |
| 18 | "I want something mind-bending" | netflix | Cognitive complexity signal |

---

## Category 5 — Year / era (tests temporal metadata without recency routing)
References to specific decades should route to netflix (catalog), not trending.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 19 | "Classic films from the 80s on Netflix" | netflix | Decade as metadata filter |
| 20 | "Something from the 90s I might have missed" | netflix | Discovery framing + era |
| 21 | "A 70s horror film" | netflix | Decade + genre combo |
| 22 | "Good comedies from the 2000s" | netflix | Recent-ish era, still catalog |

---

## Category 6 — Language / country of origin (tests metadata retrieval)
Users often want international content by language, not genre.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 23 | "A good Spanish-language film" | netflix | Language as filter |
| 24 | "Recommend something in Korean" | netflix | K-drama popularity signal |
| 25 | "I want to watch something French" | netflix | Language without explicit "film" |
| 26 | "Any good Italian movies on Netflix?" | netflix | Country of origin |

---

## Category 7 — Topic / theme (tests deep semantic retrieval)
No genre word, no actor, no year — pure thematic search.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 27 | "Movies about artificial intelligence" | netflix | Technical topic → semantic match |
| 28 | "Something about the financial crisis" | netflix | Specific event as theme |
| 29 | "A film about grief and loss" | netflix | Emotional theme without genre |
| 30 | "Something about survival in the wild" | netflix | Physical scenario theme |
| 31 | "Documentaries about climate change" | netflix | Topic + format combo |

---

## Category 8 — Recency / trending (tests routing to trending agent)
These should all route to trending. They contain explicit or implicit recency signals.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 32 | "What's popular right now?" | trending | Explicit recency |
| 33 | "Best movies out this week" | trending | Temporal "this week" |
| 34 | "What should I watch tonight that's new?" | trending | "New" + same-day recency |
| 35 | "Any good movies released lately?" | trending | "Lately" as recency signal |
| 36 | "What's everyone watching right now?" | trending | Social popularity + recency |
| 37 | "Is there anything good at the cinema currently?" | trending | Cinema implies current releases |

---

## Category 9 — Similarity queries (tests semantic search quality)
"Something like X" requires the embedding of X's themes to match similar titles.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 38 | "Something like Inception" | netflix | Abstract concept similarity |
| 39 | "If I liked Breaking Bad, what should I watch?" | netflix | TV series similarity |
| 40 | "Movies in the same vein as The Godfather" | netflix | Classic film reference |

---

## Category 10 — Audience / viewing context (tests semantic inference)
No genre stated — the context implies the type of content needed.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 41 | "Something to watch with my 8-year-old" | netflix | Age-appropriate inference |
| 42 | "A movie for date night" | netflix | Romantic context without "romance" |
| 43 | "Something I can have on in the background" | netflix | Passive viewing — low intensity |
| 44 | "A film to watch with my parents" | netflix | Multi-generational audience |

---

## Category 11 — Format / length preference (tests metadata filtering)
Users often know what format they want, not what content.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 45 | "A short film under 90 minutes" | netflix | Runtime as filter |
| 46 | "A long series I can binge this weekend" | netflix | Format (series) + length |
| 47 | "Just a mini-series, nothing too long" | netflix | Short format preference |

---

## Category 12 — Ambiguous routing (routing edge cases)
These test the boundary between netflix and trending. The router must not mis-classify.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 48 | "What's a good action movie?" | netflix | No temporal signal → catalog |
| 49 | "Best action movie out now?" | trending | "Out now" tips to trending |
| 50 | "Is there a good sci-fi film recently released?" | trending | "Recently released" = trending |
| 51 | "Recommend an action film" | netflix | Identical to 48, different phrasing |

---

## Category 13 — Multi-turn follow-ups (tests router context window)
These only make sense in a conversation. On their own they're ambiguous — the router must use prior messages.

| # | Query | Expected route | Notes |
|---|---|---|---|
| 52 | "Tell me more about the first one" | same as prior | Pronoun resolution |
| 53 | "Something shorter than that" | same as prior | Comparative reference |
| 54 | "What about something in the same genre but older?" | netflix | Relative constraint |
| 55 | "Are any of those on Netflix?" | netflix | Platform pivot from trending results |

---

## Category 14 — Negative preference (tests semantic retrieval robustness)
The user says what they DON'T want. This is harder for retrieval — absence of a feature isn't directly encoded in embeddings.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 56 | "Something not too violent, I've had a rough day" | netflix | Negative genre signal |
| 57 | "A comedy that isn't crude or raunchy" | netflix | Negative content filter |
| 58 | "Action movie but nothing too gory" | netflix | Genre + negative modifier |

---

## Category 15 — Out-of-scope (tests refusal)
These should all route to refusal. No movie/show recommendation intent.

| # | Query | Expected route | Problem tested |
|---|---|---|---|
| 59 | "What's the weather in Madrid?" | refusal | Completely unrelated domain |
| 60 | "Help me write a cover letter" | refusal | Task request, not recommendation |
| 61 | "What's 2 + 2?" | refusal | Math — zero movie intent |
| 62 | "Tell me a joke" | refusal | Entertainment but not movie rec |
| 63 | "Who won the last World Cup?" | refusal | Sports, no movie intent |
| 64 | "Can you translate this to French?" | refusal | Translation task |

---

## Identified retrieval problems by category

| Problem | Categories affected | Solution in our system |
|---|---|---|
| Named entity miss (director/actor) | 1, 2 | BM25 lexical match + credits in document text |
| Query-document vocabulary gap | 4, 7, 9 | HyDE expansion before embedding |
| Temporal signal ambiguity | 8, 12 | Router context window (last 3 messages) |
| Metadata not in embeddings (year, language, runtime) | 5, 6, 11 | Front-loaded in document text + metadata filter |
| Pronoun/ellipsis in follow-ups | 13 | Router sees last N messages |
| Negative preference | 14 | LLM filters at generation time (retrieval can't negate) |
| Scope boundary (refusal) | 15 | Structured output router with explicit refusal class |
