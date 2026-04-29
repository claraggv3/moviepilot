# Eval summary

Generated: 2026-04-29 22:19  
Cases run: 20  

## Aggregate scores

| Metric | Score |
|---|---|
| Routing accuracy | 20/20 (100%) |
| L1 groundedness  | 20/20 (100%) |
| Retrieval recall (predicate-based) | 92.2% |
| LLM judge — relevance    | 5.00 ± 0.00 / 5 |
| LLM judge — groundedness | 5.00 ± 0.00 / 5 |
| LLM judge — helpfulness  | 5.00 ± 0.00 / 5 |

## Per-case results

| ID | Query | Expected | Actual | Grounded | Recall | R / G / H |
|---|---|---|---|---|---| --- |
| t1 | What's popular right now? | trending | trending ✓ | ✓ | — | 5/5/5 |
| t2 | Best movies out this week | trending | trending ✓ | ✓ | — | 5/5/5 |
| t3 | Is there anything good at the cinema c | trending | trending ✓ | ✓ | — | 5/5/5 |
| t4 | What's everyone watching right now? | trending | trending ✓ | ✓ | — | 5/5/5 |
| t5 | Any good movies released lately? | trending | trending ✓ | ✓ | — | 5/5/5 |
| n1 | Show me Christopher Nolan films on Net | netflix | netflix ✓ | ✓ | 100% (3/3) | 5/5/5 |
| n2 | Movies with Tom Hanks on Netflix | netflix | netflix ✓ | ✓ | 100% (7/7) | 5/5/5 |
| n3 | Recommend something in Korean | netflix | netflix ✓ | ✓ | 100% (8/8) | 5/5/5 |
| n4 | Classic films from the 80s | netflix | netflix ✓ | ✓ | 62% (5/8) | 5/5/5 |
| n5 | A good documentary | netflix | netflix ✓ | ✓ | 100% (8/8) | 5/5/5 |
| n6 | I want a horror movie | netflix | netflix ✓ | ✓ | 100% (8/8) | 5/5/5 |
| n7 | Give me a romantic comedy | netflix | netflix ✓ | ✓ | 75% (6/8) | 5/5/5 |
| n8 | Something light and funny to cheer me  | netflix | netflix ✓ | ✓ | 100% (8/8) | 5/5/5 |
| e1 | Best action movie out now? | trending | trending ✓ | ✓ | — | 5/5/5 |
| e2 | What's a good action movie? | netflix | netflix ✓ | ✓ | — | 5/5/5 |
| e3 | A good sci-fi film recently released | trending | trending ✓ | ✓ | — | 5/5/5 |
| e4 | Good comedies from the 90s | netflix | netflix ✓ | ✓ | — | 5/5/5 |
| r1 | What's the weather in Madrid? | refusal | refusal ✓ | ✓ | — | — |
| r2 | Help me write a cover letter | refusal | refusal ✓ | ✓ | — | — |
| r3 | What's 2 + 2? | refusal | refusal ✓ | ✓ | — | — |