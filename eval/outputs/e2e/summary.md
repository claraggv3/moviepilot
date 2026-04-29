# Eval summary

Generated: 2026-04-28 17:55  
Cases run: 20  

## Aggregate scores

| Metric | Score |
|---|---|
| Routing accuracy | 20/20 (100%) |
| L1 groundedness  | 19/20 (95%) |
| Retrieval recall (predicate-based) | 87.5% |

## Per-case results

| ID | Query | Expected | Actual | Grounded | Recall |
|---|---|---|---|---|---|
| t1 | What's popular right now? | trending | trending ✓ | ✓ | — |
| t2 | Best movies out this week | trending | trending ✓ | ✓ | — |
| t3 | Is there anything good at the cinema c | trending | trending ✓ | ✓ | — |
| t4 | What's everyone watching right now? | trending | trending ✓ | ✓ | — |
| t5 | Any good movies released lately? | trending | trending ✓ | ✓ | — |
| n1 | Show me Christopher Nolan films on Net | netflix | netflix ✓ | ✓ | 100% (3/3) |
| n2 | Movies with Tom Hanks on Netflix | netflix | netflix ✓ | ✓ | 100% (7/7) |
| n3 | Recommend something in Korean | netflix | netflix ✓ | ✓ | 100% (8/8) |
| n4 | Classic films from the 80s | netflix | netflix ✓ | ✓ | 62% (5/8) |
| n5 | A good documentary | netflix | netflix ✓ | ✗ (1) | 100% (8/8) |
| n6 | I want a horror movie | netflix | netflix ✓ | ✓ | 100% (8/8) |
| n7 | Give me a romantic comedy | netflix | netflix ✓ | ✓ | 75% (6/8) |
| n8 | Something light and funny to cheer me  | netflix | netflix ✓ | ✓ | 62% (5/8) |
| e1 | Best action movie out now? | trending | trending ✓ | ✓ | — |
| e2 | What's a good action movie? | netflix | netflix ✓ | ✓ | — |
| e3 | A good sci-fi film recently released | trending | trending ✓ | ✓ | — |
| e4 | Good comedies from the 90s | netflix | netflix ✓ | ✓ | — |
| r1 | What's the weather in Madrid? | refusal | refusal ✓ | ✓ | — |
| r2 | Help me write a cover letter | refusal | refusal ✓ | ✓ | — |
| r3 | What's 2 + 2? | refusal | refusal ✓ | ✓ | — |

## Notable cases

### Hallucination — `n5`
> A good documentary

Titles in response not found in context: `['"The Game Changers"']`

**Response:** I recommend **"The Game Changers" (2019)**. This documentary explores the diets of elite athletes and challenges the traditional views on nutrition and performance. It features a diverse range of individuals, from UFC fighters to cultural icons, all advocating for a plant-based diet and its benefits. It's both informative and inspiring, making it a great choice for anyone interested in health and 
