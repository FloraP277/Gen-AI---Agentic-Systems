# Writing the evaluation set

`questions.sample.jsonl` only works with the small test fixture. For real results, write `questions.jsonl` with **50 questions** about the real regulations you loaded.

For each question, look up the answer yourself on [eCFR](https://www.ecfr.gov/current/title-30/chapter-I) and record:

```json
{"id": "q01", "question": "...", "answer": "the correct answer in one or two sentences",
 "sections": ["75.400"], "type": "single"}
```

Aim for this mix, so the results show where retrieval breaks:

| Type | How many | What it tests |
| --- | --- | --- |
| `single` | 30 | The answer is in one section. Include some questions that use the section number and some that paraphrase without it. |
| `multi` | 10 | The answer needs two or more sections (like the ClearFlow question in Lab 2). List every section needed. |
| `unanswerable` | 10 | The regulations don't answer it. Leave `answer` empty and `sections` as `[]`. The right response is NOT_IN_SOURCES. |

Tips:
- Write questions the way a mine safety officer would ask them, not by copying sentences from the rule. Copied wording makes keyword search look better than it is.
- Write the set **before** you look at any results, and don't edit it to make a configuration look good.
