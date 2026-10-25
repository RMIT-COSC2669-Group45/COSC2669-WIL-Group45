# Saved experiment outputs (E6–E7.1)

Generated answers from the intermediate RAG experiments described in the written report
(Section 3.4 and Appendix C). Each file contains one JSON record per evaluation question
(K01–K08, I01–I12) with the question, the generated answer and the five retrieved chunks.

| File | Experiment | What changed |
|---|---|---|
| `rag_answers_ruleaware.jsonl` | E6 | Rule-aware generation prompt |
| `rag_answers_e7_structured.jsonl` | E7 | Structured evidence extraction (RULE / EXCEPTIONS / CONDITIONS / NUMBERS/DATES / SOURCE) |
| `rag_answers_e7_1_sourcefocused.jsonl` | E7.1 | Source-focused extraction |

Notes:
- E6, E7 and E7.1 all use the same lexical evidence selection introduced in E5, so their retrieved chunks are identical and differences come from the prompt/generation stage only.
- The final E7.2 (topic-aware) outputs are in the parent folder: `../rag_answers_e7_2_topicaware.jsonl`.
- E2–E5 and E7.3 were inspected in terminal output only and were not saved.
