# RAG Evaluation Report — Indian Parliamentary Data System
**Evaluator:** Claude Sonnet 4.6 | **Total Cases:** 15

---

## Evaluation Table

| Q # | Faithfulness | Answer Relevance | Context Relevance | Critique |
|:---:|:---:|:---:|:---:|:---|
| 1 | 5.0 | 5.0 | 5.0 | Every penalty tier (Sec 57–61) and the 3-year criminal sentence (Sec 82) are verbatim-traceable to the retrieved bill text; a model answer. |
| 2 | 5.0 | 4.5 | 4.5 | The core 7-year filing limit is correctly identified; the answer then adds well-sourced context on penalties and amendments that slightly exceeds the narrow question asked. |
| 3 | 5.0 | 5.0 | 5.0 | Sections 84 and 85 are precisely distinguished—strict liability for warranty vs. negligence standard for services—with every claim directly grounded in the bill. |
| 4 | 4.5 | 4.5 | 4.5 | The full roster of MPs and their points is well-sourced across multiple debates, but the "support vs. opposition" framing in the question has no opposition voice in the context, which the answer does not flag. |
| 5 | 3.5 | 2.5 | 2.0 | The specific 30/07/2019 interruption reason is absent from the context; the answer speculatively analogises to an unrelated GST adjournment debate and presents it as explanatory context, which is misleading. |
| 6 | 5.0 | 4.0 | 1.5 | The system correctly and honestly flags that cryptocurrency is absent from the 2010 Code; however, the retriever returned a categorically irrelevant document (DTC 2010 predates crypto entirely), so no useful answer was ever possible. |
| 7 | 4.0 | 3.5 | 3.5 | IBC timelines are faithfully cited, but the Sahara Group refund precedent is only loosely applicable to a typical startup investor scenario, and the answer does not clearly distinguish guaranteed statutory periods from practical recovery timelines. |
| 8 | 5.0 | 3.0 | 3.0 | "UMMEED Guidelines" are correctly flagged as absent from the corpus; education-cost factors are well answered from what is available, but the second half of the question remains unanswerable due to a retrieval gap. |
| 9 | 4.5 | 5.0 | 4.5 | Legislative evolution from CPA 1986 through CCPA 2020 is clearly and accessibly structured; the CCPA establishment date is stated without an explicit citation in the retrieved text, which is a minor faithfulness concern. |
| 10 | 3.5 | 5.0 | 3.0 | The Hindi-language question is answered in Hindi with correct ministry guidance and FRA provisions, but the answer draws substantially beyond the single constituency-specific debate retrieved, implying broader sourcing than what the context supports. |
| 11 | 4.0 | 5.0 | 3.5 | FRA sections and PESA are correctly enumerated; the primary retrieved chunk is a forest-policy Q&A that names FRA only tangentially, so some FRA section-level detail likely draws on model knowledge rather than retrieved text. |
| 12 | 5.0 | 4.5 | 5.0 | SZCC Thanjavur, SCZCC Nagpur, Guru Shishya Parampara, and NCEP are all sourced directly from the retrieved parliamentary Q&A; minor deduction for responding in English to a Kannada question. |
| 13 | 5.0 | 5.0 | 5.0 | The government's "international collaborative advantage" position, the 80-country agreement count, and the multilateral engagement details are all cited directly from the retrieved Q&A transcript. |
| 14 | 5.0 | 5.0 | 5.0 | The debate transcript is a perfect semantic match; every objection—borewell injection, MPT waste, crop damage, health risk, and employment exclusion—is faithfully reproduced and attributed. |
| 15 | 3.5 | 3.0 | 3.0 | Aviation accident committee questions are present and correctly cited, but the compound "five most recent combining both aviation and EV topics" requirement cannot be met from the limited retrieval; the answer acknowledges this but still overstates completeness. |

---

## Average Scores

| Metric | Calculation | **Average / 5.0** |
|:---|:---|:---:|
| **Faithfulness** | (5.0+5.0+5.0+4.5+3.5+5.0+4.0+5.0+4.5+3.5+4.0+5.0+5.0+5.0+3.5) ÷ 15 = 67.5 ÷ 15 | **4.50** |
| **Answer Relevance** | (5.0+4.5+5.0+4.5+2.5+4.0+3.5+3.0+5.0+5.0+5.0+4.5+5.0+5.0+3.0) ÷ 15 = 64.5 ÷ 15 | **4.30** |
| **Context Relevance** | (5.0+4.5+5.0+4.5+2.0+1.5+3.5+3.0+4.5+3.0+3.5+5.0+5.0+5.0+3.0) ÷ 15 = 58.0 ÷ 15 | **3.87** |

---

## Summary Observations

- **Faithfulness (4.50)** is the system's strongest dimension. The LLM rarely fabricates claims; in almost all cases it either cites correctly or honestly admits the information is missing from context.
- **Answer Relevance (4.30)** is solid but pulled down by two "impossible" questions (Q5, Q8) where the query presupposed information that does not exist in the corpus, and one compound query (Q15) requiring exhaustive listing.
- **Context Relevance (3.87)** is the weakest dimension, reflecting retrieval-layer gaps: Q6 returned a 2010 tax code for a crypto question, Q5 returned an unrelated state-finance debate, and several questions (Q10, Q11) received only peripherally relevant chunks. Improving the retrieval model (e.g., better semantic filtering, query decomposition) would likely produce the largest quality gains.
