# Main Pragmatic Detection

Status: `partial`

| System | implicit_sentiment | sarcasm | irony | idiom_figurative | code_switching | mocking | macro_pragmatic_f1 | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-4.1-mini 8-shot |  |  |  |  |  |  |  | blocked |
| GPT-4.1-mini zero-shot |  |  |  |  |  |  |  | blocked |
| Local heuristic smoke baseline | 51.50 [51.50, 51.50] | 49.26 [49.26, 49.26] | 45.92 [45.92, 45.92] | 47.59 [47.59, 47.59] | 50.67 [50.67, 50.67] | 48.34 [48.34, 48.34] | 48.88 [48.88, 48.88] | complete |
| PhoBERT fine-tune |  |  |  |  |  |  |  | blocked |
| PhoBERT (single-task) |  |  |  |  |  |  |  | blocked |
| Sailor-7B SFT |  |  |  |  |  |  |  | blocked |
| ViPragSent (ours) |  |  |  |  |  |  |  | blocked |
| Vistral-7B SFT |  |  |  |  |  |  |  | blocked |
| XLM-R-large |  |  |  |  |  |  |  | blocked |
