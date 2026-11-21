# Experiment log

All comparisons: macro-F1 on the fixed validation split (configs/protocol.yaml).

| ID | Date | Model / change | Reference | Val macro-F1 | F1 human | F1 ai | F1 fusion | Public LB | Notes |
|----|------|----------------|-----------|--------------|----------|-------|-----------|-----------|-------|
| m2_tfidf | 2026-11-21 | TF-IDF word 1-2 + char 3-5 (no lowercase), LogReg C=10 | dummy 0.205 | 0.8816 | 0.8257 | 0.9854 | 0.8338 | 0.87608 | grid: svc 0.860/0.870, logreg 0.881/0.882; ablation char 0.881, word 0.861; random split 0.869 |
| m3_xlmr | 2026-11-21 | XLM-R base, 2 ep, lr 2e-5, bs 16, fp16; train head-512; eval chunks (mean logits, <=8x510) | m2_tfidf 0.8816 | 0.8975 | 0.8368 | 0.9851 | 0.8706 | 0.89004 | head 0.8672, head_tail 0.8831; epoch F1 0.816 -> 0.867; 56 min T4; arxiv 0.655 vs tfidf 0.851 |
