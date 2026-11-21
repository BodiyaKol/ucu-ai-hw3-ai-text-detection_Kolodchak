# Experiment log

All comparisons: macro-F1 on the fixed validation split (configs/protocol.yaml).

| ID | Date | Model / change | Reference | Val macro-F1 | F1 human | F1 ai | F1 fusion | Public LB | Notes |
|----|------|----------------|-----------|--------------|----------|-------|-----------|-----------|-------|
| m2_tfidf | 2026-11-21 | TF-IDF word 1-2 + char 3-5 (no lowercase), LogReg C=10 | dummy 0.205 | 0.8816 | 0.8257 | 0.9854 | 0.8338 | 0.87608 | grid: svc 0.860/0.870, logreg 0.881/0.882; ablation char 0.881, word 0.861; random split 0.869 |
