# Experiment log

All comparisons: macro-F1 on the fixed validation split (configs/protocol.yaml).

| ID | Date | Model / change | Reference | Val macro-F1 | F1 human | F1 ai | F1 fusion | Public LB | Notes |
|----|------|----------------|-----------|--------------|----------|-------|-----------|-----------|-------|
| m2_tfidf | 2026-11-21 | TF-IDF word 1-2 + char 3-5 (no lowercase), LogReg C=10 | dummy 0.205 | 0.8816 | 0.8257 | 0.9854 | 0.8338 | 0.87608 | grid: svc 0.860/0.870, logreg 0.881/0.882; ablation char 0.881, word 0.861; random split 0.869 |
| m3_xlmr | 2026-11-21 | XLM-R base, 2 ep, lr 2e-5, bs 16, fp16; train head-512; eval chunks (mean logits, <=8x510) | m2_tfidf 0.8816 | 0.8975 | 0.8368 | 0.9851 | 0.8706 | 0.89004 | head 0.8672, head_tail 0.8831; epoch F1 0.816 -> 0.867; 56 min T4; arxiv 0.655 vs tfidf 0.851 |
| m4_E1_ensemble | 2026-11-21 | 0.45 * XLM-R chunks + 0.55 * TF-IDF (probabilities) | m3_xlmr 0.8975 | 0.9180 (cross-fit) | 0.8724 | 0.9910 | 0.8904 | 0.91661 | w=0.5 untuned 0.9182; optimistic 0.9192; errors human<->fusion 303/308 |
| m4_E2_weights | 2026-11-21 | class weights on XLM-R (t_human~2, t_fusion~0.45) | m3_xlmr 0.8975 | 0.9121 (cross-fit) | 0.8701 | 0.9879 | 0.8783 | - | fixes XLM-R fusion bias; weights unstable between halves |
| m4_E1E2 | 2026-11-21 | ensemble + class weights | m4_E1 0.9180 | 0.9176 (cross-fit) | 0.8761 | 0.9911 | 0.8855 | - | FAILED: no gain on top of ensemble (bias already cancelled), optimistic 0.9207 = overfit |
| m4_tfidf_pool | 2026-11-22 | TF-IDF (same config) trained on 150k pool texts instead of 50k | m2_tfidf 0.8816 | 0.9019 | 0.8554 | 0.9872 | 0.8631 | - | 3x data = +0.020; gutenberg unchanged 0.740 |
| m4_E1_pool | 2026-11-22 | 0.45 * XLM-R chunks + 0.55 * TF-IDF pool | m4_E1 0.9180 | 0.9267 (cross-fit) | 0.8885 | 0.9914 | 0.9001 | 0.92636 | class weights on top: 0.9265 (no gain again); arxiv 0.746 vs tfidf alone 0.886 |
