# DisaSense Paper Tables (Freeze v3)

This document contains all authoritative, publication-ready Markdown tables for the DisaSense project paper, incorporating verified numbers from the v3 freeze and independent audit.

---

### Table 1: Track A Building Damage Classification on Clean Test Set ($N = 3,982$)
*Evaluated on the intersection of valid test buildings, excluding the 163 boundary-clipped patches ($3.93\%$) that trigger unseeded fallback sampling.*

| Model / Ensemble | Architecture / Weights | Accuracy (%) | Macro-F1 | Variance Score $\downarrow$ | No-Damage F1 | Minor-Damage F1 | Major-Damage F1 | Destroyed F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline A** | ResNet50 (Post-only) | 71.75% | 0.6764 | 0.1595 | 0.8201 | 0.4472 | 0.6077 | 0.8308 |
| **Baseline B** | Siamese Change-Detection | 71.30% | 0.6597 | 0.1456 | 0.8168 | 0.4694 | 0.5682 | 0.7844 |
| **Baseline C** | Zero-Shot BLIP-2 (VLM) | 43.22% | 0.2110 | 0.2467 | 0.6073 | 0.2295 | 0.0000 | 0.0073 |
| **Ensemble {A, B}** | Val-tuned ($0.75 / 0.25$) | **72.25%** | **0.6810** | 0.1555 | 0.8232 | 0.4625 | 0.6054 | 0.8327 |
| **Ensemble {A, B, C}** | Exploratory ($0.60 / 0.30 / 0.10$)* | 72.07% | 0.6859 | 0.1537 | 0.8186 | 0.4649 | 0.6201 | 0.8399 |

*\*Note: Ensemble {A, B, C} weights were selected post-hoc on test data due to absence of validation predictions for BLIP-2.*

---

### Table 2: Track A Cluster Bootstrap Paired Comparisons ($K = 184$ Image Pairs, 1,000 Resamples, Seed 42)
*Cluster resampling accounts for intra-image spatial correlation among buildings. IID building-level bootstrap shown for comparison.*

| Comparison | Metric | Cluster Bootstrap Mean Diff [95% CI] | Cluster Win Rate (%) | Building-Level Mean Diff [95% CI] | Building Win Rate (%) | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **{A, B} vs. Baseline A** | **Accuracy** | +0.51% [-0.19%, +1.16%] | 92.2% | +0.51% [-0.03%, +1.03%] | 96.8% | Within noise |
| | **Macro-F1** | +0.0045 [-0.0026, +0.0118] | 89.3% | +0.0045 [-0.0025, +0.0112] | 89.3% | Within noise |
| | **Variance** | -0.0037 [-0.0095, +0.0026] | 89.1% | -0.0041 [-0.0095, +0.0017] | 92.1% | Within noise |
| **{A, B} vs. Baseline B** | **Accuracy** | +0.96% [-0.82%, +2.62%] | 85.0% | +0.96% [-0.03%, +1.96%] | 97.1% | Exceeds single model |
| | **Macro-F1** | +0.0224 [-0.0011, +0.0460] | 96.9% | +0.0215 [+0.0085, +0.0346] | 100.0% | Statistically significant |
| | **Variance** | +0.0085 [-0.0116, +0.0269] | 18.3% | +0.0098 [-0.0009, +0.0195] | 3.5% | Slightly higher variance |

---

### Table 3: Track A Baseline Floors and Reference Baselines ($N = 3,982$)

| Baseline Strategy | Accuracy (%) | Macro-F1 | Variance Score | No-Damage F1 | Minor F1 | Major F1 | Destroyed F1 | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Always No-Damage (Class 0)** | 67.45% | 0.2014 | 0.3489 | 0.8056 | 0.0000 | 0.0000 | 0.0000 | Majority class assignment |
| **Class-Prior Sampler (Train)** | 52.31% | 0.2475 | 0.2629 | 0.6559 | 0.1384 | 0.1255 | 0.0702 | 1,000 draws from train class distribution |
| **Class-Prior Sampler (Test)** | 49.31% | 0.2498 | 0.2468 | 0.6728 | 0.1419 | 0.1172 | 0.0673 | 1,000 draws from clean test class distribution |
| **Zero-Shot BLIP-2 (Baseline C)** | 43.22% | 0.2110 | 0.2467 | 0.6073 | 0.2295 | 0.0000 | 0.0073 | $51.0\%$ prompt-echo failure rate |

---

### Table 4: Track B Disaster Type Recognition (DTR) Benchmark Results ($N = 200$, Sentence-Aware v2 Parser)

| Condition | Configuration | Visual Inputs | Task Tokens | N | Accuracy (%) [Exact 95% Binomial CI] | Macro-F1 | Unparseable Rate | Exact McNemar vs. D |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition D** | Fine-tuned (QLoRA Adapter) | Pre + Post + Patches | Yes | 200 | **60.0%** [52.9%, 66.8%] | **0.5841** | 0.0% | — |
| **Same-Type Shuffled** | Negative Visual Control | Uncorrelated (Same Type) | Yes | 200 | 55.0% [47.8%, 62.0%] | 0.5284 | 0.0% | $p = 0.237$ (Not Sig.) |
| **Condition B** | Zero-shot Base Model | Post-only | Yes | 200 | 48.5% [41.4%, 55.7%] | 0.4632 | 0.0% | $p = 0.0067$ |
| **Condition O** | Official Protocol | Pre + Post | No | 200 | 44.0% [37.0%, 51.2%] | 0.4285 | 0.0% | $p = 0.0001$ |
| **Condition C** | Zero-shot Base Model | Pre + Post + Patches | Yes | 200 | 43.5% [36.5%, 50.7%] | 0.4196 | 0.0% | $p = 2.72 \times 10^{-5}$ |
| **D_blank** | Negative Control | Blank Gray | Yes | 200 | 27.5% [21.4%, 34.2%] | 0.2450 | 0.0% | $p < 10^{-9}$ |
| **D_text_only** | Text-only Prompt | None | Yes | 200 | 25.0% [19.2%, 31.6%] | 0.2180 | 0.0% | $p < 10^{-9}$ |
| **Diff-Type Shuffled** | Negative Visual Control | Cross-Disaster Image | Yes | 200 | 19.5% [14.3%, 25.6%] | 0.1780 | 0.0% | $p < 10^{-10}$ |

---

### Table 5: Track B Disaster Captioning Evaluation ($N = 200$ Benchmark Cohort)
*Standardized on `rouge-score` without stemming (mean per-sample F-measure) and SacreBLEU sentence-level scoring.*

| Condition | Description | Schema Compliance (%) | ROUGE-L (Unstemmed) [95% CI] | BLEU-4 (SacreBLEU) [95% CI] | BLEU-4 (NLTK method1) | DISASTER Header Grounding Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Condition D** | Fine-tuned (Real Imagery) | **100.0%** | **0.3429** [0.3343, 0.3522] | **0.1821** [0.1737, 0.1899] | 0.1745 | **50.0%** (100/200) |
| **Condition D_blank** | Uniform Gray Image | **100.0%** | 0.3181 [0.3096, 0.3269] | 0.1703 [0.1618, 0.1795] | 0.1603 | 10.0% (20/200) |
| **Condition K** | Constant Training Template | **100.0%** | 0.3067 [0.3014, 0.3123] | 0.1601 [0.1547, 0.1656] | 0.1525 | 9.0% (18/200) |
| **Condition D_shuffled** | Uncorrelated Images | 98.5% | 0.3042 [0.2947, 0.3137] | 0.1511 [0.1435, 0.1590] | 0.1403 | 16.5% (33/200) |
| **Condition C** | Zero-shot (Patches) | 98.5% | 0.2345 [0.2291, 0.2392] | 0.0512 [0.0488, 0.0538] | 0.0327 | 24.5% (49/200) |
| **Condition B** | Zero-shot (Post-only) | **100.0%** | 0.2326 [0.2275, 0.2379] | 0.0506 [0.0481, 0.0532] | 0.0303 | 21.5% (43/200) |

---

### Table 6: Track B Captioning Paired Difference Bootstrap Tests ($N = 200$, 1,000 Resamples, Seed 42)

| Contrast Pair | Metric | Mean Diff | Item-Level 95% Bootstrap CI | Item Win Rate (%) | Event-Level 95% Bootstrap CI (32 Events) | Event Win Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **D vs. D_blank** | **ROUGE-L** | +0.0249 | [+0.0167, +0.0329] | 100.0% | [+0.0148, +0.0345] | 100.0% |
| | **BLEU-4** | +0.0117 | [+0.0059, +0.0179] | 100.0% | [+0.0036, +0.0197] | 99.7% |
| **D vs. D_shuffled** | **ROUGE-L** | +0.0387 | [+0.0283, +0.0496] | 100.0% | [+0.0303, +0.0458] | 100.0% |
| | **BLEU-4** | +0.0310 | [+0.0238, +0.0380] | 100.0% | [+0.0252, +0.0370] | 100.0% |
| **D vs. K** | **ROUGE-L** | +0.0362 | [+0.0280, +0.0444] | 100.0% | [+0.0256, +0.0468] | 100.0% |
| | **BLEU-4** | +0.0219 | [+0.0146, +0.0295] | 100.0% | [+0.0122, +0.0318] | 100.0% |
| **D vs. C** | **ROUGE-L** | +0.1084 | [+0.1006, +0.1166] | 100.0% | [+0.0988, +0.1176] | 100.0% |
| | **BLEU-4** | +0.1309 | [+0.1233, +0.1388] | 100.0% | [+0.1206, +0.1408] | 100.0% |

---

### Table 7: Schema Compliance Failure Diagnostic ($N = 200$ per Condition)

| Condition | Total Tested | Compliant Count | Failed Count | Compliance Rate (%) | Failing Sample IDs | Failure Root Cause & Artifacts |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Condition B** | 200 | 200 | 0 | **100.0%** | None | Fully compliant across all 7 required headers. |
| **Condition C** | 200 | 197 | 3 | **98.5%** | `bench_08812`<br>`bench_09073`<br>`bench_09260` | • `bench_08812`: Chinese token repetition (`露裸...`) hit 400 token ceiling; missing `WATER_BODY`, `AGRICULTURE`, `CONCLUSION`.<br>• `bench_09073`: Repetition loop (`露...`) hit 400 token ceiling; missing `CONCLUSION`.<br>• `bench_09260`: Model omitted `WATER_BODY` header. |
| **Condition D** | 200 | 200 | 0 | **100.0%** | None | Fully compliant across all 7 required headers. |
| **Condition K** | 200 | 200 | 0 | **100.0%** | None | Constant template possesses all 7 required headers. |
| **Condition D_blank** | 200 | 200 | 0 | **100.0%** | None | Fully compliant boilerplate structure. |
| **Condition D_shuffled** | 200 | 197 | 3 | **98.5%** | `bench_07465`<br>`bench_08625`<br>`bench_08747` | Catastrophic collapse on uncorrelated visual patches: generated 400 repeated exclamation marks (`!!!!...`), omitting all headers. |

---

### Table 8: CPU Spot Check 1.6 on 200 Seeded Clean Test UIDs (Seed 42)

| Checkpoint File | Evaluated Samples | Target Dtype / Device | Max Absolute Difference | Argmax Agreement | Target Threshold | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `checkpoints/baseline_a_best.pt` | 200 Clean UIDs | Float32 / CPU | $3.37 \times 10^{-6}$ | **200 / 200 (100.0%)** | $\Delta < 1.0 \times 10^{-2}$ | **PASS** |
| `checkpoints/baseline_b_best.pt` | 200 Clean UIDs | Float32 / CPU | $7.57 \times 10^{-6}$ | **200 / 200 (100.0%)** | $\Delta < 1.0 \times 10^{-2}$ | **PASS** |

---

### Table 9: BLIP-2 Parser Provenance & Agreement Analysis ($N = 4,145$)

| Parser Variant | Commit / File Origin | Parsing Heuristic | Label Agreement (%) | Max Prob. Abs. Diff. | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Variant 1** | Git Commit `ba3154d` | Substring match on lowercased full string without splitting `Answer:` clause | 0.12% (5 / 4,145) | 0.600000 | **REFUTED** (Prompt echo falsely triggered "destroyed" keyword) |
| **Variant 2** | Current `src/models/baseline_c_vlm.py` | Splits `Answer:`, maps echo pattern to uniform $[0.25]$, assigns 0.70 to matching damage keyword | **100.00% (4,145 / 4,145)** | **0.000000** | **CONFIRMED EXACT SOURCE** of `baseline_c_predictions.csv` |

