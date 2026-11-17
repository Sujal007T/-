import os
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    'language_info': {'name': 'python', 'version': '3.10'},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
}

cells = []

# Cell 1: Title and Overview
cells.append(nbf.v4.new_markdown_cell("""# DisasterM3 Track B: Comprehensive Ablation Study & Benchmark Evaluation

This notebook visualizes and aggregates the rigorous statistical evaluation of the **DisaSense Track B Vision-Language Model** (`Qwen/Qwen2.5-VL-3B-Instruct` fine-tuned with QLoRA) on the official **DisasterM3 Benchmark** (`DisasterM3_Bench/benchmark_release.json`).

---

### Experimental Conditions Evaluated
| Condition | Configuration | Visual Inputs | Task Prompts | Adapter |
| :---: | :--- | :--- | :--- | :--- |
| **A** | Zero-Shot Base | Post-disaster image only | Plain raw prompt (no task tokens) | None (Base weights) |
| **B** | Task Tokens | Post-disaster image only | `[DISASTER_TYPE]` / `[DAMAGE_REPORT]` system prompt | None (Base weights) |
| **C** | Multi-Scale Patches | Pre + Post + 2x2 local patches | Task tokens system prompt | None (Base weights) |
| **D** | Fine-Tuned Model | Pre + Post + 2x2 local patches | Task tokens system prompt | **QLoRA Adapter Enabled** |
| **K** | Control Baseline | None (No model) | Dataset statistics (Majority class / Central report) | None |

---

### Key Rigor & Integrity Guards
1. **Zero Data Leakage:** The evaluation set is strictly drawn from clean benchmark samples. Any samples sharing images with `track_b_train.jsonl` are automatically filtered out (asserted 0 overlap).
2. **Identical Sample IDs:** All conditions are evaluated on the exact same stratified benchmark IDs saved in `results/track_b/bench_eval_ids.json`.
3. **Paired Statistical Testing:** 1,000-iteration empirical bootstrap hypothesis tests evaluate per-sample differences with 95% confidence intervals.
4. **Template-Copy Baseline (Condition K):** Validates whether the fine-tuned model performs grounded image perception or merely reproduces training report boilerplate.
"""))

# Cell 2: Imports and Setup
cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import pandas as pd
from IPython.display import display, Markdown, Image, HTML

# Project root resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.exists("../configs") else os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "track_b")
print(f"Project root : {PROJECT_ROOT}")
print(f"Results path : {RESULTS_DIR}")
"""))

# Cell 3: Main Comparison Table
cells.append(nbf.v4.new_markdown_cell("""## 1. Main Ablation Comparison Table
Summary of headline performance across conditions with **95% Empirical Bootstrap Confidence Intervals**:
"""))

cells.append(nbf.v4.new_code_cell("""main_csv = os.path.join(RESULTS_DIR, "tables", "main_comparison.csv")
if os.path.exists(main_csv):
    df_main = pd.read_csv(main_csv)
    display(HTML(df_main.to_html(classes="table table-striped table-bordered", index=False)))
else:
    print(f"Main table not found at {main_csv}. Run eval_trackb.py and aggregate_trackb.py first.")
"""))

# Cell 4: Per-Disaster-Type Breakdown
cells.append(nbf.v4.new_markdown_cell("""## 2. Per-Disaster-Type Performance Breakdown
Detailed class-by-class evaluation across disaster categories (e.g., flood, wildfire, earthquake, hurricane):
"""))

cells.append(nbf.v4.new_code_cell("""per_type_csv = os.path.join(RESULTS_DIR, "tables", "per_disaster_type.csv")
if os.path.exists(per_type_csv):
    df_type = pd.read_csv(per_type_csv)
    display(HTML(df_type.to_html(classes="table table-hover table-bordered", index=False)))
else:
    print(f"Per-disaster-type table not found at {per_type_csv}.")
"""))

# Cell 5: Publication Figures
cells.append(nbf.v4.new_markdown_cell("""## 3. High-Resolution Publication Figures (300 DPI)
"""))

cells.append(nbf.v4.new_code_cell("""fig_dir = os.path.join(RESULTS_DIR, "figures")

figures_to_show = [
    ("training_loss_curve.png", "Figure 1: Qwen2.5-VL-3B LoRA Fine-Tuning Convergence Loss Curve"),
    ("compliance_bars.png", "Figure 2: 7-Section Schema Compliance across Conditions (with 95% CI Error Bars)"),
    ("rouge_l_bars.png", "Figure 3: Damage Report ROUGE-L Perception Quality across Conditions (with 95% CI Error Bars)"),
    ("confusion_matrices_dtr.png", "Figure 4: Disaster Type Recognition Confusion Matrices"),
    ("failure_gallery.png", "Figure 5: 12-Case Qualitative Analysis Gallery (Condition D Win/Loss Audit)")
]

for fname, title in figures_to_show:
    fpath = os.path.join(fig_dir, fname)
    if os.path.exists(fpath):
        display(Markdown(f"### {title}"))
        display(Image(filename=fpath, width=800))
    else:
        print(f"Figure {fname} not yet generated.")
"""))

# Cell 6: Paired Statistical Claims & Hypothesis Verification
cells.append(nbf.v4.new_markdown_cell("""## 4. Paired Hypothesis Tests & Template-Copy Check
Summary of empirical difference tests with 95% Confidence Intervals:
"""))

cells.append(nbf.v4.new_code_cell("""claims_md_path = os.path.join(RESULTS_DIR, "claims_check.md")
if os.path.exists(claims_md_path):
    with open(claims_md_path, "r", encoding="utf-8") as f:
        claims_text = f.read()
    display(Markdown(claims_text))
else:
    print(f"Claims document not found at {claims_md_path}.")
"""))

# Cell 7: Consolidated Summary JSON
cells.append(nbf.v4.new_markdown_cell("""## 5. Consolidated Manifest & Summary
"""))

cells.append(nbf.v4.new_code_cell("""summary_json_path = os.path.join(RESULTS_DIR, "track_b_ablation_summary.json")
if os.path.exists(summary_json_path):
    with open(summary_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    print(json.dumps({k: v for k, v in summary_data.items() if k != 'conditions'}, indent=2))
"""))

nb['cells'] = cells
nb_path = 'd:/Sem-7/Minor/Project/disaSense_project/notebooks/06_trackb_results.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated notebook at: {nb_path}")

