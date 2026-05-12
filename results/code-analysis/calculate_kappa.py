import pandas as pd
from sklearn.metrics import cohen_kappa_score

# === LOAD FILES ===
judged = pd.read_csv("all_results.csv")

rabbi_std   = pd.read_csv("human_annotated/standard_rabbi.csv")
jelle_std   = pd.read_csv("human_annotated/standard_jelle.csv")
rabbi_think = pd.read_csv("human_annotated/think_rabbi.csv")
jelle_think = pd.read_csv("human_annotated/think_jelle.csv")

# === SPLIT JUDGED BY TYPE ===
# Standard = has completion but no reasoning_trace
# Think    = has reasoning_trace
judged_std   = judged[judged['reasoning_trace'].isna()][['id', 'age_condition', 'annotation_new']].copy()
judged_think = judged[judged['reasoning_trace'].notna()][['id', 'age_condition', 'annotation_new']].copy()

KEY = ['id', 'age_condition']

def calc_kappa(judged_subset, human_df, label_human, label_name):
    merged = judged_subset.merge(
        human_df[KEY + ['final_label']],
        on=KEY
    )
    missing = len(judged_subset) - len(merged)
    if missing > 0:
        print(f"  ⚠️  {missing} rows not matched for {label_name}")

    kappa = cohen_kappa_score(merged['annotation_new'], merged['final_label'])
    agree = (merged['annotation_new'] == merged['final_label']).mean()
    print(f"  {label_name}")
    print(f"    Matched:     {len(merged)}")
    print(f"    Agreement:   {agree:.1%}")
    print(f"    Kappa:       {kappa:.4f}")
    return merged

print("=== STANDARD ===")
m1 = calc_kappa(judged_std, rabbi_std,   'final_label', 'judged vs Rabbi')
m2 = calc_kappa(judged_std, jelle_std,   'final_label', 'judged vs Jelle')

print("\n=== THINK ===")
m3 = calc_kappa(judged_think, rabbi_think, 'final_label', 'judged vs Rabbi')
m4 = calc_kappa(judged_think, jelle_think, 'final_label', 'judged vs Jelle')
