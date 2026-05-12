import pandas as pd

# ============================================================================
# LOAD
# ============================================================================
df = pd.read_csv('all_results.csv')

print(f"Total rows: {len(df)}")
print(df['age_condition'].value_counts())

# ============================================================================
# STRATIFIED SAMPLE — 25 per age condition = 100 total
# Same sample for both files (same random_state!)
# ============================================================================
sample = (
    df.groupby('age_condition', group_keys=False)
    .apply(lambda x: x.sample(n=min(25, len(x)), random_state=42))
    .reset_index(drop=True)
)

# ============================================================================
# EXPORT 1 — completion (final response)
# ============================================================================
cols_completion = ['id', 'type', 'age_condition', 'prompt', 'completion', 'llm_annotation']
cols_completion = [c for c in cols_completion if c in sample.columns]

sample[cols_completion].to_csv('kappa_sample_completion.csv', index=False)
print(f"\nkappa_sample_completion.csv saved: {len(sample)} rows")

# ============================================================================
# EXPORT 2 — reasoning_trace (think section)
# ============================================================================
cols_think = ['id', 'type', 'age_condition', 'prompt', 'reasoning_trace', 'llm_annotation']
cols_think = [c for c in cols_think if c in sample.columns]

sample[cols_think].to_csv('kappa_sample_think.csv', index=False)
print(f"kappa_sample_think.csv saved: {len(sample)} rows")

print("\nAge condition distribution:")
print(sample['age_condition'].value_counts())
