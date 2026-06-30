"""
Minimal standalone test: load CRISPR_BERT and run inference on a single
on-target / off-target pair, without touching Cas-OFFinder at all.

Run from inside the crispr-bert-resource directory:
    python test_model_inference.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crismer_bert_modules import CRISMER_BERT

def main():
    print("=== Initializing CRISMER_BERT (loading weights, scaler, bins) ===")
    crismer = CRISMER_BERT()
    print(f"Scenario detected: {crismer.scenario}")
    print(f"Weights loaded: {crismer.model is not None}")

    # A real 23nt sgRNA target and a synthetic off-target with a couple of mismatches
    on_target = "CGTGCGCAGGAGGACGAGGANGG"   # 20nt + NGG PAM, adjust to your real sgRNA
    off_target = "CGTGCGCAGGAGGACGAGGATGG"  # identical except PAM base, sanity check

    print("\n=== Running single_score_ on a hand-picked pair ===")
    score = crismer.single_score_(on_target, off_target)
    print(f"On-target:  {on_target}")
    print(f"Off-target: {off_target}")
    print(f"CRISMER-Score: {score}")

    print("\n=== Running score() on a small batch ===")
    import pandas as pd
    df = pd.DataFrame({
        'On':  [on_target, on_target, on_target],
        'Off': [off_target, "CGTGCGCAGGAGGACGAGGAAGG", "TTTTCGCAGGAGGACGAGGATGG"]
    })
    scores = crismer.score(df)
    print(df.assign(score=scores))

    print("\n=== If you see numeric scores above with no errors, model + inference path is healthy ===")

if __name__ == "__main__":
    main()