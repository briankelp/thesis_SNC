#!/usr/bin/env python3
"""
Compute confusion matrix and metrics from hardware test CSV.
Usage: python analyse_results.py --csv hardware_results.csv
"""
import argparse
import pandas as pd
import numpy as np

def compute_metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc  = (tp + tn) / total if total > 0 else 0
    prec = tp / (tp + fp)    if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn)    if (tp + fn) > 0 else 0
    fpr  = fp / (tn + fp)    if (tn + fp) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, rec, fpr, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["ground_truth"].isin([0, 1])]  # skip rows without ground truth

    print(f"\nLoaded {len(df)} results from {args.csv}")
    print(f"Scenarios: {df['scenario'].unique().tolist()}\n")

    # ── Overall confusion matrix ───────────────────────────────────────────
    tp = ((df["ground_truth"] == 1) & (df["prediction"] == 1)).sum()
    tn = ((df["ground_truth"] == 0) & (df["prediction"] == 0)).sum()
    fp = ((df["ground_truth"] == 0) & (df["prediction"] == 1)).sum()
    fn = ((df["ground_truth"] == 1) & (df["prediction"] == 0)).sum()

    acc, prec, rec, fpr, f1 = compute_metrics(tp, tn, fp, fn)

    print("=" * 55)
    print("  OVERALL HARDWARE RESULTS")
    print("=" * 55)
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  FPR:       {fpr:.4f} ({fpr*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print()
    print("  Confusion Matrix:")
    print(f"                   Predicted Benign   Predicted Attack")
    print(f"  Actual Benign       {tn:<20} {fp}")
    print(f"  Actual Attack       {fn:<20} {tp}")
    print("=" * 55)

    # ── Per-scenario breakdown ─────────────────────────────────────────────
    print("\n  PER-SCENARIO BREAKDOWN")
    print("=" * 55)
    print(f"  {'Scenario':<25} {'Runs':>5} {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} {'Acc':>7} {'Recall':>8}")
    print("-" * 55)

    for scenario, group in df.groupby("scenario"):
        stp = ((group["ground_truth"] == 1) & (group["prediction"] == 1)).sum()
        stn = ((group["ground_truth"] == 0) & (group["prediction"] == 0)).sum()
        sfp = ((group["ground_truth"] == 0) & (group["prediction"] == 1)).sum()
        sfn = ((group["ground_truth"] == 1) & (group["prediction"] == 0)).sum()
        sacc, _, srec, _, _ = compute_metrics(stp, stn, sfp, sfn)
        total = len(group)
        print(f"  {scenario:<25} {total:>5} {stp:>4} {stn:>4} {sfp:>4} {sfn:>4} {sacc:>7.2%} {srec:>8.2%}")

    print("=" * 55)

    # ── Average inference time ─────────────────────────────────────────────
    avg_ms = df["elapsed_ms"].mean()
    print(f"\n  Avg prediction latency: {avg_ms:.1f} ms")
    print(f"  Min: {df['elapsed_ms'].min():.1f} ms   Max: {df['elapsed_ms'].max():.1f} ms\n")

if __name__ == "__main__":
    main()