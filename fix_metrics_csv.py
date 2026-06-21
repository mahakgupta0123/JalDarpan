"""
PATCH for fix_metrics_csv.py
Replace the entire FIX A+B block with this code.
This computes Persistence KGE from PBIAS (no Alpha/Beta needed)
and keeps the existing recompute_kge logic for RF and XGBoost.
"""

import numpy as np
import pandas as pd
import argparse, os, sys, shutil

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    default="research_metrics_append.csv")
    p.add_argument("--output",   default=None)
    p.add_argument("--beta-cap", type=float, default=10.0)
    p.add_argument("--no-backup", action="store_true")
    return p.parse_args()

def recompute_kge(pearson_r, alpha, beta, beta_cap):
    try:
        beta = float(beta); alpha = float(alpha); pearson_r = float(pearson_r)
    except (ValueError, TypeError):
        return np.nan
    if any(np.isnan(x) for x in [beta, alpha, pearson_r]):
        return np.nan
    if abs(beta) > beta_cap:
        return np.nan
    return float(1.0 - np.sqrt((pearson_r-1)**2 + (alpha-1)**2 + (beta-1)**2))

def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found"); sys.exit(1)

    output_path = args.output or args.input
    if not args.no_backup and output_path == args.input:
        shutil.copy2(args.input, args.input + ".bak")
        print(f"  Backup: {args.input}.bak")

    print(f"\nLoading {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # ── FIX A: Persistence KGE from PBIAS (no Alpha/Beta needed) ─────────────
    # For a constant predictor: r=0, alpha=0 by definition.
    # KGE = 1 - sqrt((r-1)² + (alpha-1)² + (beta-1)²)
    #      = 1 - sqrt(1      + 1          + (PBIAS/100)²)
    #      = 1 - sqrt(2 + (PBIAS/100)²)
    # beta = mean(pred)/mean(true) = 1 + PBIAS/100
    print("\nFIX A: Persistence KGE from PBIAS (constant predictor formula)")

    pbias_col = "Persistence__PBIAS"
    kge_col   = "Persistence__KGE"

    if pbias_col in df.columns and kge_col in df.columns:
        df[pbias_col] = pd.to_numeric(df[pbias_col], errors="coerce")
        df[kge_col]   = pd.to_numeric(df[kge_col],   errors="coerce")

        pbias_vals      = df[pbias_col]
        beta_from_pbias = 1.0 + pbias_vals / 100.0

        # Skip degenerate districts (beta overflow means mean_true ≈ 0)
        valid = pbias_vals.notna() & (beta_from_pbias.abs() <= args.beta_cap)

        kge_computed = 1.0 - np.sqrt(2.0 + (pbias_vals / 100.0) ** 2)
        kge_computed[~valid] = np.nan

        before_nan = int(df[kge_col].isna().sum())
        nan_mask   = df[kge_col].isna()
        df.loc[nan_mask & valid, kge_col] = kge_computed[nan_mask & valid]
        after_nan  = int(df[kge_col].isna().sum())

        print(f"  NaN before: {before_nan} → after: {after_nan}")
        print(f"  Values recovered: {before_nan - after_nan}")

        valid_kge = df.loc[valid & df[kge_col].notna(), kge_col]
        if len(valid_kge):
            print(f"  Persistence KGE  mean={valid_kge.mean():.4f}  "
                  f"min={valid_kge.min():.4f}  max={valid_kge.max():.4f}")
            print(f"  (Expected range: roughly -0.41 to -1.5 for typical PBIAS)")
    else:
        print(f"  SKIP: columns {pbias_col} or {kge_col} not found")

    # ── FIX B: RF and XGBoost KGE from stored Alpha/Beta/Pearson r ───────────
    print("\nFIX B: RF and XGBoost KGE from stored components")

    for model in ["Random Forest", "XGBoost"]:
        kge_col    = f"{model}__KGE"
        alpha_col  = f"{model}__Alpha"
        beta_col   = f"{model}__Beta"
        pearson_col= f"{model}__Pearson r"

        if kge_col not in df.columns:
            print(f"  SKIP {model}: KGE column missing"); continue

        for c in [kge_col, alpha_col, beta_col, pearson_col]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        old_nan = int(df[kge_col].isna().sum())
        new_kge = []
        for _, row in df.iterrows():
            if float(row.get("rows", 0) or 0) == 0:
                new_kge.append(np.nan); continue
            kge = recompute_kge(
                pearson_r = row.get(pearson_col, np.nan),
                alpha     = row.get(alpha_col,   np.nan),
                beta      = row.get(beta_col,    np.nan),
                beta_cap  = args.beta_cap,
            )
            new_kge.append(kge)
        df[kge_col] = new_kge
        new_nan = int(pd.Series(new_kge).isna().sum())
        print(f"  {model:20s}: NaN {old_nan} → {new_nan}  "
              f"(recovered {max(old_nan-new_nan,0)})")

    # ── FIX C: Rename NSE → mNSE if needed ───────────────────────────────────
    print("\nFIX C: NSE → mNSE column rename")
    for model in ["Persistence", "Random Forest", "XGBoost"]:
        old = f"{model}__NSE";  new = f"{model}__mNSE"
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
            print(f"  Renamed: {old} → {new}")
        elif new in df.columns:
            print(f"  OK: {new} already exists")

    # ── FIX D: Confusion matrix prefix xgb_cm__ → XGBoost_cm__ ──────────────
    print("\nFIX D: xgb_cm__ → XGBoost_cm__ prefix")
    old_cm = [c for c in df.columns if c.startswith("xgb_cm__")]
    if old_cm:
        df.rename(columns={c: c.replace("xgb_cm__","XGBoost_cm__") for c in old_cm}, inplace=True)
        print(f"  Renamed {len(old_cm)} columns")
    else:
        print("  OK: no old xgb_cm__ columns found")

    # ── Preview ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PREVIEW — Updated metrics (valid districts only)")
    print("=" * 55)
    valid_df = df[pd.to_numeric(df.get("rows",0), errors="coerce") > 0]
    rows = []
    for model in ["Persistence", "Random Forest", "XGBoost"]:
        r = {"Model": model}
        for metric in ["NRMSE","R2","mNSE","KGE"]:
            col = f"{model}__{metric}"
            if col in valid_df.columns:
                vals = pd.to_numeric(valid_df[col], errors="coerce").dropna()
                r[f"{metric} (mean)"] = round(vals.mean(),6) if len(vals) else np.nan
        r["Valid rows"] = len(valid_df)
        rows.append(r)
    print(pd.DataFrame(rows).set_index("Model").to_string(float_format="{:.6f}".format))

    # Persistence KGE specifically
    p_kge = pd.to_numeric(valid_df.get("Persistence__KGE", pd.Series(dtype=float)),
                          errors="coerce").dropna()
    print(f"\n  Persistence KGE: {len(p_kge)} valid values")
    if len(p_kge):
        print(f"    mean={p_kge.mean():.4f}  min={p_kge.min():.4f}  max={p_kge.max():.4f}")
        if (p_kge > 0).any():
            print(f"    WARNING: {int((p_kge>0).sum())} positive KGE values — check PBIAS")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"  Done. {len(df)} rows, {len(df.columns)} columns")
    print(f"\nNext step:")
    print(f"  python aggregate_research_metrics.py --input {output_path}")

if __name__ == "__main__":
    main()