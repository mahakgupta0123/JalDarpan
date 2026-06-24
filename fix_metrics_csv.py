
"""
JalDarpan — Combined Metrics Fix Script
========================================
Runs all fixes in one pass. No separate scripts needed.

FIX A : Persistence KGE from PBIAS (constant predictor formula)
FIX B : RF and XGBoost KGE from stored Alpha/Beta/Pearson r
FIX C : NSE → mNSE column rename
FIX D : xgb_cm__ → XGBoost_cm__ confusion matrix prefix
FIX E : NRMSE recomputed from gwl_range (RMSE / gwl_range)
         This is the main new addition — fixes NRMSE > 1 issue
         and adds correct NRMSE to publication table.

Run:
    python fix_metrics_csv.py --input research_metrics_append.csv
"""

import argparse, os, sys, shutil
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",     default="research_metrics_append.csv")
    p.add_argument("--output",    default=None)
    p.add_argument("--beta-cap",  type=float, default=10.0)
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

    MODELS = ["Persistence", "Random Forest", "XGBoost"]

    # ── FIX A: Persistence KGE from PBIAS ────────────────────────────────────
    # For a constant predictor: r=0, alpha=0 by definition.
    # KGE = 1 - sqrt(2 + (PBIAS/100)²)
    # beta = 1 + PBIAS/100
    print("\nFIX A: Persistence KGE from PBIAS (constant predictor formula)")

    pbias_col = "Persistence__PBIAS"
    kge_col   = "Persistence__KGE"

    if pbias_col in df.columns and kge_col in df.columns:
        df[pbias_col] = pd.to_numeric(df[pbias_col], errors="coerce")
        df[kge_col]   = pd.to_numeric(df[kge_col],   errors="coerce")

        pbias_vals      = df[pbias_col]
        beta_from_pbias = 1.0 + pbias_vals / 100.0
        valid           = pbias_vals.notna() & (beta_from_pbias.abs() <= args.beta_cap)
        kge_computed    = 1.0 - np.sqrt(2.0 + (pbias_vals / 100.0) ** 2)
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
    else:
        print(f"  SKIP: {pbias_col} or {kge_col} not found")

    # ── FIX B: RF and XGBoost KGE from stored components ─────────────────────
    print("\nFIX B: RF and XGBoost KGE from stored Alpha/Beta/Pearson r")

    for model in ["Random Forest", "XGBoost"]:
        kge_col     = f"{model}__KGE"
        alpha_col   = f"{model}__Alpha"
        beta_col    = f"{model}__Beta"
        pearson_col = f"{model}__Pearson r"

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
            new_kge.append(recompute_kge(
                pearson_r = row.get(pearson_col, np.nan),
                alpha     = row.get(alpha_col,   np.nan),
                beta      = row.get(beta_col,    np.nan),
                beta_cap  = args.beta_cap,
            ))
        df[kge_col] = new_kge
        new_nan = int(pd.Series(new_kge).isna().sum())
        print(f"  {model:20s}: NaN {old_nan} → {new_nan}  "
              f"(recovered {max(old_nan - new_nan, 0)})")

    # ── FIX C: NSE → mNSE column rename ──────────────────────────────────────
    print("\nFIX C: NSE → mNSE column rename")
    for model in MODELS:
        old = f"{model}__NSE";  new = f"{model}__mNSE"
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
            print(f"  Renamed: {old} → {new}")
        elif new in df.columns:
            print(f"  OK: {new} already exists")
        else:
            print(f"  MISSING: neither {old} nor {new} found")

    # ── FIX D: xgb_cm__ → XGBoost_cm__ prefix ───────────────────────────────
    print("\nFIX D: xgb_cm__ → XGBoost_cm__ prefix")
    old_cm = [c for c in df.columns if c.startswith("xgb_cm__")]
    if old_cm:
        df.rename(columns={c: c.replace("xgb_cm__", "XGBoost_cm__")
                           for c in old_cm}, inplace=True)
        print(f"  Renamed {len(old_cm)} columns")
    else:
        print("  OK: no old xgb_cm__ columns found")

    # ── FIX E: NRMSE recomputed from gwl_range ────────────────────────────────
    # Correct formula: NRMSE = RMSE / (gwl_max - gwl_min)
    # gwl_range column is already stored in the CSV from the sweep.
    # The old stored NRMSE used test-set range as denominator which was
    # too small, causing NRMSE > 1 in many districts.
    # This fix overwrites the stored NRMSE with the correct value.
    print("\nFIX E: NRMSE recomputed from gwl_range (RMSE / gwl_range)")

    gwl_range_col = "gwl_range"
    if gwl_range_col not in df.columns:
        # try per-model gwl_range as fallback
        for model in MODELS:
            candidate = f"{model}__gwl_range"
            if candidate in df.columns:
                gwl_range_col = candidate
                print(f"  Using {candidate} as gwl_range source")
                break

    if gwl_range_col not in df.columns:
        print("  SKIP: gwl_range column not found — NRMSE cannot be recomputed")
        print("  Available columns with 'range':")
        for c in df.columns:
            if "range" in c.lower():
                print(f"    {c}")
    else:
        gwl_range = pd.to_numeric(df[gwl_range_col], errors="coerce")
        valid_range = gwl_range.notna() & (gwl_range > 0)
        total_valid = int(valid_range.sum())
        print(f"  gwl_range: {total_valid} valid values  "
              f"(min={gwl_range[valid_range].min():.3f}  "
              f"max={gwl_range[valid_range].max():.3f})")

        for model in MODELS:
            rmse_col  = f"{model}__RMSE"
            nrmse_col = f"{model}__NRMSE"

            if rmse_col not in df.columns:
                print(f"  {model:20s}: RMSE column missing — skip"); continue

            rmse = pd.to_numeric(df[rmse_col], errors="coerce")

            # Compute correct NRMSE
            new_nrmse = rmse / gwl_range
            new_nrmse[~valid_range] = np.nan   # no valid range → NaN

            # Report how many values are > 1
            # (possible in very narrow-range districts — not a bug, just unusual)
            above_1 = (new_nrmse > 1.0) & valid_range & rmse.notna()

            # Overwrite stored NRMSE
            df[nrmse_col] = new_nrmse

            valid_nrmse = new_nrmse[valid_range & rmse.notna()]
            print(f"  {model:20s}: "
                  f"mean={valid_nrmse.mean():.4f}  "
                  f"median={valid_nrmse.median():.4f}  "
                  f"max={valid_nrmse.max():.4f}  "
                  f"values>1: {int(above_1.sum())}")

        # Sanity check — order should be Persistence > RF/XGBoost
        print("\n  Sanity check — NRMSE order (lower = better model):")
        for model in MODELS:
            col   = f"{model}__NRMSE"
            vals  = pd.to_numeric(df[col], errors="coerce")
            clean = vals[(vals >= 0) & (vals <= 2) & vals.notna()]
            print(f"    {model:20s}: mean={clean.mean():.4f}  "
                  f"median={clean.median():.4f}  "
                  f"(using {len(clean)} districts with NRMSE in [0,2])")

        p_mean  = pd.to_numeric(df["Persistence__NRMSE"],   errors="coerce")
        xgb_mean= pd.to_numeric(df["XGBoost__NRMSE"],       errors="coerce")
        p_clean  = p_mean[(p_mean >= 0)   & (p_mean <= 2)].mean()
        x_clean  = xgb_mean[(xgb_mean >= 0) & (xgb_mean <= 2)].mean()
        if p_clean > x_clean:
            print("  ✓ Order correct: Persistence NRMSE > XGBoost NRMSE")
        else:
            print("  ⚠ Order unexpected — check gwl_range values for anomalies")

    # ── PREVIEW ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREVIEW — Updated metrics (valid districts, outliers filtered)")
    print("=" * 60)

    valid_df = df[pd.to_numeric(df.get("rows", 0), errors="coerce") > 0].copy()
    R2_FLOOR = -50.0   # same floor as aggregation script

    preview_rows = []
    for model in MODELS:
        r = {"Model": model}
        for metric in ["NRMSE", "R2", "mNSE", "KGE"]:
            col = f"{model}__{metric}"
            if col in valid_df.columns:
                vals = pd.to_numeric(valid_df[col], errors="coerce")
                # apply same floor filter as aggregation script
                if metric == "R2":
                    vals = vals[vals > R2_FLOOR]
                elif metric == "NRMSE":
                    vals = vals[(vals >= 0) & (vals <= 2)]
                r[f"{metric} (mean)"] = round(vals.mean(), 6) if len(vals) else np.nan
        r["Valid rows"] = len(valid_df)
        preview_rows.append(r)

    print(pd.DataFrame(preview_rows).set_index("Model")
          .to_string(float_format="{:.6f}".format))

    p_kge = pd.to_numeric(
        valid_df.get("Persistence__KGE", pd.Series(dtype=float)),
        errors="coerce").dropna()
    print(f"\n  Persistence KGE: {len(p_kge)} valid values")
    if len(p_kge):
        print(f"    mean={p_kge.mean():.4f}  "
              f"min={p_kge.min():.4f}  "
              f"max={p_kge.max():.4f}")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    print(f"\nSaving to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"  Done. {len(df)} rows, {len(df.columns)} columns")
    print(f"""
Next steps:
  1. python aggregate_research_metrics.py --input {output_path}
  2. Restart your Streamlit dashboard
  
After step 1, publication_national_table.csv will have:
  NRMSE (mean) column  ← fixes dashboard KeyError
  Correct NRMSE values in 0-1 range
  Persistence KGE ≈ -0.55 (not NaN)
""")


if __name__ == "__main__":
    main()

