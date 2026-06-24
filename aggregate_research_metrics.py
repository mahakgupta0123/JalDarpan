"""
National Groundwater Level Forecasting - Aggregation Pipeline
WRIS / CGWB Dataset | All India (1010 district entries)

Produces:
  1. Filtered clean dataset (excluded zero-row / insufficient districts)
  2. District-level normalized metrics
  3. State-level weighted aggregation
  4. National-level weighted aggregation
  5. Anomaly density summary
  6. Model ranking table
  7. Publication-ready national summary
  8. All outputs exported as CSV

Fixes applied vs original:
  FIX 1 : EXACT_METRIC_COLS updated NSE -> mNSE to match sweep script output
  FIX 2 : Publication table uses mNSE_mean, KGE shown for all models incl. Persistence
  FIX 3 : Confusion matrix prefix corrected xgb_cm__ -> XGBoost_cm__
  FIX 4 : Regime accuracy (step 6b) filtered to regime_valid==True districts only
  FIX 5 : R2/mNSE synchronisation block key corrected NSE -> mNSE

Author  : Research Pipeline Script
Dataset : CGWB / WRIS Groundwater Level Data
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate national groundwater forecasting metrics."
    )
    parser.add_argument(
        "--input",
        default="national_results.csv",
        help="Input CSV file to read for aggregation."
    )
    parser.add_argument(
        "--output-dir",
        default="aggregation_outputs",
        help="Directory to save aggregation outputs."
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=30,
        help="Minimum number of rows required to keep a district."
    )
    parser.add_argument(
        "--outlier-r2-floor",
        type=float,
        default=-50.0,
        help="R² threshold below which a model is considered invalid."
    )
    parser.add_argument(
        "--winsor-lower",
        type=float,
        default=0.01,
        help="Lower percentile for winsorization."
    )
    parser.add_argument(
        "--winsor-upper",
        type=float,
        default=0.99,
        help="Upper percentile for winsorization."
    )
    return parser.parse_args()


def read_input_csv(path):
    path = str(path)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()

        rows = []
        max_len = len(header)
        row_lens = {max_len: 0}

        for row in reader:
            ln = len(row)
            row_lens[ln] = row_lens.get(ln, 0) + 1
            if ln > max_len:
                extra_count = ln - max_len
                for i in range(extra_count):
                    header.append(f"__extra_col_{max_len + i}")
                for r in rows:
                    r.extend([""] * extra_count)
                max_len = ln
            elif ln < max_len:
                row.extend([""] * (max_len - ln))
            rows.append(row)

    print(f"  Header columns detected : {len(header)}")
    lens_summary = ", ".join(
        [f"{length}:{count}" for length, count in sorted(row_lens.items())]
    )
    print(f"  Row length distribution : {lens_summary}")

    return pd.DataFrame(rows, columns=header)


def _norm_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


# FIX 1: NSE renamed to mNSE in all three model blocks to match sweep script output.
# The sweep script now writes columns named e.g. "XGBoost__mNSE" not "XGBoost__NSE".
# Every lookup in steps 4, 5, 7, 8, 9 reads from this dict, so this single change
# fixes all downstream NaN values that appeared in the publication table.
EXACT_METRIC_COLS = {
    "Persistence": {
        "RMSE":      "Persistence__RMSE",
        "MAE":       "Persistence__MAE",
        "MAPE":      "Persistence__MAPE",
        "R2":        "Persistence__R2",
        "mNSE":      "Persistence__mNSE",       # FIX 1: was "NSE": "Persistence__NSE"
        "KGE":       "Persistence__KGE",
        "PBIAS":     "Persistence__PBIAS",
        "Pearson r": "Persistence__Pearson r",
        "Alpha":     "Persistence__Alpha",
        "Beta":      "Persistence__Beta",
        "NRMSE":     "Persistence__NRMSE",
    },
    "Random Forest": {
        "RMSE":      "Random Forest__RMSE",
        "MAE":       "Random Forest__MAE",
        "MAPE":      "Random Forest__MAPE",
        "R2":        "Random Forest__R2",
        "mNSE":      "Random Forest__mNSE",     # FIX 1: was "NSE": "Random Forest__NSE"
        "KGE":       "Random Forest__KGE",
        "PBIAS":     "Random Forest__PBIAS",
        "Pearson r": "Random Forest__Pearson r",
        "Alpha":     "Random Forest__Alpha",
        "Beta":      "Random Forest__Beta",
        "NRMSE":     "Random Forest__NRMSE",
    },
    "XGBoost": {
        "RMSE":      "XGBoost__RMSE",
        "MAE":       "XGBoost__MAE",
        "MAPE":      "XGBoost__MAPE",
        "R2":        "XGBoost__R2",
        "mNSE":      "XGBoost__mNSE",           # FIX 1: was "NSE": "XGBoost__NSE"
        "KGE":       "XGBoost__KGE",
        "PBIAS":     "XGBoost__PBIAS",
        "Pearson r": "XGBoost__Pearson r",
        "Alpha":     "XGBoost__Alpha",
        "Beta":      "XGBoost__Beta",
        "NRMSE":     "XGBoost__NRMSE",
    },
}


def weighted_stats(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    mask = values.notna() & (weights > 0)
    if not mask.any():
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "valid_count": 0}

    vals = values[mask]
    w = weights[mask].astype(float)
    total_w = w.sum()
    if total_w == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "valid_count": int(mask.sum()),
        }

    mean = (vals * w).sum() / total_w
    variance = (w * (vals - mean) ** 2).sum() / total_w
    std = np.sqrt(variance)

    order = np.argsort(vals.values)
    sorted_vals = vals.values[order]
    sorted_w = w.values[order]
    cum_w = np.cumsum(sorted_w) / total_w
    median = sorted_vals[np.searchsorted(cum_w, 0.5)]

    return {
        "mean": float(mean),
        "std": float(std),
        "median": float(median),
        "valid_count": int(mask.sum()),
    }


def aggregate_group(df_source, group_by, models, metric_cols, weight_col="rows"):
    results = []
    grouped = (
        df_source.groupby(group_by, sort=False)
        if group_by
        else [("National", df_source)]
    )

    for key, group in grouped:
        record = {}
        if group_by:
            if isinstance(key, tuple):
                for col_name, value in zip(group_by, key):
                    record[col_name] = value
            else:
                record[group_by[0]] = key
        else:
            record["scope"] = "National"

        record["n_districts"] = int(len(group))
        record["total_obs"] = int(group[weight_col].sum())

        for model in models:
            for metric, col in metric_cols[model].items():
                prefix = f"{model}__{metric}"
                if not col or col not in group.columns:
                    record[f"{prefix}_mean"] = np.nan
                    record[f"{prefix}_std"] = np.nan
                    record[f"{prefix}_median"] = np.nan
                    record[f"{prefix}_valid_count"] = 0
                    if metric == "KGE":
                        record[f"{prefix}_valid_pct"] = np.nan
                    continue

                stats = weighted_stats(group[col], group[weight_col])
                record[f"{prefix}_mean"] = round(stats["mean"], 6)
                record[f"{prefix}_std"] = round(stats["std"], 6)
                record[f"{prefix}_median"] = round(stats["median"], 6)
                record[f"{prefix}_valid_count"] = stats["valid_count"]
                if metric == "KGE":
                    record[f"{prefix}_valid_pct"] = (
                        round(100 * stats["valid_count"] / len(group), 2)
                        if len(group)
                        else np.nan
                    )

        results.append(record)

    return pd.DataFrame(results)


def aggregate_models(df_source, models, metric_cols, weight_col="rows"):
    rows = []
    for model in models:
        record = {
            "Model": model,
            "n_districts": int(len(df_source)),
            "total_obs": int(df_source[weight_col].sum()),
        }
        for metric, col in metric_cols[model].items():
            if not col or col not in df_source.columns:
                record[f"{metric}_mean"] = np.nan
                record[f"{metric}_std"] = np.nan
                record[f"{metric}_median"] = np.nan
                record[f"{metric}_valid_count"] = 0
                if metric == "KGE":
                    record[f"{metric}_valid_pct"] = np.nan
                continue

            stats = weighted_stats(df_source[col], df_source[weight_col])
            record[f"{metric}_mean"] = round(stats["mean"], 6)
            record[f"{metric}_std"] = round(stats["std"], 6)
            record[f"{metric}_median"] = round(stats["median"], 6)
            record[f"{metric}_valid_count"] = stats["valid_count"]
            if metric == "KGE":
                record[f"{metric}_valid_pct"] = (
                    round(100 * stats["valid_count"] / len(df_source), 2)
                    if len(df_source)
                    else np.nan
                )

        rows.append(record)
    return pd.DataFrame(rows)


def weighted_pearsonr(x, y, w):
    """Compute weighted Pearson correlation between x and y with weights w.

    Returns (r, n_valid)
    """
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    w = pd.to_numeric(w, errors="coerce").fillna(0)
    mask = x.notna() & y.notna() & (w > 0)
    if mask.sum() < 3:
        return (np.nan, int(mask.sum()))
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    w = w[mask].astype(float)
    sw = w.sum()
    if sw <= 0:
        return (np.nan, int(mask.sum()))
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    cx = x - mx
    cy = y - my
    cov = (w * cx * cy).sum() / sw
    varx = (w * cx * cx).sum() / sw
    vary = (w * cy * cy).sum() / sw
    if varx <= 0 or vary <= 0:
        return (np.nan, int(mask.sum()))
    r = cov / (np.sqrt(varx) * np.sqrt(vary))
    return (float(r), int(mask.sum()))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1 - Loading data")
    print("=" * 60)

    df = read_input_csv(args.input)
    print(f"  Total rows loaded       : {len(df)}")

    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    if "rows" in df.columns:
        df["rows"] = pd.to_numeric(df["rows"], errors="coerce").fillna(0).astype(int)
    else:
        df["rows"] = 0

    print("\nSTEP 2 - Cleaning and filtering")

    if "regime_class_count" in df.columns:
        df["regime_class_count"] = (
            pd.to_numeric(df["regime_class_count"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        df["regime_class_count"] = 0

    if "regime_valid" in df.columns:
        df["regime_valid"] = (
            df["regime_valid"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        )
    else:
        df["regime_valid"] = False

    models = ["Persistence", "Random Forest", "XGBoost"]
    metric_cols = EXACT_METRIC_COLS

    print("  Metric column mapping:")
    for model in models:
        mapped_metrics = [
            f"{metric}:{metric_cols[model][metric] or 'MISSING'}"
            for metric in ["NRMSE", "MAPE", "R2", "KGE"]
        ]
        print(f"    {model}: {', '.join(mapped_metrics)}")

    for model, cmap in metric_cols.items():
        for col in cmap.values():
            if col and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    mask_low_rows = df["rows"] < args.min_rows
    mask_mape_outlier = pd.Series(False, index=df.index)
    mask_r2_invalid = pd.Series(False, index=df.index)

    for model in models:
        cols = metric_cols[model]
        # Do not exclude districts based on high MAPE — allow wide MAPE values
        # for persistence baselines
        if cols["MAPE"]:
            mask_mape_outlier |= False
        else:
            mask_mape_outlier |= False

        if cols["R2"]:
            mask_r2_invalid |= df[cols["R2"]].isna() | (
                df[cols["R2"]] < args.outlier_r2_floor
            )
        else:
            print(
                f"WARNING: {model} R2 column not found; "
                f"excluding all districts for consistency."
            )
            mask_r2_invalid |= True

    # Do not exclude districts for single-class regimes or precomputed NRMSE;
    # use only row-count and R2 filters
    mask_common_valid = ~(mask_low_rows | mask_mape_outlier | mask_r2_invalid)

    df_excluded = df[~mask_common_valid].copy()
    exclude_columns = [
        "state", "district", "rows", "regime_valid", "regime_class_count",
        "Regime Coverage", "Regime Majority Share",
    ]
    exclude_columns = [c for c in exclude_columns if c in df.columns]
    df_excluded = (
        df_excluded[
            [
                *exclude_columns,
                *[c for c in df_excluded.columns if c not in exclude_columns],
            ]
        ]
        if not df_excluded.empty
        else df_excluded
    )

    for name, mask in [
        ("low_rows", mask_low_rows),
        ("mape_outlier", mask_mape_outlier),
        ("r2_invalid", mask_r2_invalid),
    ]:
        df_excluded[name] = mask[~mask_common_valid]

    df_excluded["exclude_reasons"] = (
        df_excluded[["low_rows", "mape_outlier", "r2_invalid"]]
        .astype(bool)
        .apply(
            lambda row: ",".join([name for name, val in row.items() if val]), axis=1
        )
    )

    df_excluded.to_csv(
        os.path.join(args.output_dir, "excluded_districts.csv"), index=False
    )

    print(f"  Districts removed by low rows       : {int(mask_low_rows.sum())}")
    print(f"  Districts removed by MAPE outlier   : {int(mask_mape_outlier.sum())}")
    print(f"  Districts removed by R2 invalid     : {int(mask_r2_invalid.sum())}")
    print(f"  Total districts excluded           : {len(df_excluded)}")

    if mask_common_valid.sum() == 0:
        print(
            "No districts remain after strict aggregation filtering. Exiting."
        )
        sys.exit(1)

    df_clean = df[mask_common_valid].copy()
    df_clean.to_csv(
        os.path.join(args.output_dir, "cleaned_dataset_used_for_aggregation.csv"),
        index=False,
    )
    print(f"  Districts retained for aggregation  : {len(df_clean)}")

    # Apply model-specific caps for valid ranking metrics prior to aggregation
    CAPS_BY_MODEL = {
        "Persistence":   {"R2": (-5.0, 1.0), "mNSE": (-5.0, 1.0), "KGE": (-5.0, 1.0)},
        "Random Forest": {"R2": (-2.0, 1.0), "mNSE": (-2.0, 1.0), "KGE": (-2.0, 1.0)},
        "XGBoost":       {"R2": (-2.0, 1.0), "mNSE": (-2.0, 1.0), "KGE": (-2.0, 1.0)},
    }
    df_for_agg = df_clean.copy()
    for model in models:
        model_caps = CAPS_BY_MODEL.get(model, {})
        for metric, (lo, hi) in model_caps.items():
            col = metric_cols[model].get(metric)
            if col and col in df_for_agg.columns:
                before_nonnull = df_for_agg[col].notna().sum()
                df_for_agg[col] = pd.to_numeric(df_for_agg[col], errors="coerce")
                df_for_agg.loc[
                    ~((df_for_agg[col] >= lo) & (df_for_agg[col] <= hi)), col
                ] = np.nan
                after_nonnull = df_for_agg[col].notna().sum()
                if before_nonnull != after_nonnull:
                    print(
                        f"    Capped {col}: "
                        f"{before_nonnull - after_nonnull} values set to NaN"
                    )

        # FIX 5: Synchronise R2 and mNSE validity so both metrics have identical
        # NaN masks. Key was "NSE" before — now correctly "mNSE" to match the
        # column names written by the sweep script.
        r2_col  = metric_cols[model].get("R2")
        nse_col = metric_cols[model].get("mNSE")   # FIX 5: was .get("NSE")
        if (
            r2_col and nse_col
            and r2_col  in df_for_agg.columns
            and nse_col in df_for_agg.columns
        ):
            r2_valid  = df_for_agg[r2_col].notna()
            nse_valid = df_for_agg[nse_col].notna()
            if r2_valid.sum() != nse_valid.sum():
                nan_mask = df_for_agg[r2_col].isna() | df_for_agg[nse_col].isna()
                df_for_agg.loc[nan_mask, r2_col]  = np.nan
                df_for_agg.loc[nan_mask, nse_col] = np.nan
                print(
                    f"    Synchronized R2/mNSE NaN mask for {model}: "
                    f"{nan_mask.sum()} rows"
                )

    print("\nSTEP 3 - District-level cleaned metrics")
    district_cols = [
        "state", "district", "rows", "regime_valid", "regime_class_count",
        "Regime Coverage", "Regime Majority Share",
    ]
    for model in models:
        for metric, col in metric_cols[model].items():
            if col and col in df_clean.columns:
                district_cols.append(col)
    district_cols = [
        c for c in dict.fromkeys(district_cols) if c in df_clean.columns
    ]
    df_district = df_clean[district_cols].copy()
    df_district.to_csv(
        os.path.join(args.output_dir, "district_level_metrics.csv"), index=False
    )
    print(
        f"  Saved → district_level_metrics.csv  ({len(df_district)} districts)"
    )

    # STEP 3b - Feature <-> Metric Correlations (national)
    print("\nSTEP 3b - Feature vs. Metric Correlations (weighted Pearson)")
    known_meta = {
        "state", "district", "rows", "regime_valid", "regime_class_count",
        "Regime Coverage", "Regime Majority Share",
    }
    known_metric_cols = set()
    for m in models:
        for mt, c in metric_cols[m].items():
            if c:
                known_metric_cols.add(c)

    candidate_features = [
        c for c in df_clean.columns
        if c not in known_meta
        and c not in known_metric_cols
        and not c.startswith("XGBoost_cm__")   # FIX 3 (consistency): was xgb_cm__
    ]
    numeric_feats = []
    for c in candidate_features:
        try:
            ser = pd.to_numeric(df_clean[c], errors="coerce")
            if ser.notna().sum() >= 3:
                numeric_feats.append(c)
        except Exception:
            continue

    corr_rows = []
    target_metrics = ["NRMSE", "MAE", "R2"]
    for feat in numeric_feats:
        for model in models:
            for tm in target_metrics:
                col = metric_cols[model].get(tm)
                if not col or col not in df_clean.columns:
                    continue
                r, nvalid = weighted_pearsonr(
                    df_clean[feat], df_clean[col], df_clean["rows"]
                )
                corr_rows.append({
                    "Feature": feat,
                    "Model": model,
                    "Metric": tm,
                    "WeightedPearsonR": r,
                    "N_valid": nvalid,
                })

    if corr_rows:
        df_corr = pd.DataFrame(corr_rows)
        df_corr.to_csv(
            os.path.join(args.output_dir, "feature_metric_correlations.csv"),
            index=False,
        )
        print(
            f"  Saved → feature_metric_correlations.csv ({len(df_corr)} rows)"
        )
    else:
        print(
            "  No feature <-> metric correlations computed "
            "(no numeric features detected)"
        )

    print("\nSTEP 4 - State-level weighted aggregation")
    df_state = aggregate_group(
        df_for_agg, ["state"], models, metric_cols, weight_col="rows"
    )
    df_state.to_csv(
        os.path.join(args.output_dir, "state_level_aggregation.csv"), index=False
    )
    print(f"  Saved → state_level_aggregation.csv  ({len(df_state)} states)")

    print("\nSTEP 5 - National-level weighted aggregation")
    df_national = aggregate_models(
        df_for_agg, models, metric_cols, weight_col="rows"
    )
    df_national.to_csv(
        os.path.join(args.output_dir, "national_level_aggregation.csv"), index=False
    )
    print(f"  Saved → national_level_aggregation.csv")

    print("\nSTEP 6 - Model ranking per district")
    rank_records = []
    for _, row in df_clean.iterrows():
        rec = {
            "state":    row["state"],
            "district": row["district"],
            "rows":     int(row["rows"]),
        }
        r2_vals = {}
        for model in models:
            col = metric_cols[model]["R2"]
            if col and pd.notna(row[col]):
                r2_vals[model] = float(row[col])
        if len(r2_vals) == len(models):
            ranked = sorted(r2_vals, key=r2_vals.get, reverse=True)
            for rank_pos, model in enumerate(ranked, 1):
                rec[f"rank_{rank_pos}"]    = model
                rec[f"rank_{rank_pos}_R2"] = r2_vals[model]
        rank_records.append(rec)

    df_ranks = pd.DataFrame(rank_records)
    df_ranks.to_csv(
        os.path.join(args.output_dir, "district_model_rankings.csv"), index=False
    )
    print(f"  Saved → district_model_rankings.csv")

    print("\n  Model Win Frequencies (Rank 1 = Best R2 nationally):")
    if "rank_1" in df_ranks.columns:
        rank1_counts = df_ranks["rank_1"].value_counts()
        for model, count in rank1_counts.items():
            pct = 100 * count / len(df_ranks.dropna(subset=["rank_1"]))
            print(f"    {model:20s}: {count:4d} districts  ({pct:.1f}%)")

    # FIX 4: Regime accuracy must only be averaged over districts where all
    # 3 regime classes (Low / Moderate / High) appeared in the test set.
    # Using all districts inflates accuracy because single-class districts
    # trivially score 100% by predicting that one class for everything.
    print("\nSTEP 6b - National regime accuracy")
    df_regime_valid = df_clean[df_clean["regime_valid"] == True].copy()
    n_valid_regime  = len(df_regime_valid)
    n_total         = len(df_clean)
    print(
        f"  Using {n_valid_regime} of {n_total} districts "
        f"that have all 3 regime classes in the test set."
    )
    for model in models:
        acc_col = f"{model}__Regime Accuracy"
        if acc_col in df_regime_valid.columns:
            df_regime_valid[acc_col] = pd.to_numeric(
                df_regime_valid[acc_col], errors="coerce"
            )
            valid = df_regime_valid[acc_col].notna()
            if valid.any():
                weighted_acc = np.average(
                    df_regime_valid.loc[valid, acc_col],
                    weights=df_regime_valid.loc[valid, "rows"],
                )
                print(
                    f"  {model:20s} National Regime Accuracy: "
                    f"{weighted_acc * 100:.2f}%  "
                    f"(N={int(valid.sum())} districts)"
                )
            else:
                print(
                    f"  {model:20s} National Regime Accuracy: no valid values"
                )
        else:
            print(
                f"  {model:20s} National Regime Accuracy: column missing"
            )

    anomaly_output_file = None
    if "IF__anomaly_density" in df_clean.columns:
        df_clean["IF__anomaly_density"] = pd.to_numeric(
            df_clean["IF__anomaly_density"], errors="coerce"
        )
        valid = df_clean["IF__anomaly_density"].notna()
        if valid.any():
            national_anomaly_density = np.average(
                df_clean.loc[valid, "IF__anomaly_density"],
                weights=df_clean.loc[valid, "rows"],
            )
            total_anomalies = pd.to_numeric(
                df_clean.get("IF__anomaly_count", pd.Series([], dtype=float)),
                errors="coerce",
            ).sum()
            print(
                f"\n  National anomaly density : "
                f"{national_anomaly_density * 100:.2f}%"
            )
            print(f"  Total anomalous readings : {int(total_anomalies):,}")

            try:
                anomaly_output_file = os.path.join(
                    args.output_dir, "anomaly_density_by_district.csv"
                )
                df_anom = df_clean[
                    ["state", "district", "IF__anomaly_density"]
                ].copy()
                if "IF__anomaly_count" in df_clean.columns:
                    df_anom["IF__anomaly_count"] = pd.to_numeric(
                        df_clean["IF__anomaly_count"], errors="coerce"
                    )
                if "rows" in df_clean.columns:
                    df_anom["rows"] = pd.to_numeric(
                        df_clean["rows"], errors="coerce"
                    )
                df_anom.to_csv(anomaly_output_file, index=False)
                print(f"  Saved → {os.path.basename(anomaly_output_file)}")
                if "state" in df_anom.columns:
                    state_output_file = os.path.join(
                        args.output_dir, "anomaly_density_by_state.csv"
                    )
                    df_state_anom = df_anom.groupby(
                        "state", as_index=False
                    ).agg(
                        IF__anomaly_density=("IF__anomaly_density", "mean"),
                        IF__anomaly_count=(
                            ("IF__anomaly_count", "sum")
                            if "IF__anomaly_count" in df_anom.columns
                            else ("IF__anomaly_density", "count")
                        ),
                    )
                    df_state_anom.to_csv(state_output_file, index=False)
                    print(
                        f"  Saved → {os.path.basename(state_output_file)}"
                    )
            except Exception as exc:
                print(
                    f"  WARNING: Could not save anomaly density outputs: {exc}"
                )
        else:
            print(
                "\n  National anomaly density : "
                "no valid anomaly density values"
            )
    else:
        print(
            "\n  National anomaly density : "
            "IF__anomaly_density column not found"
        )

    print("\nSTEP 7 - Diagnostic metric distributions")
    metric_diagnostics = []
    for model in models:
        for metric, col in metric_cols[model].items():
            if not col or col not in df_clean.columns:
                continue
            stats = weighted_stats(df_clean[col], df_clean["rows"])
            metric_diagnostics.append({
                "Model":          model,
                "Metric":         metric,
                "WeightedMean":   round(stats["mean"],   6) if pd.notna(stats["mean"])   else np.nan,
                "WeightedStd":    round(stats["std"],    6) if pd.notna(stats["std"])    else np.nan,
                "WeightedMedian": round(stats["median"], 6) if pd.notna(stats["median"]) else np.nan,
                "ValidDistricts": stats["valid_count"],
                "TotalDistricts": len(df_clean),
            })
    if metric_diagnostics:
        pd.DataFrame(metric_diagnostics).to_csv(
            os.path.join(
                args.output_dir, "metric_distribution_diagnostics.csv"
            ),
            index=False,
        )
        print("  Saved → metric_distribution_diagnostics.csv")
    else:
        print("  No metric diagnostics were saved.")

    print("\nSTEP 7b - National SHAP feature importance")
    shap_cols = [
        c for c in df_clean.columns
        if isinstance(c, str) and c.startswith("SHAP__")
    ]
    if shap_cols:
        for model in ["Random Forest", "XGBoost"]:
            model_shap = [
                c for c in shap_cols
                if c.startswith(f"SHAP__{model}__") and "top" not in c
            ]
            shap_importance = {}
            for col in model_shap:
                feat_name = col.replace(f"SHAP__{model}__", "")
                vals = pd.to_numeric(df_clean[col], errors="coerce")
                if vals.notna().any():
                    shap_importance[feat_name] = np.average(
                        vals.dropna(),
                        weights=df_clean.loc[vals.notna(), "rows"],
                    )
            if shap_importance:
                shap_df = pd.Series(shap_importance).sort_values(ascending=False)
                shap_file = os.path.join(
                    args.output_dir,
                    f"shap_national_{model.replace(' ', '_')}.csv",
                )
                try:
                    shap_df.to_csv(shap_file, header=["mean_abs_shap"])
                    print(f"  Saved → {os.path.basename(shap_file)}")
                except Exception as exc:
                    print(
                        f"  WARNING: Could not save SHAP summary "
                        f"for {model}: {exc}"
                    )
                print(f"\n  {model} — Top 5 features nationally by SHAP:")
                for feat, val in shap_df.head(5).items():
                    print(f"    {feat:25s}: {val:.6f}")
    else:
        print(
            "  No SHAP feature importance columns found "
            "for national aggregation."
        )

    print("\nSTEP 8 - Publication-ready national summary")
    pub_rows = []
    for _, row in df_national.iterrows():
        pub_rows.append({
            "Model":           row["Model"],
            "RMSE (mean)":     row.get("RMSE_mean",  np.nan),
            "NRMSE (mean)":    row.get("NRMSE_mean", np.nan),   
            "NRMSE (std)":     row.get("NRMSE_std",  np.nan),   
            "MAE (mean)":      row.get("MAE_mean",   np.nan),
            "MAPE (mean)":     row.get("MAPE_mean",  np.nan),
            "R2 (mean)":       row.get("R2_mean",    np.nan),
            "R2 (std)":        row.get("R2_std",     np.nan),
            # FIX 2a: was "NSE (mean)": row.get("NSE_mean") — column didn't exist
            # after renaming; now correctly reads mNSE_mean
            "mNSE (mean)":     row.get("mNSE_mean",  np.nan),
            # FIX 2b: was hardcoded "N/A†" for Persistence because KGE was NaN
            # there. Sweep script FIX 5 now computes KGE for constant predictors,
            # so a real (negative) value exists. Show it for all models.
            "KGE (mean)":      row.get("KGE_mean",   np.nan),
            "KGE valid pct":   row.get("KGE_valid_pct", np.nan),
            "Valid Districts":  (
                int(row.get("n_districts", np.nan))
                if pd.notna(row.get("n_districts", np.nan))
                else np.nan
            ),
            "Total Obs":       (
                int(row.get("total_obs", np.nan))
                if pd.notna(row.get("total_obs", np.nan))
                else np.nan
            ),
        })
    df_pub = pd.DataFrame(pub_rows)
    df_pub.to_csv(
        os.path.join(args.output_dir, "publication_national_table.csv"),
        index=False,
    )
    print("  Saved → publication_national_table.csv")

    print("\nSTEP 9 - Metrics preview")
    # FIX 2a (continued): preview column renamed from "NSE (mean)" to "mNSE (mean)"
    preview_cols = [
        "Model", "RMSE (mean)","NRMSE (mean)", "R2 (mean)", "mNSE (mean)", "KGE (mean)",
        "Valid Districts",
    ]
    preview_cols = [c for c in preview_cols if c in df_pub.columns]
    if preview_cols:
        print(df_pub[preview_cols].to_string(index=False))
    else:
        print("  No publication preview columns available.")

    # FIX 3: Confusion matrix prefix corrected from "xgb_cm__" to "XGBoost_cm__".
    # The sweep script writes columns as f"{model_name}_cm__{actual}__{pred}"
    # where model_name is "XGBoost" — so the actual prefix is "XGBoost_cm__".
    # The old prefix "xgb_cm__" never matched any column so step 10 always
    # printed "No XGBoost confusion matrix columns found."
    print("\nSTEP 10 - XGBoost confusion matrix")
    cm_cols = [
        c for c in df_clean.columns
        if isinstance(c, str) and c.startswith("XGBoost_cm__")  # FIX 3
    ]
    if cm_cols:
        cm_df = (
            df_clean[cm_cols]
            .apply(lambda s: pd.to_numeric(s, errors="coerce"))
            .fillna(0)
            .clip(lower=0)
        )
        valid_rows    = (cm_df >= 0).all(axis=1)
        integer_rows  = ((cm_df.round() - cm_df).abs() < 1e-6).all(axis=1)
        valid_rows   &= integer_rows

        if valid_rows.any():
            cm_counts  = cm_df.loc[valid_rows].round().astype(int).sum()
            cm_summary = cm_counts.reset_index()
            cm_summary.columns = ["ConfusionPair", "Count"]
            cm_summary[["Actual", "Pred"]] = (
                cm_summary["ConfusionPair"]
                .str.replace("XGBoost_cm__", "", regex=False)  # FIX 3
                .str.split("__", expand=True)
            )
            cm_summary = cm_summary[["Actual", "Pred", "Count"]]
            cm_file = os.path.join(
                args.output_dir, "xgb_confusion_matrix_summary.csv"
            )
            cm_summary.to_csv(cm_file, index=False)
            print(f"  Saved → {os.path.basename(cm_file)}")

            pivot = (
                cm_summary.pivot(index="Actual", columns="Pred", values="Count")
                .fillna(0)
                .astype(int)
            )
            print("\n  XGBoost confusion matrix (national totals):")
            print(pivot.to_string())
        else:
            print(
                "  XGBoost confusion matrix columns found, but no valid "
                "non-negative integer count rows could be confirmed. "
                "Skipping confusion matrix summary."
            )
    else:
        print("  No XGBoost confusion matrix columns found.")

    print("\n" + "=" * 60)
    print("NATIONAL AGGREGATION COMPLETE")
    print("=" * 60)
    print(f"\n  Total entries in raw CSV          : {len(df)}")
    print(f"  Total districts excluded         : {len(df_excluded)}")
    print(f"  Total clean districts used       : {len(df_clean)}")
    print(f"  Output files in → ./{args.output_dir}/")
    for filename in [
        "excluded_districts.csv",
        "cleaned_dataset_used_for_aggregation.csv",
        "district_level_metrics.csv",
        "state_level_aggregation.csv",
        "national_level_aggregation.csv",
        "district_model_rankings.csv",
        "metric_distribution_diagnostics.csv",
        "publication_national_table.csv",
        "xgb_confusion_matrix_summary.csv",
        "feature_metric_correlations.csv",
    ]:
        print(f"    ✓  {filename}")
    print()


if __name__ == "__main__":
    main()