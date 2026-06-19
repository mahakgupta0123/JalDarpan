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
        default=-10.0,
        help="R� threshold below which a model is considered invalid."
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
    lens_summary = ", ".join([f"{length}:{count}" for length, count in sorted(row_lens.items())])
    print(f"  Row length distribution : {lens_summary}")

    return pd.DataFrame(rows, columns=header)


def _norm_key(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def find_metric_columns(columns, models):
    found = {}
    for col in columns:
        if isinstance(col, str) and "__" in col:
            model_name, metric_name = col.split("__", 1)
            model_name = model_name.strip()
            metric_name = metric_name.strip()
            found.setdefault(model_name, {})[metric_name] = col

    canonical_metrics = [
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "NSE",
        "NRMSE",
        "KGE",
        "PBIAS",
        "Pearson r",
        "Bias",
        "Alpha",
        "Beta",
    ]

    metric_map = {}
    for model in models:
        cols = found.get(model, {})
        if not cols:
            for candidate, cdict in found.items():
                if _norm_key(candidate) == _norm_key(model):
                    cols = cdict
                    break

        normalized = { _norm_key(k): v for k, v in cols.items() }
        model_map = {}
        for metric in canonical_metrics:
            key = _norm_key(metric)
            actual = normalized.get(key)
            if actual is None:
                for nk, colname in normalized.items():
                    if key in nk or nk in key:
                        actual = colname
                        break
            model_map[metric] = actual
        metric_map[model] = model_map

    return metric_map


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
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "valid_count": int(mask.sum())}

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
    grouped = df_source.groupby(group_by, sort=False) if group_by else [("National", df_source)]

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
                    record[f"{prefix}_valid_pct"] = round(100 * stats["valid_count"] / len(group), 2) if len(group) else np.nan

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
                record[f"{metric}_valid_pct"] = round(100 * stats["valid_count"] / len(df_source), 2) if len(df_source) else np.nan

        rows.append(record)
    return pd.DataFrame(rows)


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
        df["regime_class_count"] = pd.to_numeric(df["regime_class_count"], errors="coerce").fillna(0).astype(int)
    else:
        df["regime_class_count"] = 0

    if "regime_valid" in df.columns:
        df["regime_valid"] = df["regime_valid"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
    else:
        df["regime_valid"] = False

    models = ["Persistence", "Random Forest", "XGBoost"]
    metric_cols = find_metric_columns(df.columns, models)

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
    mask_low_regime = df["regime_class_count"] < 2
    mask_mape_outlier = pd.Series(False, index=df.index)
    mask_nrmse_invalid = pd.Series(False, index=df.index)
    mask_r2_invalid = pd.Series(False, index=df.index)

    for model in models:
        cols = metric_cols[model]
        if cols["MAPE"]:
            mask_mape_outlier |= df[cols["MAPE"]].isna() | (df[cols["MAPE"]] > 300)
        else:
            print(f"WARNING: {model} MAPE column not found; excluding all districts for consistency.")
            mask_mape_outlier |= True

        if cols["NRMSE"]:
            mask_nrmse_invalid |= df[cols["NRMSE"]].isna() | (df[cols["NRMSE"]] > 1)
        else:
            print(f"WARNING: {model} NRMSE column not found; excluding all districts for consistency.")
            mask_nrmse_invalid |= True

        if cols["R2"]:
            mask_r2_invalid |= df[cols["R2"]].isna() | (df[cols["R2"]] < args.outlier_r2_floor)
        else:
            print(f"WARNING: {model} R2 column not found; excluding all districts for consistency.")
            mask_r2_invalid |= True

    mask_common_valid = ~(mask_low_rows | mask_low_regime | mask_mape_outlier | mask_nrmse_invalid | mask_r2_invalid)

    df_excluded = df[~mask_common_valid].copy()
    exclude_columns = ["state", "district", "rows", "regime_valid", "regime_class_count", "Regime Coverage", "Regime Majority Share"]
    exclude_columns = [c for c in exclude_columns if c in df.columns]
    df_excluded = df_excluded[[*exclude_columns, *[c for c in df_excluded.columns if c not in exclude_columns]]] if not df_excluded.empty else df_excluded

    for name, mask in [
        ("low_rows", mask_low_rows),
        ("low_regime_diversity", mask_low_regime),
        ("mape_outlier", mask_mape_outlier),
        ("nrmse_invalid", mask_nrmse_invalid),
        ("r2_invalid", mask_r2_invalid),
    ]:
        df_excluded[name] = mask[~mask_common_valid]

    df_excluded["exclude_reasons"] = df_excluded[["low_rows", "low_regime_diversity", "mape_outlier", "nrmse_invalid", "r2_invalid"]].astype(bool) \
        .apply(lambda row: ",".join([name for name, val in row.items() if val]), axis=1)

    df_excluded.to_csv(os.path.join(args.output_dir, "excluded_districts.csv"), index=False)

    print(f"  Districts removed by low rows       : {int(mask_low_rows.sum())}")
    print(f"  Districts removed by low regime mix : {int(mask_low_regime.sum())}")
    print(f"  Districts removed by MAPE outlier   : {int(mask_mape_outlier.sum())}")
    print(f"  Districts removed by NRMSE invalid   : {int(mask_nrmse_invalid.sum())}")
    print(f"  Districts removed by R2 invalid     : {int(mask_r2_invalid.sum())}")
    print(f"  Total districts excluded           : {len(df_excluded)}")

    if mask_common_valid.sum() == 0:
        print("No districts remain after strict aggregation filtering. Exiting.")
        sys.exit(1)

    df_clean = df[mask_common_valid].copy()
    df_clean.to_csv(os.path.join(args.output_dir, "cleaned_dataset_used_for_aggregation.csv"), index=False)
    print(f"  Districts retained for aggregation  : {len(df_clean)}")

    print("\nSTEP 3 - District-level cleaned metrics")
    district_cols = ["state", "district", "rows", "regime_valid", "regime_class_count", "Regime Coverage", "Regime Majority Share"]
    for model in models:
        for metric, col in metric_cols[model].items():
            if col and col in df_clean.columns:
                district_cols.append(col)
    district_cols = [c for c in dict.fromkeys(district_cols) if c in df_clean.columns]
    df_district = df_clean[district_cols].copy()
    df_district.to_csv(os.path.join(args.output_dir, "district_level_metrics.csv"), index=False)
    print(f"  Saved → district_level_metrics.csv  ({len(df_district)} districts)")

    print("\nSTEP 4 - State-level weighted aggregation")
    df_state = aggregate_group(df_clean, ["state"], models, metric_cols, weight_col="rows")
    df_state.to_csv(os.path.join(args.output_dir, "state_level_aggregation.csv"), index=False)
    print(f"  Saved → state_level_aggregation.csv  ({len(df_state)} states)")

    print("\nSTEP 5 - National-level weighted aggregation")
    df_national = aggregate_models(df_clean, models, metric_cols, weight_col="rows")
    df_national.to_csv(os.path.join(args.output_dir, "national_level_aggregation.csv"), index=False)
    print(f"  Saved → national_level_aggregation.csv")

    print("\nSTEP 6 - Model ranking per district")
    rank_records = []
    for _, row in df_clean.iterrows():
        rec = {"state": row["state"], "district": row["district"], "rows": int(row["rows"])}
        nrmse_vals = {}
        for model in models:
            col = metric_cols[model]["NRMSE"]
            if col and pd.notna(row[col]):
                nrmse_vals[model] = float(row[col])
        if len(nrmse_vals) == len(models):
            ranked = sorted(nrmse_vals, key=nrmse_vals.get)
            for rank_pos, model in enumerate(ranked, 1):
                rec[f"rank_{rank_pos}"] = model
                rec[f"rank_{rank_pos}_NRMSE"] = nrmse_vals[model]
        rank_records.append(rec)

    df_ranks = pd.DataFrame(rank_records)
    df_ranks.to_csv(os.path.join(args.output_dir, "district_model_rankings.csv"), index=False)
    print(f"  Saved → district_model_rankings.csv")

    print("\n  Model Win Frequencies (Rank 1 = Best NRMSE nationally):")
    if "rank_1" in df_ranks.columns:
        rank1_counts = df_ranks["rank_1"].value_counts()
        for model, count in rank1_counts.items():
            pct = 100 * count / len(df_ranks.dropna(subset=["rank_1"]))
            print(f"    {model:20s}: {count:4d} districts  ({pct:.1f}%)")

    print("\nSTEP 7 - Diagnostic metric distributions")
    metric_diagnostics = []
    for model in models:
        for metric, col in metric_cols[model].items():
            if not col or col not in df_clean.columns:
                continue
            stats = weighted_stats(df_clean[col], df_clean["rows"])
            metric_diagnostics.append({
                "Model": model,
                "Metric": metric,
                "WeightedMean": round(stats["mean"], 6) if pd.notna(stats["mean"]) else np.nan,
                "WeightedStd": round(stats["std"], 6) if pd.notna(stats["std"]) else np.nan,
                "WeightedMedian": round(stats["median"], 6) if pd.notna(stats["median"]) else np.nan,
                "ValidDistricts": stats["valid_count"],
                "TotalDistricts": len(df_clean),
            })
    if metric_diagnostics:
        pd.DataFrame(metric_diagnostics).to_csv(os.path.join(args.output_dir, "metric_distribution_diagnostics.csv"), index=False)
        print("  Saved → metric_distribution_diagnostics.csv")
    else:
        print("  No metric diagnostics were saved.")

    print("\nSTEP 8 - Publication-ready national summary")
    pub_rows = []
    for _, row in df_national.iterrows():
        pub_rows.append({
            "Model": row["Model"],
            "NRMSE (mean)": row.get("NRMSE_mean", np.nan),
            "NRMSE (std)": row.get("NRMSE_std", np.nan),
            "RMSE (mean)": row.get("RMSE_mean", np.nan),
            "MAE (mean)": row.get("MAE_mean", np.nan),
            "MAPE (mean)": row.get("MAPE_mean", np.nan),
            "R2 (mean)": row.get("R2_mean", np.nan),
            "R2 (std)": row.get("R2_std", np.nan),
            "NSE (mean)": row.get("NSE_mean", np.nan),
            "KGE (mean)": row.get("KGE_mean", np.nan),
            "KGE valid pct": row.get("KGE_valid_pct", np.nan),
            "Valid Districts": int(row.get("n_districts", np.nan)) if pd.notna(row.get("n_districts", np.nan)) else np.nan,
            "Total Obs": int(row.get("total_obs", np.nan)) if pd.notna(row.get("total_obs", np.nan)) else np.nan,
        })
    df_pub = pd.DataFrame(pub_rows)
    df_pub.to_csv(os.path.join(args.output_dir, "publication_national_table.csv"), index=False)
    print("  Saved → publication_national_table.csv")

    print("\nSTEP 9 - Metrics preview")
    preview_cols = ["Model", "NRMSE (mean)", "R2 (mean)", "NSE (mean)", "KGE (mean)", "Valid Districts"]
    preview_cols = [c for c in preview_cols if c in df_pub.columns]
    if preview_cols:
        print(df_pub[preview_cols].to_string(index=False))
    else:
        print("  No publication preview columns available.")

    print("\nSTEP 10 - XGBoost confusion matrix")
    cm_cols = [c for c in df_clean.columns if isinstance(c, str) and c.startswith("xgb_cm__")]
    if cm_cols:
        cm_df = df_clean[cm_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        valid_rows = cm_df.notna().all(axis=1) & (cm_df >= 0).all(axis=1)
        integer_rows = (cm_df.round() - cm_df).abs() < 1e-6
        valid_rows &= integer_rows.all(axis=1)

        if valid_rows.any():
            cm_counts = cm_df.loc[valid_rows].round().astype(int).sum()
            cm_summary = cm_counts.reset_index()
            cm_summary.columns = ["ConfusionPair", "Count"]
            cm_summary[["Actual", "Pred"]] = cm_summary["ConfusionPair"].str.replace("xgb_cm__", "", regex=False).str.split("__", expand=True)
            cm_summary = cm_summary[["Actual", "Pred", "Count"]]
            cm_file = os.path.join(args.output_dir, "xgb_confusion_matrix_summary.csv")
            cm_summary.to_csv(cm_file, index=False)
            print(f"  Saved → {os.path.basename(cm_file)}")

            pivot = cm_summary.pivot(index="Actual", columns="Pred", values="Count").fillna(0).astype(int)
            print("\n  XGBoost confusion matrix (national totals):")
            print(pivot.to_string())
        else:
            print("  XGBoost confusion matrix columns found, but no valid non-negative integer count rows could be confirmed. Skipping confusion matrix summary.")
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
    ]:
        print(f"    ✓  {filename}")
    print()


if __name__ == "__main__":
    main()
