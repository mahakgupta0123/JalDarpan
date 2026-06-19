import os
import pandas as pd


def load_benchmark_results(csv_path):
    """Load benchmark results from the given CSV path."""
    if not csv_path:
        return pd.DataFrame()
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return pd.DataFrame()


def build_pooled_summaries(runs_df):
    if runs_df is None or runs_df.empty:
        return None

    weight = runs_df.get("rows", pd.Series(1, index=runs_df.index)).fillna(1.0).astype(float)
    regime_valid_runs = runs_df[runs_df.get("regime_valid", False)]
    if regime_valid_runs.empty:
        target_df = runs_df
    else:
        target_df = regime_valid_runs

    model_names = set()
    for col in target_df.columns:
        if "__" in col:
            model_names.add(col.split("__", 1)[0])
    model_names = sorted(model_names)

    summary_rows = []
    for model_name in model_names:
        model_cols = [col for col in target_df.columns if col.startswith(f"{model_name}__")]
        row = {"model": model_name, "num_districts": int(target_df["district"].nunique()) if "district" in target_df.columns else 0}
        for col in model_cols:
            metric_name = col.split("__", 1)[1]
            values = target_df[col].astype(float, errors="ignore").fillna(0.0)
            if metric_name in {"RMSE", "MAE", "MAPE", "PBIAS", "Bias"}:
                row[metric_name] = float((values * weight).sum() / max(weight.sum(), 1.0))
            elif metric_name in {"R2", "NSE", "Pearson r", "KGE", "Regime Acc", "Regime Coverage", "Regime Majority Share"}:
                row[metric_name] = float((values * weight).sum() / max(weight.sum(), 1.0))
            else:
                row[metric_name] = float(values.median()) if not values.empty else float("nan")
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("model")

    xgb_cm = pd.DataFrame(0, index=["Actual Low", "Actual Moderate", "Actual High"], columns=["Pred Low", "Pred Moderate", "Pred High"])
    if "XGBoost" in model_names:
        for actual_label in xgb_cm.index:
            for pred_label in xgb_cm.columns:
                col_name = f"xgb_cm__{actual_label}__{pred_label}"
                if col_name in runs_df.columns:
                    xgb_cm.loc[actual_label, pred_label] = int(runs_df[col_name].fillna(0).sum())

    return {
        "summary": summary,
        "xgb_cm": xgb_cm,
    }


def get_regime_validity_note(valid_run_count, total_run_count):
    try:
        valid_run_count = int(valid_run_count)
        total_run_count = int(total_run_count)
    except Exception:
        return "Regime validity could not be determined from the benchmark dataset."

    if total_run_count == 0:
        return "No benchmark runs are available."
    if valid_run_count == total_run_count:
        return "All saved runs contain full three-class regime coverage, so benchmark calculations are based on complete regime evaluation."
    if valid_run_count > 0:
        return f"{valid_run_count} of {total_run_count} saved runs contain full three-class regime coverage. Pooled metrics use only those valid runs for better consistency."
    return "No saved runs contain full three-class regime coverage. Historical benchmark metrics should be interpreted with caution."
