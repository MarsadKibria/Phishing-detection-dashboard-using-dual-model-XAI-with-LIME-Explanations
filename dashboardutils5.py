# dashboard_utils6.py — HITL + Dual-Model Evaluation with Semantic/LIME logging (v13-ready)

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime

# -------------------------
# HITL Feedback
# -------------------------
def hitl_feedback(df_results: pd.DataFrame) -> pd.DataFrame:
    df_results = df_results.copy()
    for col in ["hitl_override_prediction_email", "hitl_override_prediction_url", "hitl_semantic_features"]:
        if col not in df_results.columns:
            df_results[col] = np.nan
    for col in ["hitl_comments", "dual_model_notes"]:
        if col not in df_results.columns:
            df_results[col] = ""
    return df_results

def save_hitl_feedback(df_results: pd.DataFrame, filepath: str = None) -> str:
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"hitl_feedback_{timestamp}.csv"
    df_results.to_csv(filepath, index=False)
    return f"✅ HITL feedback saved to {filepath}"

# -------------------------
# Single Model Metrics
# -------------------------
def evaluate_single_model(y_true=None, y_pred=None, lime_lists=None, shap_vals_list=None, labels=None) -> dict:
    metrics = {}
    if y_true is not None and y_pred is not None:
        y_true = pd.Series(y_true).astype(str)
        y_pred = pd.Series(y_pred).astype(str)
        if labels is None:
            labels = list(sorted(set(y_true).union(set(y_pred))))
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels)
        }
    else:
        metrics = {k:0 for k in ["accuracy","precision","recall","f1_score","confusion_matrix"]}

    # LIME contradictions
    if lime_lists:
        contradiction_counts = [sum("⚠️ Unexpected effect" in str(line) for line in lime_exp)
                                for lime_exp in lime_lists if lime_exp]
        metrics["avg_lime_contradictions_per_row"] = float(np.mean(contradiction_counts)) if contradiction_counts else 0
    else:
        metrics["avg_lime_contradictions_per_row"] = 0

    # SHAP magnitude
    if shap_vals_list:
        shap_mags = []
        for vals in shap_vals_list:
            if vals is None: continue
            flat_vals = []
            if isinstance(vals,list):
                for v in vals:
                    flat_vals.extend(np.abs(np.array(v).flatten()))
            else:
                flat_vals.extend(np.abs(np.array(vals).flatten()))
            if flat_vals: shap_mags.append(np.mean(flat_vals))
        metrics["avg_shap_magnitude"] = float(np.mean(shap_mags)) if shap_mags else 0
    else:
        metrics["avg_shap_magnitude"] = 0

    return metrics

# -------------------------
# Dual Model Metrics
# -------------------------
def evaluate_dual_model(df_results: pd.DataFrame) -> dict:
    df_eval = df_results.copy()
    email_col = next((c for c in df_eval.columns if "email_prediction" in c), None)
    url_col = next((c for c in df_eval.columns if "url_prediction" in c), None)
    if email_col is None or url_col is None:
        return {"summary": {}, "details": None}

    df_eval = df_eval.dropna(subset=[email_col, url_col])
    total = len(df_eval)
    if total==0: return {"summary": {}, "details": None}

    email_pred = df_eval[email_col].astype(str).str.lower()
    url_pred = df_eval[url_col].astype(str).str.lower()
    agreement_mask = email_pred==url_pred

    email_risk = df_eval.get("email_risk_percent", pd.Series(0,index=df_eval.index))
    url_risk = df_eval.get("url_risk_percent", pd.Series(0,index=df_eval.index))

    # New graded disagreement levels
    full_disagreement = (~agreement_mask) & (email_risk>50) & (url_risk>50)
    partial_disagreement = (~agreement_mask) & ~full_disagreement
    agreement_mask_lowrisk = agreement_mask & (email_risk<=50) & (url_risk<=50)

    df_eval["agreement_level"] = "Agreement"
    df_eval.loc[partial_disagreement,"agreement_level"] = "Partial Disagreement"
    df_eval.loc[full_disagreement,"agreement_level"] = "Full Disagreement"
    df_eval.loc[agreement_mask_lowrisk,"agreement_level"] = "Low-risk Agreement"

    # Dual LIME & SHAP
    dual_lime_contradictions = []
    dual_shap_diff = []

    for _, row in df_eval.iterrows():
        e_lime = row.get("email_lime") or []
        u_lime = row.get("url_lime") or []
        dual_lime_contradictions.append(sum("⚠️ Unexpected effect" in str(l) for l in e_lime+u_lime))

        # SHAP difference safe
        e_shap = row.get("email_shap_values")
        u_shap = row.get("url_shap_values")
        try:
            e_vals = np.concatenate([np.abs(np.array(v).flatten()) for v in e_shap]) if isinstance(e_shap,list) else np.abs(np.array(e_shap).flatten())
            u_vals = np.concatenate([np.abs(np.array(v).flatten()) for v in u_shap]) if isinstance(u_shap,list) else np.abs(np.array(u_shap).flatten())
            min_len = min(len(e_vals),len(u_vals))
            dual_shap_diff.append(np.mean(np.abs(e_vals[:min_len]-u_vals[:min_len])) if min_len>0 else np.nan)
        except:
            dual_shap_diff.append(np.nan)

    df_eval["dual_lime_contradictions"] = dual_lime_contradictions
    df_eval["dual_shap_diff"] = dual_shap_diff

    summary = {
        "total_evaluated": total,
        "agreement_pct": round((agreement_mask & ~full_disagreement).mean()*100,2),
        "partial_disagreement_pct": round(partial_disagreement.mean()*100,2),
        "full_disagreement_pct": round(full_disagreement.mean()*100,2),
        "low_risk_agreement_count": int(agreement_mask_lowrisk.sum()),
        "avg_dual_lime_contradictions": float(np.nanmean(dual_lime_contradictions)),
        "avg_dual_shap_diff": float(np.nanmean([d for d in dual_shap_diff if not np.isnan(d)])) if dual_shap_diff else 0
    }

    return {"summary": summary,"details": df_eval}

# -------------------------
# Pretty Table
# -------------------------
def pretty_evaluation_table(metrics_dict: dict, model_name: str="Model") -> pd.DataFrame:
    rows=[
        ["Accuracy", metrics_dict.get("accuracy","N/A")],
        ["Precision", metrics_dict.get("precision","N/A")],
        ["Recall", metrics_dict.get("recall","N/A")],
        ["F1 Score", metrics_dict.get("f1_score","N/A")],
        ["Avg LIME Contradictions / Row", metrics_dict.get("avg_lime_contradictions_per_row","N/A")],
        ["Avg SHAP Magnitude", metrics_dict.get("avg_shap_magnitude","N/A")],
        ["Avg Dual LIME Contradictions", metrics_dict.get("avg_dual_lime_contradictions","N/A")],
        ["Avg Dual SHAP Difference", metrics_dict.get("avg_dual_shap_diff","N/A")],
        ["Agreement %", metrics_dict.get("agreement_pct","N/A")],
        ["Partial Disagreement %", metrics_dict.get("partial_disagreement_pct","N/A")],
        ["Full Disagreement %", metrics_dict.get("full_disagreement_pct","N/A")],
        ["Low-risk Agreement Count", metrics_dict.get("low_risk_agreement_count","N/A")]
    ]
    return pd.DataFrame(rows,columns=[f"{model_name} Metric","Value"])
