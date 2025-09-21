# xai_layer15.py — Robust Dual-Model XAI Layer with semantic LIME enhancements + contradiction flags

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from lime.lime_tabular import LimeTabularExplainer
from utils4 import (
    extract_email_features,
    extract_url_features,
    prepare_features_for_model,
    assign_department,
    EMAIL_FEATURES,
    URL_FEATURES
)

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Model paths
# -------------------------
EMAIL_MODEL_PATH = "data/New models/email_rf_final.pkl"
URL_MODEL_PATH = "data/New models/url_rf_explainable_v2.pkl"

def safe_load_model(path):
    if not os.path.exists(path):
        print(f"❌ Model not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

email_model = safe_load_model(EMAIL_MODEL_PATH)
url_model = safe_load_model(URL_MODEL_PATH)

print("✅ Email Model loaded" if email_model else "❌ Email Model failed")
print("✅ URL Model loaded" if url_model else "❌ URL Model failed")

# -------------------------
# Semantic heuristics
# -------------------------
SUSPICIOUS_KEYWORDS = ['login','secure','update','verify','bank','account','confirm','reset','password']
MALWARE_KEYWORDS = ['invoice','attachment','download','install','exe','script']
MALICIOUS_PATTERNS = ['.php', '.exe', '.cmd', '.js', '.scr', 'invoice', 'attachment']

def extract_semantic_features(row):
    """Lightweight semantic + extension-based heuristics"""
    text = ""
    if isinstance(row, dict):
        text = " ".join([str(row.get(k,"")) for k in ["subject","body","content","url"] if k in row])
    else:
        text = " ".join([str(row.get(k,"")) for k in row.index if k in ["subject","body","content","url"]])
    text = text.lower()

    return {
        "num_suspicious_keywords": sum(text.count(kw) for kw in SUSPICIOUS_KEYWORDS),
        "num_malware_keywords": sum(text.count(kw) for kw in MALWARE_KEYWORDS),
        "num_exclamations": text.count("!"),
        "num_uppercase_words": sum(1 for w in text.split() if w.isupper()),
        "num_malicious_patterns": sum(text.count(p) for p in MALICIOUS_PATTERNS)
    }

# -------------------------
# Risk computation with semantic boost
# -------------------------
def compute_risk(probs, classes, model_type="email", semantic_boost=None):
    try:
        probs = np.array(probs, dtype=float)
        if probs.size == 0:
            return 0.0

        if model_type=="email":
            idx = int(np.argmax(probs)) if not hasattr(classes,"__iter__") else list(classes).index(1) if 1 in classes else int(np.argmax(probs))
            base_risk = min(probs[idx]*100, 99.9)
        else:
            weight_map = {0:0.0,1:0.3,2:0.7,3:1.0}
            base_risk = sum([p*weight_map.get(i,0.5) for i,p in enumerate(probs)])*100
            base_risk = min(base_risk, 99.9)

        if semantic_boost:
            boost = sum(semantic_boost.values()) * 2
            base_risk = min(base_risk + boost, 99.9)

        return round(base_risk,1)
    except:
        return 0.0

# -------------------------
# LIME Explainer
# -------------------------
def build_lime_explainer(feat_df, class_names):
    X_train = feat_df.values.astype(float)
    if X_train.ndim==1:
        X_train = X_train.reshape(1,-1)
    return LimeTabularExplainer(
        training_data=X_train,
        feature_names=list(feat_df.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=False
    )

# -------------------------
# Feature groups for LIME
# -------------------------
FEATURE_GROUPS = {
    "Urgent content":["num_urgent_keywords","num_exclamations"],
    "Links / URLs":["num_links","num_unique_domains","has_ip"],
    "Capitalization":["num_uppercase_words"],
    "Spelling / anomalies":["num_spelling_errors","num_nonalpha_words"],
    "Semantic phishing indicators":["num_suspicious_keywords","num_malware_keywords","num_malicious_patterns"],
    "Other":[]
}

def _base_feature_name(feat_string):
    return feat_string.split()[0] if feat_string else feat_string

# -------------------------
# LIME Cleanup + Contradictions
# -------------------------
def clean_lime(exp_list, model_type="email", row=None, top_n=5, semantic=None, dual_context=None):
    sorted_exp = sorted(exp_list,key=lambda x: -abs(x[1]))[:top_n]
    grouped_scores = {g:0.0 for g in FEATURE_GROUPS}
    lines = []

    positive_signals, negative_signals = [], []

    for feat_str, weight in sorted_exp:
        if abs(weight)<1e-5: continue
        base_feat = _base_feature_name(feat_str)
        group_name = next((g for g,flist in FEATURE_GROUPS.items() if base_feat in flist),"Other")
        grouped_scores[group_name]+=weight
        sign = "increases risk" if weight>0 else "reduces risk"
        evidence_val = row.get(base_feat) if row is not None and hasattr(row,"get") else row[base_feat] if row is not None else None
        evidence = f"(value: {evidence_val})" if evidence_val is not None else ""
        lines.append(f"Feature **{base_feat}** {sign} {evidence} (weight {weight:.3f})")

        if weight > 0: positive_signals.append(base_feat)
        else: negative_signals.append(base_feat)

    # Contradiction detection
    if positive_signals and negative_signals:
        lines.append(f"⚠️ Contradiction detected: {', '.join(positive_signals)} suggest risk ↑ but {', '.join(negative_signals)} suggest risk ↓.")

    if semantic:
        for k,v in semantic.items():
            if v:
                lines.append(f"Semantic indicator: **{k}** detected {v} time(s)")

    if dual_context:
        email_pred,url_pred = str(dual_context["email_pred"]).lower(),str(dual_context["url_pred"]).lower()
        email_risk,url_risk = dual_context["email_risk"],dual_context["url_risk"]
        if email_pred==url_pred:
            lines.append(f"⚡ Dual Agreement: both predicted '{email_pred}' (email {email_risk}%, url {url_risk}%)")
        elif email_risk>50 and url_risk>50:
            lines.append(f"⚡ Full Disagreement: email '{email_pred}' vs url '{url_pred}' (both high risk)")
        else:
            lines.append(f"⚡ Partial Disagreement: email '{email_pred}' vs url '{url_pred}'")

    for g,total in grouped_scores.items():
        if abs(total)<1e-5: continue
        sign = "↑ increases risk" if total>0 else "↓ reduces risk"
        lines.append(f"⚡ {g}: {sign} (cumulative {total:.3f})")

    lines.append("🔎 Overall risk assessment based on above signals")
    return lines

# -------------------------
# Single-row analysis
# -------------------------
def analyze_single_row(row, model_type="email", model=None, dual_row=None, max_lime=5):
    mdl = model if model else (email_model if model_type=="email" else url_model)
    if mdl is None:
        return {"error":"Model not loaded"}

    feat_df = extract_email_features(row) if model_type=="email" else extract_url_features(row)
    feat_df = prepare_features_for_model(feat_df, mdl)
    X = feat_df.values.astype(float)

    pred,probs = None,[]
    try:
        if hasattr(mdl,"predict"): pred=mdl.predict(X)[0]
        if hasattr(mdl,"predict_proba"): probs=mdl.predict_proba(X)[0]
    except: pass

    semantic = extract_semantic_features(row)
    risk = compute_risk(probs,getattr(mdl,"classes_",[]),model_type, semantic_boost=semantic)
    department = assign_department(pred)

    dual_context = None
    if dual_row is not None:
        mdl2 = url_model if model_type=="email" else email_model
        feat_df2 = extract_url_features(dual_row) if model_type=="email" else extract_email_features(dual_row)
        feat_df2 = prepare_features_for_model(feat_df2, mdl2)
        X2 = feat_df2.values.astype(float)
        pred2,probs2=None,[]
        try:
            if hasattr(mdl2,"predict"): pred2=mdl2.predict(X2)[0]
            if hasattr(mdl2,"predict_proba"): probs2=mdl2.predict_proba(X2)[0]
        except: pass
        risk2 = compute_risk(probs2,getattr(mdl2,"classes_",[]),"url" if model_type=="email" else "email")
        dual_context = {
            "email_pred": pred if model_type=="email" else pred2,
            "url_pred": pred2 if model_type=="email" else pred,
            "email_risk": risk if model_type=="email" else risk2,
            "url_risk": risk2 if model_type=="email" else risk
        }

    lime_lines=[]
    try:
        class_names=getattr(mdl,"classes_",["benign","malicious"])
        explainer=build_lime_explainer(feat_df,class_names)
        exp=explainer.explain_instance(X[0],mdl.predict_proba,num_features=max_lime,num_samples=500)
        lime_lines=clean_lime(exp.as_list(),model_type=model_type,row=feat_df.iloc[0],
                              top_n=max_lime,semantic=semantic,dual_context=dual_context)
    except Exception as e:
        heur=[f"{k} (heuristic count: {v}) → possible indicator" for k,v in semantic.items() if v]
        if dual_context:
            heur.append(f"⚡ Dual Disagreement detected: email '{dual_context['email_pred']}' vs url '{dual_context['url_pred']}'")
        heur.append(f"⚠️ LIME explanation failed: {str(e)}")
        lime_lines=heur

    content = ""
    if isinstance(row, dict):
        content = row.get("body",row.get("subject","")) if model_type=="email" else row.get("url","")
    else:
        content = row.get("body",row.get("subject","")) if model_type=="email" else row.get("url","")

    return {
        "prediction":pred,
        "risk_percent":risk,
        "department":department,
        "lime":lime_lines,
        "content":content
    }
