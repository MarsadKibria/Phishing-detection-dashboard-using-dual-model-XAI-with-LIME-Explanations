# Dashboard Logic
import streamlit as st
import pandas as pd
import numpy as np
from xai_layer16 import analyze_single_row, email_model, url_model
from dashboardutils5 import (
    hitl_feedback, save_hitl_feedback,
    evaluate_single_model, evaluate_dual_model, pretty_evaluation_table
)

#
# Dark Blue SOC Theme
#
PRIMARY_BG = "#0b1e4a"
SECONDARY_BG = "#142d5c"
TEXT_COLOR = "#ffffff"
HIGHLIGHT = "#ffdd57"
SUBTITLE = "#00ffcc"
ACCENT = "#39ff14"
DANGER = "#ff5555"
SAFE = "#55ff55"
SEMANTIC = "#66ffff"
CONTRADICT = "#ffa500"  # Orange for contradictions

st.set_page_config(page_title="XAI Dashboard v16 SOC", layout="wide")

#
# CSS Styling
#
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{ background-color: {PRIMARY_BG}; }}
[data-testid="stSidebar"] {{ background-color: {SECONDARY_BG}; }}
body, .stText, p, span {{ color: {TEXT_COLOR}; font-family: 'Courier New', monospace; }}
h1, h2, h3, h4, h5 {{ color: {HIGHLIGHT} !important; }}
.stButton>button {{ background-color: {ACCENT}; color: {PRIMARY_BG}; font-weight: bold; }}
.lime-text {{ color: {SEMANTIC}; }}
.danger-text {{ color: {DANGER}; }}
.safe-text {{ color: {SAFE}; }}
.contradiction-text {{ color: {CONTRADICT}; font-weight: bold; }}
</style>
""", unsafe_allow_html=True)


# Header

st.title("🛡️ XAI Dual-Model Security Dashboard (v16 SOC)")
st.markdown(
    f"<span style='color:{SUBTITLE}'>Upload Emails/URLs for dual-model risk analysis with enhanced LIME insights and contradiction detection.</span>",
    unsafe_allow_html=True
)


# CSV Upload

uploaded_file = st.file_uploader("Upload CSV (Max 1000 rows)", type=["csv"])
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file).head(1000)
st.success(f"✅ CSV loaded with {len(df)} rows.", icon="✅")

has_email = any(c.lower() in ["body", "subject", "content"] for c in df.columns)
has_url = any(c.lower() in ["url", "urls", "link", "links"] for c in df.columns)
if not (has_email or has_url):
    st.error("CSV must contain 'body'/'subject' or 'url'.")
    st.stop()
st.info(f"Detected columns for: {', '.join([t for t,p in zip(['Email','URL'], [has_email,has_url]) if p])}")


# Row-wise Analysis (no heavy LIME here)

results = []
for row in df.itertuples(index=False):
    combined = {}
    row_dict = row._asdict() if hasattr(row, "_asdict") else {f: getattr(row,f) for f in row._fields}

    if has_email:
        res_email = analyze_single_row(row_dict, "email", model=email_model,
                                       dual_row=row_dict if has_url else None, max_lime=0)
        combined.update({
            "email_prediction": res_email.get("prediction", "N/A"),
            "email_risk_percent": res_email.get("risk_percent", 0),
            "email_department": res_email.get("department", "General Inbox"),
            "email_content": res_email.get("content", "")
        })

    if has_url:
        res_url = analyze_single_row(row_dict, "url", model=url_model,
                                     dual_row=row_dict if has_email else None, max_lime=0)
        combined.update({
            "url_prediction": res_url.get("prediction", "N/A"),
            "url_risk_percent": res_url.get("risk_percent", 0),
            "url_department": res_url.get("department", "General Inbox"),
            "url_content": res_url.get("content", "")
        })

    # Dual-model flags
    if has_email and has_url:
        e_pred, u_pred = str(combined["email_prediction"]).lower(), str(combined["url_prediction"]).lower()
        e_risk, u_risk = combined["email_risk_percent"], combined["url_risk_percent"]
        if e_pred == u_pred:
            combined["dual_model_flag"] = "Agreement"
        elif e_risk > 50 and u_risk > 50:
            combined["dual_model_flag"] = "High-Risk Disagreement"
        else:
            combined["dual_model_flag"] = "Partial/Low-Risk Disagreement"
    else:
        combined["dual_model_flag"] = "N/A"

    results.append(combined)

result_df = pd.DataFrame(results)

# -------------------------
# Risk Table
# -------------------------
st.subheader("✅ Risk Table")
display_cols = [c for c in result_df.columns if
                "prediction" in c or "risk_percent" in c or "department" in c or "dual_model_flag" in c or "content" in c]
st.dataframe(result_df[display_cols].fillna("N/A"), use_container_width=True)

# -------------------------
# Helper: Safe LIME Display with Contradictions
# -------------------------
def lime_display(row, model_type):
    try:
        res = analyze_single_row(row, model_type,
                                 model=email_model if model_type=="email" else url_model,
                                 max_lime=5)
        lines, contradictions = [], []

        for l in res.get("lime", []):
            if "contradiction" in l.lower():
                contradictions.append(f"<span class='contradiction-text'>{l}</span>")
            elif "↑" in l:
                lines.append(f"<span class='danger-text'>{l}</span>")
            elif "↓" in l:
                lines.append(f"<span class='safe-text'>{l}</span>")
            else:
                lines.append(f"<span class='lime-text'>{l}</span>")

        explanation_html = "<br>".join(lines)
        contradiction_html = ""
        if contradictions:
            contradiction_html = "<br><b>⚠️ Contradictions Detected:</b><br>" + "<br>".join(contradictions)

        return explanation_html + contradiction_html

    except Exception as e:
        return f"<span class='danger-text'>⚠️ LIME failed: {str(e)}</span>"

# -------------------------
# Top 5 Risky Items
# -------------------------
st.subheader("🔥 Top 5 Risky Items")

def display_top(df_top, prefix):
    for idx, r in df_top.iterrows():
        st.markdown(
            f"**Row {idx}** — Risk: {r[f'{prefix}_risk_percent']}% — Dept: {r[f'{prefix}_department']} — Flag: {r.get('dual_model_flag','N/A')}",
            unsafe_allow_html=True
        )
        st.text_area("Content", r.get(f"{prefix}_content",""), height=120, key=f"content_{prefix}_{idx}")
        st.markdown(lime_display(df.iloc[idx], prefix), unsafe_allow_html=True)

if has_email:
    top_email = result_df.sort_values("email_risk_percent", ascending=False).head(5)
    st.markdown("### 📧 Email Top 5 Risks")
    display_top(top_email, "email")

if has_url:
    top_url = result_df.sort_values("url_risk_percent", ascending=False).head(5)
    st.markdown("### 🌐 URL Top 5 Risks")
    display_top(top_url, "url")

# -------------------------
# Inspect Row LIME
# -------------------------
st.subheader("🔍 Inspect Row LIME")
row_number = st.number_input("Select Row Number", 0, len(df)-1, 0, key="inspect_row")
row_sel = df.iloc[row_number]
if has_email: st.markdown("**Email Body / Subject:**"); st.text_area("Email Content", row_sel.get("body","N/A"), height=120, key="inspect_email")
if has_url: st.markdown("**URL:**"); st.text(row_sel.get("url","N/A"))
if st.button("Generate LIME Explanation", key="lime_btn"):
    if has_email: st.markdown(lime_display(row_sel, "email"), unsafe_allow_html=True)
    if has_url: st.markdown(lime_display(row_sel, "url"), unsafe_allow_html=True)

# -------------------------
# HITL Feedback
# -------------------------
st.subheader("✍️ Human-in-the-Loop Feedback")
row_number_hitl = st.number_input("Select Row for Feedback", 0, len(df)-1, 0, key="hitl_row")
comment = st.text_area("Add Comment / Correction", key="hitl_comment")
override = st.selectbox("Override Prediction", ["No Change","Benign","Phishing","Malware"], key="hitl_override")
if st.button("Save Feedback", key="hitl_save"):
    df_hitl = hitl_feedback(df.loc[[row_number_hitl]])
    df_hitl.at[df_hitl.index[0], "hitl_comments"] = comment
    row_features = analyze_single_row(df.iloc[row_number_hitl], "email" if has_email else "url",
                                      model=email_model if has_email else url_model, max_lime=5)
    df_hitl.at[df_hitl.index[0], "hitl_semantic_features"] = str(row_features.get("lime", []))
    if has_email: df_hitl.at[df_hitl.index[0], "hitl_override_prediction_email"] = override if override!="No Change" else np.nan
    if has_url: df_hitl.at[df_hitl.index[0], "hitl_override_prediction_url"] = override if override!="No Change" else np.nan
    save_hitl_feedback(df_hitl)
    st.success(f"✅ Feedback saved for Row {row_number_hitl}")

# -------------------------
# Evaluation Metrics
# -------------------------
st.subheader("📊 Evaluation Metrics")
if has_email:
    metrics_email = evaluate_single_model(result_df.get("email_prediction"), result_df.get("email_prediction"))
    st.markdown("### 📧 Email Model Metrics")
    st.table(pretty_evaluation_table(metrics_email,"Email Model"))

if has_url:
    metrics_url = evaluate_single_model(result_df.get("url_prediction"), result_df.get("url_prediction"))
    st.markdown("### 🌐 URL Model Metrics")
    st.table(pretty_evaluation_table(metrics_url,"URL Model"))

if has_email and has_url:
    dual_metrics = evaluate_dual_model(result_df)
    st.markdown("### 🔀 Dual-Model Metrics")
    if dual_metrics["details"] is not None:
        st.dataframe(dual_metrics["details"].head(50))
    st.info(f"Agreement: {dual_metrics['summary'].get('agreement_pct',0)}%, Disagreement: {dual_metrics['summary'].get('full_disagreement_pct',0)}%")

# -------------------------
# Legend
# -------------------------
st.markdown(
    f"<span style='color:{HIGHLIGHT}'>Legend:</span> 📧 Email | 🌐 URL | ⚠️ Flags | ↑ increase risk | ↓ reduce risk | Semantic = cyan | Contradiction = orange | HITL = manual corrections",
    unsafe_allow_html=True
)
