# utils3.py — Feature extraction and helpers (enhanced for XAI layer)

import pandas as pd
import numpy as np
import re
import math
from urllib.parse import urlparse

# -------------------------
# Feature definitions
# -------------------------
EMAIL_FEATURES = [
    'num_words',
    'num_unique_words',
    'num_stopwords',
    'num_links',
    'num_unique_domains',
    'num_email_addresses',
    'num_spelling_errors',
    'num_urgent_keywords',
    'num_malicious_patterns'  # NEW: aligns with enhanced XAI
]

URL_FEATURES = [
    'url_length',
    'num_subdomains',
    'has_ip',
    'num_special_chars',
    'num_suspicious_keywords',
    'domain_entropy'
]

# -------------------------
# Stopwords set (lightweight)
# -------------------------
STOPWORDS = set([
    'the', 'and', 'in', 'to', 'of', 'a', 'for', 'on', 'is',
    'with', 'this', 'that', 'at', 'by', 'an', 'be', 'as', 'from'
])

# -------------------------
# Malicious patterns for semantic detection
# -------------------------
MALICIOUS_PATTERNS = ['.php', '.exe', '.js', '.scr', 'invoice', 'attachment']

# -------------------------
# Email feature extraction
# -------------------------
def extract_email_features(row):
    """
    Extracts email features from a dict or pandas Series.
    Returns a 1-row DataFrame matching EMAIL_FEATURES order.
    """
    features = {}
    text = str(row.get('body','') if isinstance(row, dict) else row.get('body',''))
    subject = str(row.get('subject','') if isinstance(row, dict) else row.get('subject',''))
    full_text = (subject + " " + text).strip()

    # word-level
    words = full_text.split()
    features['num_words'] = len(words)
    features['num_unique_words'] = len(set(words))
    features['num_stopwords'] = sum(1 for w in words if w.lower() in STOPWORDS)

    # links & emails
    urls = re.findall(r'http[s]?://\S+', full_text)
    features['num_links'] = len(urls)
    emails = re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', full_text)
    features['num_email_addresses'] = len(emails)

    domains = set()
    for u in urls:
        try:
            dom = urlparse(u).netloc
            if dom:
                domains.add(dom)
        except:
            continue
    features['num_unique_domains'] = len(domains)

    # spelling errors
    features['num_spelling_errors'] = sum(1 for w in words if not w.isalpha())

    # urgent keywords
    urgent_keywords = ['urgent', 'verify', 'update', 'account', 'password', 'alert']
    features['num_urgent_keywords'] = sum(1 for w in words if w.lower() in urgent_keywords)

    # malicious patterns
    full_text_lower = full_text.lower()
    features['num_malicious_patterns'] = sum(full_text_lower.count(p) for p in MALICIOUS_PATTERNS)

    return pd.DataFrame([features], columns=EMAIL_FEATURES)

# -------------------------
# URL feature extraction
# -------------------------
def extract_url_features(row):
    """
    Extracts URL features from dict or pandas Series.
    Returns a 1-row DataFrame matching URL_FEATURES order.
    """
    features = {}
    url_text = None

    # detect URL field
    keys_to_check = ('urls','url','URL','link','links')
    if isinstance(row, dict):
        for k in keys_to_check:
            if k in row and row[k]:
                url_text = str(row[k])
                break
    else:
        for k in keys_to_check:
            if k in row.index and pd.notna(row[k]) and row[k] != '':
                url_text = str(row[k])
                break
    if url_text is None:
        url_text = ''

    # basic features
    features['url_length'] = len(url_text)
    features['num_subdomains'] = max(url_text.count('.') - 1, 0)
    features['has_ip'] = int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url_text)))
    features['num_special_chars'] = sum(1 for c in url_text if not c.isalnum())
    features['num_suspicious_keywords'] = sum(
        1 for kw in ['login', 'secure', 'update', 'verify', 'bank', 'account',
                     '.php', '.exe', '.js', '.scr', 'invoice', 'attachment']  # NEW patterns
        if kw in url_text.lower()
    )
    features['domain_entropy'] = calculate_entropy(urlparse(url_text).netloc if url_text else '')

    return pd.DataFrame([features], columns=URL_FEATURES)

# -------------------------
# Entropy calculation
# -------------------------
def calculate_entropy(domain):
    if not domain:
        return 0.0
    probs = [float(domain.count(c))/len(domain) for c in set(domain)]
    return -sum([p*math.log2(p) for p in probs if p>0])

# -------------------------
# Feature alignment for model
# -------------------------
def prepare_features_for_model(features_df, model):
    """
    Align features to model.feature_names_in_ if exists, fill missing with zeros.
    """
    try:
        expected = list(model.feature_names_in_)
        features_df = features_df.reindex(columns=expected, fill_value=0)
    except AttributeError:
        pass
    return features_df

# -------------------------
# Map prediction to department
# -------------------------
def assign_department(prediction_label: str) -> str:
    label = str(prediction_label).lower()
    if "phish" in label or str(prediction_label) == "1":
        return "Fraud Response"
    elif "malware" in label:
        return "Cybersecurity Ops"
    elif "deface" in label:
        return "Web Security"
    elif "benign" in label or label in ("0", "safe", "clean"):
        return "General Inbox"
    else:
        return "General Inbox"

# -------------------------
# Safe column retrieval
# -------------------------
def safe_get(df, col):
    if col in df.columns:
        return df[col]
    return pd.Series([np.nan]*len(df))
