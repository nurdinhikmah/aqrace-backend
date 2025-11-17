# interactive_test.py
import os
import sys
import time
import joblib
import requests
from urllib.parse import urlparse, parse_qs, unquote

# Add project path to import features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import extract_url_features

# ------------------------
# Config
# ------------------------
MODEL_PATH = '/Users/imma/Desktop/Backend/xgboost.pkl'
SHORTENER_DOMAINS = {"tinyurl.com", "bit.ly", "t.co", "goo.gl", "is.gd", "ow.ly"}
TRUSTED_DOMAINS = {
    "youtube.com", "music.youtube.com", "google.com", "gmail.com",
    "facebook.com", "instagram.com", "tiktok.com", "shopee.com"
}
THRESHOLD = 0.45  # tuned threshold to prioritise malicious recall

# ------------------------
# Load model
# ------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
feature_cols = model_data["features"]

print(f"✅ Model loaded: {MODEL_PATH}  (using {len(feature_cols)} features)\n")
print("Enter URL to classify (type 'exit' to quit)\n")

# ------------------------
# Helper Functions
# ------------------------
def normalise_input(u: str) -> str:
    u = u.strip()
    if not u:
        return u
    if not u.startswith(('http://', 'https://')):
        u = 'https://' + u
    parsed = urlparse(u)
    domain = parsed.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    rest = parsed.path or ""
    if parsed.query:
        rest += '?' + parsed.query
    if parsed.fragment:
        rest += '#' + parsed.fragment
    return f"{parsed.scheme}://{domain}{rest}"

def expand_url(url: str, timeout: int = 5) -> str:
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        final = resp.url
        if final and final != url:
            return final
        resp = requests.get(url, allow_redirects=True, timeout=timeout)
        return resp.url or url
    except:
        return url

def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def is_shortener_domain(url: str) -> bool:
    host = domain_from_url(url)
    return any(host == sd or host.endswith("." + sd) for sd in SHORTENER_DOMAINS)

def domain_is_trusted(url: str) -> bool:
    host = domain_from_url(url)
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)

def url_contains_redirect(url: str) -> bool:
    try:
        qs = parse_qs(urlparse(url).query, keep_blank_values=True)
        for vals in qs.values():
            for v in vals:
                v_dec = unquote(v)
                if "http://" in v_dec or "https://" in v_dec:
                    return True
        path = urlparse(url).path or ""
        if "http://" in path or "https://" in path or "http%3a" in path.lower() or "https%3a" in path.lower():
            return True
    except:
        return True
    return False

def classify_url(url: str):
    X = extract_url_features([url])
    X = X.reindex(columns=feature_cols, fill_value=0)
    proba = model.predict_proba(X)[:, 1][0]
    label = "Malicious" if proba > THRESHOLD else "Safe"
    return label, proba

# ------------------------
# Interactive Loop
# ------------------------
while True:
    try:
        user_input = input("🌐 Enter URL: ").strip()
    except:
        print("\n👋 Exiting...")
        break

    if not user_input:
        print("⚠️ Please enter a URL.\n")
        continue

    if user_input.lower() == "exit":
        print("\n👋 Exiting...")
        break

    user_url = normalise_input(user_input)

    if is_shortener_domain(user_url):
        expanded = expand_url(user_url)
        if expanded != user_url:
            print(f"🔗 Expanded short URL → {expanded}")
            user_url = expanded

    # SMART TRUST CHECK
    if domain_is_trusted(user_url) and not url_contains_redirect(user_url):
        print("\n✅ Classification Source: TRUST CHECK (not model)")
        print("Reason: Trusted domain with no external redirect.")
        print("Final Decision: SAFE ✅")
        print("----------------------------------------\n")
        continue

    # MODEL CLASSIFICATION
    try:
        label, proba = classify_url(user_url)
        proba = round(float(proba), 3)
        print("\n🤖 Classification Source: AI MODEL (XGBoost)")
        print(f"Final Decision: {label.upper()}")
        print(f"Malicious Probability Score: {proba}")
        print("----------------------------------------\n")
    except Exception as e:
        print(f"⚠️ Error classifying URL: {e}\n")
        time.sleep(0.5)
        continue
