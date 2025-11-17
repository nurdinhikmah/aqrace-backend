from flask import Flask, request, jsonify
import joblib
import requests
import pandas as pd
from flask_cors import CORS
from urllib.parse import urlparse, parse_qs, unquote
from features import extract_url_features
from pyzbar.pyzbar import decode
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load Model
model_data = joblib.load('xgboost.pkl')
model = model_data["model"]
feature_cols = model_data["features"]

# ------------------------
# Config (same as interactive_test.py)
# ------------------------
SHORTENER_DOMAINS = {"tinyurl.com", "bit.ly", "t.co", "goo.gl", "is.gd", "ow.ly"}
TRUSTED_DOMAINS = {
    "youtube.com", "music.youtube.com", "google.com", "gmail.com",
    "facebook.com", "instagram.com", "tiktok.com", "shopee.com"
}
THRESHOLD = 0.45

# ------------------------
# Helper Functions
# ------------------------

def normalise_input(u):
    u = u.strip()
    if not u.startswith(('http://', 'https://')):
        u = 'https://' + u
    parsed = urlparse(u)
    domain = parsed.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    rest = parsed.path or ""
    if parsed.query:
        rest += '?' + parsed.query
    return f"{parsed.scheme}://{domain}{rest}"

def expand_url(url):
    try:
        resp = requests.head(url, allow_redirects=True, timeout=5)
        if resp.url and resp.url != url:
            return resp.url
        resp = requests.get(url, allow_redirects=True, timeout=5)
        return resp.url or url
    except:
        return url

def domain_from_url(url):
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def is_shortener_domain(url):
    host = domain_from_url(url)
    return any(host == sd or host.endswith("." + sd) for sd in SHORTENER_DOMAINS)

def domain_is_trusted(url):
    host = domain_from_url(url)
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)

def url_contains_redirect(url):
    qs = parse_qs(urlparse(url).query)
    for vals in qs.values():
        for v in vals:
            v_dec = unquote(v)
            if "http://" in v_dec or "https://" in v_dec:
                return True
    return False

# ------------------------
# MAIN CLASSIFIER LOGIC
# ------------------------
def classify_url(url):
    url = normalise_input(url)

    # Step 1: Expand shorteners
    if is_shortener_domain(url):
        url = expand_url(url)

    # Step 2: TRSUTED DOMAIN CHECK
    if domain_is_trusted(url) and not url_contains_redirect(url):
        return {
            "url": url,
            "label": "Safe",
            "probability": 0.05,
            "confidence": "HIGH",
            "classification_source": "TRUST CHECK"
        }

    # Step 3: AI MODEL CLASSIFICATION
    X = extract_url_features([url]).reindex(columns=feature_cols, fill_value=0)
    prob = float(model.predict_proba(X)[0, 1])
    label = "Malicious" if prob > THRESHOLD else "Safe"

    confidence = (
        "HIGH" if prob >= 0.85 or prob <= 0.15 else
        "MEDIUM" if 0.6 <= prob < 0.85 or 0.15 < prob <= 0.4 else
        "LOW"
    )

    return {
        "url": url,
        "label": label,
        "probability": round(prob, 3),
        "confidence": confidence,
        "classification_source": "AI MODEL"
    }

# ------------------------
# ROUTES
# ------------------------

@app.route('/')
def home():
    return jsonify({"message": "✅ aQRace API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        url = data.get("url")
        if not url:
            return jsonify({"error": "Missing URL"}), 400

        result = classify_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/decode_and_predict', methods=['POST'])
def decode_and_predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        decoded_objects = decode(image)
        if not decoded_objects:
            return jsonify({"error": "Failed to decode QR code"}), 400

        qr_text = decoded_objects[0].data.decode('utf-8')

        result = classify_url(qr_text)
        result["decoded_from_qr"] = qr_text

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
