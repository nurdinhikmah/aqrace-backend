# ==========================================
# INTERACTIVE MODEL TESTING (USER INPUT)
# ==========================================

import os
import sys
import joblib
import pandas as pd

# Import feature extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import extract_url_features

# ======================
# 1. Load trained model
# ======================
model_path = '/Users/imma/Desktop/Backend/ranfor.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

model_data = joblib.load(model_path)
model = model_data["model"]
feature_cols = model_data["features"]

print("✅ Model loaded successfully!")
print(f"Features used: {len(feature_cols)}\n")

# ======================
# 2. Interactive loop
# ======================
print("🔍 URL Classification System")
print("========================================")
print("Enter URLs to classify (type 'exit' to quit)")
print("========================================\n")

while True:
    url = input("🌐 Enter URL: ").strip()
    if url.lower() in ['exit', 'quit', 'q']:
        print("\n👋 Exiting URL classifier.")
        break
    if not url:
        print("⚠️ Please enter a valid URL.\n")
        continue

    try:
        # Normalise user input — tambah scheme & buang www.
        from urllib.parse import urlparse

        def normalise_input(u):
            u = u.strip()
            if not u.startswith(('http://', 'https://')):
                u = 'http://' + u  # default to http
            parsed = urlparse(u)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            normalised = f"{parsed.scheme}://{domain}{parsed.path}"
            return normalised

        url = normalise_input(url)

        # Extract features from user input
        X_test = extract_url_features([url])
        X_test = X_test[feature_cols].fillna(0)

        # Predict
        prob = model.predict_proba(X_test)[0, 1]  # probability of malicious
        label = "MALICIOUS" if prob > 0.5 else "SAFE"

        # Confidence level
        if prob >= 0.85 or prob <= 0.15:
            confidence = "HIGH"
        elif 0.6 <= prob < 0.85 or 0.15 < prob <= 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # Print results
        print(f"\n📊 Result: {label}")
        print(f"🎯 Malicious Probability: {prob:.3f}")
        print(f"🔒 Confidence: {confidence}")
        print("----------------------------------------\n")

    except Exception as e:
        print(f"⚠️ Error processing URL: {e}\n")
