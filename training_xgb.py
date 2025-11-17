# ==========================================
# IMPROVED TRAINING SCRIPT (FINAL DATASET)
# ==========================================
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Import feature extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features import extract_url_features

# ======================
# 1. Load dataset
# ======================
csv_path = '/Users/imma/Desktop/Backend/final_SAFE_MALICIOUS_500k.csv'  # ✅ update path
df = pd.read_csv(csv_path)
print(f"Dataset loaded: {df.shape}")

# ======================
# 2. Clean + prepare labels
# ======================
df = df.dropna(subset=['url', 'label'])
df['label'] = df['label'].astype(str).str.lower().str.strip()

# ✅ handle labels (safe → 0, malicious → 1)
df['label'] = df['label'].replace({'safe': 'benign'})  # uniform naming

print(f"After cleaning: {df.shape}")
print("Label distribution:")
print(df['label'].value_counts())

# ======================
# 3. Train-test split
# ======================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"Train size: {train_df.shape}, Test size: {test_df.shape}")

# ======================
# 4. Feature extraction
# ======================
print("Extracting features... (this may take a moment)")

X_train = extract_url_features(train_df['url'].tolist())
X_test = extract_url_features(test_df['url'].tolist())

# Buang kolum tak perlu
for col in ['TLD']:
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])
    if col in X_test.columns:
        X_test = X_test.drop(columns=[col])

print(f"Features extracted: Train {X_train.shape}, Test {X_test.shape}")

# Label encoding
y_train = train_df['label'].map({'benign': 0, 'malicious': 1})
y_test = test_df['label'].map({'benign': 0, 'malicious': 1})

# ======================
# 5. Handle missing values
# ======================
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# ======================
# 6. Train XGBoost model
# ======================
print("Training XGBoost model...")
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.08,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    tree_method='hist'
)
model.fit(X_train, y_train)

# ======================
# 7. Evaluate model
# ======================

# dapatkan probability untuk kelas malicious
proba = model.predict_proba(X_test)[:, 1]

# tetapkan threshold di sini 
threshold = 0.45

# classify
pred = (proba > threshold).astype(int)

acc = accuracy_score(y_test, pred)
print(f"\n✅ Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=['Safe', 'Malicious']))

# Confusion Matrix
cm = confusion_matrix(y_test, pred)
print("\nConfusion Matrix:")
print(cm)


# ======================
# 8. Save model
# ======================
model_data = {
    "model": model,
    "features": list(X_train.columns)
}

save_path = '/Users/imma/Desktop/Backend/xgboost.pkl'
joblib.dump(model_data, save_path)
print(f"\n✅ Model saved successfully at: {save_path}")
