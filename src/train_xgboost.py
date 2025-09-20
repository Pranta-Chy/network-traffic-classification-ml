"""
train_xgboost.py
Trains an XGBoost classifier (XGBClassifier), evaluates it, saves model and metrics.
"""

# Imports
import os
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Paths & config
DATA_PATH = "data/consolidated_traffic_data.csv"
RESULTS_DIR = "results"
TARGET_COL = "traffic_type"

os.makedirs(RESULTS_DIR, exist_ok=True)

features = [
    "duration", "total_fiat", "total_biat", "min_fiat", "min_biat",
    "max_fiat", "max_biat", "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond",
    "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle"
]

# 1) Load
df = pd.read_csv(DATA_PATH)

# 2) Prepare features/labels
X = df[features].copy()
y_raw = df[TARGET_COL].copy()
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, os.path.join(RESULTS_DIR, "label_encoder.pkl"))

# 3) Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4) Scaling (XGBoost does not require scaling, but using a scaler keeps consistency if you use the same preprocessor later)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(RESULTS_DIR, "scaler.pkl"))

# 5) Train XGBoost using scikit-learn wrapper (easy to save with joblib)
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    eval_metric='mlogloss',
    random_state=42
)
xgb_clf.fit(X_train_scaled, y_train)

# 6) Predict & evaluate
y_pred = xgb_clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

print("XGBoost Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# 7) Save metrics & plot
with open(os.path.join(RESULTS_DIR, "xgboost_metrics.json"), "w") as f:
    json.dump({"accuracy": acc, "classification_report": report}, f, indent=4)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("XGBoost - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "xgboost_confusion_matrix.png"))
plt.close()

# 8) Save model
joblib.dump(xgb_clf, os.path.join(RESULTS_DIR, "xgboost_model.joblib"))
print(f"\nXGBoost model saved to {os.path.join(RESULTS_DIR, 'xgboost_model.joblib')}")
joblib.dump(xgb_clf, "xgboost_model.pkl")

y_pred_xgb = xgb_clf.predict(X_test_scaled)

