# =============================================
# Model Comparison: MLP vs XGBoost vs RandomForest
# =============================================

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Paths
RESULTS_DIR = "results"
DATA_PATH = "data/consolidated_traffic_data.csv"
TARGET_COL = "traffic_type"

# Features
features = [
    "duration", "total_fiat", "total_biat", "min_fiat", "min_biat",
    "max_fiat", "max_biat", "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond",
    "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle"
]

# 1) Load dataset
df = pd.read_csv(DATA_PATH)
X = df[features]
y = df[TARGET_COL]

# Train/test split (same random_state for consistency)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Load preprocessing objects
scaler = joblib.load(os.path.join(RESULTS_DIR, "scaler.pkl"))
le = joblib.load(os.path.join(RESULTS_DIR, "label_encoder.pkl"))

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_test_encoded = le.transform(y_test)

# 3) Load models
mlp = joblib.load(os.path.join(RESULTS_DIR, "mlp_model.joblib"))
xgb_clf = joblib.load(os.path.join(RESULTS_DIR, "xgboost_model.joblib"))
rf_model = joblib.load("random_forest_model.pkl")

# 4) Predictions
preds = {
    "MLP": mlp.predict(X_test_scaled),
    "XGBoost": xgb_clf.predict(X_test_scaled),
    "RandomForest": rf_model.predict(X_test)  # RF trained on raw features
}

# 5) Evaluation
results = {}
for model_name, y_pred in preds.items():
    acc = accuracy_score(y_test_encoded, y_pred) if model_name != "RandomForest" else accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test_encoded if model_name != "RandomForest" else y_test,
        y_pred,
        target_names=le.classes_,
        output_dict=True
    )
    cm = confusion_matrix(
        y_test_encoded if model_name != "RandomForest" else y_test,
        y_pred
    )
    results[model_name] = {"accuracy": acc, "report": report, "confusion_matrix": cm}

# 6) Accuracy bar chart


acc_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[m]["accuracy"] for m in results]
})

plt.figure(figsize=(6, 4))
sns.barplot(data=acc_df, x="Model", y="Accuracy", legend=False)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()


# 7) Confusion matrices
for model_name, data in results.items():
    plt.figure(figsize=(6, 4))
    sns.heatmap(data["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# 8) Precision/Recall/F1 comparison
metrics_df = pd.DataFrame({
    model: {
        "precision": data["report"]["weighted avg"]["precision"],
        "recall": data["report"]["weighted avg"]["recall"],
        "f1-score": data["report"]["weighted avg"]["f1-score"],
        "accuracy": data["accuracy"]
    }
    for model, data in results.items()
}).T

plt.figure(figsize=(8, 5))
metrics_df[["precision", "recall", "f1-score", "accuracy"]].plot(kind="bar")
plt.title("Precision, Recall, F1, Accuracy Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print("\n=== Metrics Summary ===")
print(metrics_df.round(4))
