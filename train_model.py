import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

print("Loading data...")
# df = pd.read_csv("pose_dataset_yolo26.csv")
df = pd.read_csv("pose_dataset_yolo26_updated.csv")
X = df.drop(columns=["label", "video_id"])
y = df["label"]
groups = df["video_id"]

print("Training LightGBM model...")
# LightGBM with class balancing to handle the imbalanced dataset
clf = LGBMClassifier(
    n_estimators=1000,          # More trees for better learning
    learning_rate=0.05,         # Slower learning rate for higher accuracy
    num_leaves=63,              # More complex trees
    class_weight='balanced',    # Fixes the imbalanced classes (like rare falls)
    n_jobs=-1,                  # Use all CPU cores
    random_state=42,
    verbose=-1
)

# StratifiedGroupKFold ensures classes are balanced in every fold
cv = StratifiedGroupKFold(n_splits=5)

print("Running cross-validation...")
y_pred = cross_val_predict(clf, X, y, groups=groups, cv=cv, n_jobs=-1)

print("\n--- RESULTS ---")
print("Accuracy:", accuracy_score(y, y_pred))
print(classification_report(y, y_pred))

# Train the final model on ALL data and save it
print("Fitting final model on all data...")
clf.fit(X, y)
joblib.dump(clf, "behavior_model_yolo26.pkl")
print("Final model saved as behavior_model_yolo26.pkl")