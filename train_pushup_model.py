import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------- LOAD DATA ----------------
df = pd.read_csv("pushup_data.csv")

print("Dataset shape:", df.shape)
print(df.head())

# ---------------- SPLIT ----------------
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- TRAIN ----------------
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# ---------------- SAVE ----------------
joblib.dump({
    "model": model,
    "features": list(X.columns)
}, "pushup_system.pkl")

print("\nModel saved as pushup_system.pkl ✅")