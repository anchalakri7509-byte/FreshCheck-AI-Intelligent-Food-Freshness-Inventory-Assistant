
"""
FreshCheck AI - Spoilage Risk Prediction

Educational ML demonstration using synthetic proxy labels.
These are NOT real food spoilage observations.
"""

from pathlib import Path
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).parent
OUT = ROOT / "models" / "spoilage_risk.joblib"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Generate synthetic demonstration dataset
rng = np.random.default_rng(42)
n = 3000

# Fruit encoding: apple=0, banana=1, orange=2
fruit = rng.integers(0, 3, n)
days = rng.integers(0, 15, n)
temp = rng.uniform(2, 30, n)
humidity = rng.uniform(30, 95, n)

# Synthetic proxy risk probability
latent = (
    0.10
    + days * 0.045
    + np.maximum(temp - 5, 0) * 0.018
    + np.maximum(humidity - 70, 0) * 0.003
    + (fruit == 1) * 0.08
)

prob = np.clip(latent, 0.02, 0.98)
y = (rng.random(n) < prob).astype(int)

# Feature matrix
X = np.column_stack((fruit, days, temp, humidity))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest
print("Training FreshCheck AI risk model...")

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=1
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["lower proxy risk", "higher proxy risk"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
joblib.dump(model, OUT)

print("\nSUCCESS: Risk model trained!")
print("Saved at:", OUT)

print("\nFeature order:")
print("1. Fruit code (0=apple, 1=banana, 2=orange)")
print("2. Days since purchase")
print("3. Temperature in Celsius")
print("4. Humidity percentage")
