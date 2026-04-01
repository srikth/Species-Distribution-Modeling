import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

print("Loading occurrence data...")
df = pd.read_csv("../data/occurrences.csv")

df = df[['decimalLatitude', 'decimalLongitude']].dropna()
df['presence'] = 1

print(f"Presence points: {len(df)}")

print("Generating pseudo-absence data...")
n = len(df)

pseudo_absences = pd.DataFrame({
    'decimalLatitude': np.random.uniform(df.decimalLatitude.min(), df.decimalLatitude.max(), n),
    'decimalLongitude': np.random.uniform(df.decimalLongitude.min(), df.decimalLongitude.max(), n),
    'presence': 0
})

data = pd.concat([df, pseudo_absences], ignore_index=True)

print("Adding environmental variables...")
np.random.seed(42)

data['temperature'] = np.random.uniform(20, 35, len(data))
data['rainfall'] = np.random.uniform(500, 3000, len(data))

X = data[['temperature', 'rainfall']]
y = data['presence']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred)
print(f"AUC Score: {auc:.3f}")

fpr, tpr, _ = roc_curve(y_test, y_pred)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("../outputs/images/roc_curve.png")
print("ROC curve saved!")

print("\nFeature Importance:")
for name, val in zip(X.columns, model.feature_importances_):
    print(f"{name}: {val:.3f}")

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted_Probability": y_pred
})

results.to_csv("../outputs/model_results.csv", index=False)

print("Results saved!")
print("Done")
