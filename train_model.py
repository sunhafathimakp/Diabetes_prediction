import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report

# -----------------------
# Load Dataset
# -----------------------

data = pd.read_csv("data/diabetes.csv")

print("First 5 Rows")
print(data.head())

# -----------------------
# Features and Target
# -----------------------

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# -----------------------
# Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------
# Models
# -----------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

best_model = None
best_accuracy = 0

print("\nModel Results\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print("=" * 50)
    print(name)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print(classification_report(y_test, prediction))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# -----------------------
# Save Best Model
# -----------------------

os.makedirs("model", exist_ok=True)

joblib.dump(best_model, "model/diabetes_model.pkl")

print("=" * 50)
print("Best Accuracy :", round(best_accuracy * 100, 2), "%")
print("Best Model Saved Successfully!")