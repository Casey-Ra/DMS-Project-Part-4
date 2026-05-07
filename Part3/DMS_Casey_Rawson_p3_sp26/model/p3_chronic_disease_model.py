import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Sample chronic disease risk dataset.
# This is not a real medical prediction model. It is a class project model

data = [
    # age, exercise, unhealthy_eating, smoking_drinking, heredity, living_standard, high_risk
    [25, 4, 1, 0, 0, 4, 0],
    [30, 3, 1, 0, 0, 4, 0],
    [35, 3, 2, 1, 0, 3, 0],
    [40, 2, 2, 1, 1, 3, 1],
    [45, 2, 3, 1, 1, 3, 1],
    [50, 1, 3, 1, 1, 2, 1],
    [55, 1, 4, 1, 1, 2, 1],
    [60, 1, 4, 1, 1, 1, 1],
    [28, 4, 1, 0, 0, 4, 0],
    [33, 3, 2, 0, 0, 3, 0],
    [38, 2, 2, 0, 1, 3, 0],
    [42, 2, 3, 1, 0, 3, 1],
    [48, 1, 3, 1, 1, 2, 1],
    [52, 1, 4, 0, 1, 2, 1],
    [58, 1, 4, 1, 0, 2, 1],
    [63, 1, 4, 1, 1, 1, 1],
    [27, 4, 2, 0, 0, 4, 0],
    [36, 3, 2, 1, 0, 3, 0],
    [44, 2, 3, 0, 1, 3, 1],
    [57, 1, 3, 1, 1, 2, 1],
]

columns = [
    "age",
    "exercise_level",
    "unhealthy_eating_level",
    "smoking_drinking",
    "heredity",
    "living_standard",
    "high_risk",
]

df = pd.DataFrame(data, columns=columns)

X = df.drop("high_risk", axis=1)
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Decision Tree Chronic Disease Risk Model")
print("----------------------------------------")
print(f"Accuracy: {accuracy:.2f}")
print()

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print()

print("Classification Report:")
print(classification_report(y_test, predictions))
print()

print("Decision Rules:")
print(export_text(model, feature_names=list(X.columns)))

joblib.dump(model, "chronic_disease_risk_model.joblib")
df.to_csv("chronic_disease_sample_data.csv", index=False)

print()
print("Saved model as chronic_disease_risk_model.joblib")
print("Saved sample data as chronic_disease_sample_data.csv")