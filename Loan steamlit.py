import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = pd.DataFrame({
    "Gender": [
        "Male", "Female", "Male", "Male", "Female",
        "Male", "Female", "Male", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male",
        "Male", "Female", "Male", "Male", "Female"
    ],

    "Married": [
        "Yes", "No", "Yes", "Yes", "No",
        "Yes", "No", "Yes", "Yes", "No",
        "Yes", "Yes", "No", "No", "Yes",
        "No", "Yes", "Yes", "No", "Yes"
    ],

    "Education": [
        "Graduate", "Graduate", "Not Graduate", "Graduate",
        "Graduate", "Graduate", "Not Graduate", "Graduate",
        "Graduate", "Graduate", "Not Graduate", "Graduate",
        "Graduate", "Not Graduate", "Graduate", "Graduate",
        "Graduate", "Not Graduate", "Graduate", "Graduate"
    ],

    "Self_Employed": [
        "No", "No", "No", "Yes", "No",
        "No", "Yes", "No", "No", "No",
        "No", "Yes", "No", "No", "Yes",
        "No", "No", "Yes", "No", "No"
    ],

    "ApplicantIncome": [
        5000, 3500, 2800, 6000, 4200,
        7000, 2500, 4500, 8000, 3200,
        3000, 6500, 4000, 2600, 7500,
        3800, 5200, 2900, 4800, 6100
    ],

    "CoapplicantIncome": [
        1500, 0, 1000, 2000, 0,
        2500, 1200, 1000, 3000, 0,
        800, 1800, 0, 500, 2200,
        0, 1200, 900, 0, 1500
    ],

    "LoanAmount": [
        150, 100, 90, 200, 110,
        250, 80, 140, 300, 100,
        90, 180, 120, 75, 220,
        130, 160, 85, 145, 190
    ],

    "Loan_Amount_Term": [
        360, 360, 360, 360, 360,
        360, 360, 360, 360, 360,
        360, 360, 360, 360, 360,
        360, 360, 360, 360, 360
    ],

    "Credit_History": [
        1, 1, 0, 1, 1,
        1, 0, 1, 1, 1,
        0, 1, 1, 0, 1,
        1, 1, 0, 1, 1
    ],

    "Property_Area": [
        "Urban", "Rural", "Rural", "Urban", "Semiurban",
        "Urban", "Rural", "Semiurban", "Urban", "Rural",
        "Rural", "Semiurban", "Urban", "Rural", "Semiurban",
        "Urban", "Semiurban", "Rural", "Urban", "Semiurban"
    ],

    "Loan_Status": [
        1, 1, 0, 1, 1,
        1, 0, 1, 1, 1,
        0, 1, 1, 0, 1,
        1, 1, 0, 1, 1
    ]
})

# Separate features and target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Categorical and numerical columns
categorical_columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical_columns
        ),

        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        )
    ]
)

# ML model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save model
with open("loan_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nloan_model.pkl created successfully!")
