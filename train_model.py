import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
df = pd.read_csv("Titanic-Dataset.csv")
print("Dataset shape:", df.shape)
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]


numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])
categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])
preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),
    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])
ml_models = {

    "Logistic Regression": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(max_iter=1000)
        )
    ]),

    "Decision Tree": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            DecisionTreeClassifier(
                random_state=42,
                max_depth=5
            )
        )
    ]),

    "Random Forest": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42
            )
        )
    ]),

    "Gradient Boosting": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            GradientBoostingClassifier(
                random_state=42
            )
        )
    ]),

    "XGBoost": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
        )
    ])
}
ml_results = []
for name, model in ml_models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    result = {
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "Precision": precision_score(
            y_test,
            y_pred
        ),
        "Recall": recall_score(
            y_test,
            y_pred
        ),
        "F1 Score": f1_score(
            y_test,
            y_pred
        ),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_prob
        )
    }

    ml_results.append(result)
results_df = pd.DataFrame(
    ml_results
)
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)
print("\nModel Comparison:")
print(results_df.to_string(index=False))
best_model_name = results_df.iloc[0]["Model"]
print(
    "\nBest ML model based on F1 Score:",
    best_model_name
)
best_ml_model = ml_models[
    best_model_name
]
import os
os.makedirs(
    "models",
    exist_ok=True
)
model_path = "models/titanic_best_model.pkl"
joblib.dump(
    best_ml_model,
    model_path
)

print(
    "\nBest model saved successfully:"
)
print(model_path)