from sklearn.linear_model import LogisticRegression

def make_model():
    return LogisticRegression(max_iter=1000)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


def make_model():
    # Colonnes numériques / catégorielles
    numeric_features = ["age", "dependents", "income", "credit_amount", "loan_duration", "employment_years", "sex"]
    categorical_features = ["education_level", "marital_status", "housing_status", "nationality"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=2000)

    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf),
    ])