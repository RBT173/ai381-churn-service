"""Train the churn model and save it as models/model.joblib."""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "internet_service"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # churn_v2 renamed monthly_charges -> monthly_fee; use one name everywhere
    return df.rename(columns={"monthly_fee": "monthly_charges"})


def build_model() -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="churn_v2.csv")
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args()

    df = load_data(args.data)
    X, y = df[NUMERIC + CATEGORICAL], df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print("F1 :", round(f1_score(y_test, proba > 0.5), 3))
    print("AUC:", round(roc_auc_score(y_test, proba), 3))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
