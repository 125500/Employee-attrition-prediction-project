"""Train and compare models, then save the best full pipeline.

Run from the project root:  python -m src.train
"""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config


def load_data():
    df = pd.read_csv(config.DATA_PATH)
    df = df.drop(columns=config.DROP_COLS, errors="ignore")
    y = (df[config.TARGET] == "Yes").astype(int)
    X = df.drop(columns=[config.TARGET])
    return X, y


def build_preprocessor(X):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def get_models(pos_weight):
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=config.RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=config.RANDOM_STATE),
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    results, pipelines = [], {}
    for name, model in get_models(pos_weight).items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results.append({
            "model": name,
            "cv_auc": round(cv_auc, 3),
            "test_auc": round(roc_auc_score(y_test, proba), 3),
            "precision": round(precision_score(y_test, pred), 3),
            "recall": round(recall_score(y_test, pred), 3),
            "f1": round(f1_score(y_test, pred), 3),
        })
        pipelines[name] = pipe

    table = pd.DataFrame(results).sort_values("test_auc", ascending=False)
    print(table.to_string(index=False))

    best_name = table.iloc[0]["model"]
    print(f"\nBest model: {best_name}")
    best = pipelines[best_name]
    print(classification_report(y_test, best.predict(X_test), target_names=["Stays", "Leaves"]))

    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, config.MODEL_PATH)
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    print(f"Saved pipeline to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
