"""Show the precision/recall trade-off at different thresholds.

Run from the project root:  python -m src.sweep
"""
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src import config
from src.train import load_data


def stats(y_true, proba, thr):
    pred = proba >= thr
    pos = int((y_true == 1).sum())
    flagged = int(pred.sum())
    caught = int((pred & (y_true == 1)).sum())
    recall = caught / pos
    precision = caught / flagged if flagged else 0.0
    return recall, precision, flagged, caught, pos


def main():
    pipe = joblib.load(config.MODEL_PATH)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    proba_test = pipe.predict_proba(X_test)[:, 1]
    ytr, yte = y_train.values, y_test.values

    print(f"{'thr':>5} | {'CV recall':>9} {'CV prec':>8} | {'test recall':>11} {'test prec':>9} {'flagged':>8} {'caught':>9}")
    for thr in np.arange(0.20, 0.65, 0.05):
        cr, cp, _, _, _ = stats(ytr, oof, thr)
        tr, tp_, fl, ca, pos = stats(yte, proba_test, thr)
        print(f"{thr:>5.2f} | {cr:>9.2f} {cp:>8.2f} | {tr:>11.2f} {tp_:>9.2f} {fl:>8d} {ca:>4d}/{pos}")


if __name__ == "__main__":
    main()
    