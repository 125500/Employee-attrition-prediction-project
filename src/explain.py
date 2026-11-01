"""Explain the trained model with SHAP.

Run from the project root:  python -m src.explain
Saves plots to reports/.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.model_selection import train_test_split

from src import config
from src.train import load_data


def to_dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def main():
    pipe = joblib.load(config.MODEL_PATH)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    X_test_t = to_dense(prep.transform(X_test))
    background = shap.sample(to_dense(prep.transform(X_train)), 100, random_state=0)

    explainer = shap.Explainer(model, background, feature_names=names)
    exp = explainer(X_test_t)
    if exp.values.ndim == 3:          # some models return one set of values per class
        exp = exp[:, :, 1]            # keep the "leaves" class

    # 1) Global: which features matter most overall
    shap.plots.bar(exp, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "shap_bar.png", dpi=150)
    plt.close()

    # 2) Global: direction of effect (high/low feature values push risk up or down)
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "shap_beeswarm.png", dpi=150)
    plt.close()

    # 3) Local: why the highest-risk employee in the test set is flagged
    proba = pipe.predict_proba(X_test)[:, 1]
    idx = int(np.argmax(proba))
    shap.plots.waterfall(exp[idx], max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "shap_waterfall_top_risk.png", dpi=150)
    plt.close()

    print(f"Highest-risk test employee: predicted risk {proba[idx]:.0%}")
    print("Saved 3 plots to", config.REPORTS_DIR)


if __name__ == "__main__":
    main()