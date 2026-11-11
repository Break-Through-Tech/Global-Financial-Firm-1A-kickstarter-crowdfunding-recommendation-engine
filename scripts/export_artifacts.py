"""
Export the model artifacts used by the Streamlit app.

Trains the two final XGBoost models from the Milestone 4 notebook
(Plan A = without `updates`, Plan B = with `updates`) using the same data,
split (80/20, stratified, random_state=42) and settings, then saves:

    models/model_a.joblib   Plan A pipeline (preprocessing + XGBoost)
    models/model_b.joblib   Plan B pipeline (preprocessing + XGBoost)
    models/lookups.json     dropdown options, peer medians, caps, driver importance, test metrics

Run from the repository root:

    python scripts/export_artifacts.py

Requires data/kickstarter_us_filtered.csv (see the README for how to build it).
"""

import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path("data/kickstarter_us_filtered.csv")
OUT_DIR = Path("models")

CAT = ["category", "subcategory", "state"]
NUM_A = ["goal", "duration", "levels"]               # Plan A: no updates
NUM_B = ["goal", "duration", "levels", "updates"]    # Plan B: with updates


def make_model(num_features):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([
        ("preprocessor", pre),
        ("model", XGBClassifier(random_state=42, eval_metric="logloss",
                                learning_rate=0.1, max_depth=3, n_estimators=400)),
    ])


def driver_importance(model, num, test, n=2000):
    """Mean absolute TreeSHAP value per original feature (encoded columns are summed per feature)."""
    pre, clf = model.named_steps["preprocessor"], model.named_steps["model"]
    sample = test[num + CAT].sample(min(n, len(test)), random_state=0)
    Xt = pre.transform(sample)  # keep the sparse form the pipeline gives the model
    names = pre.get_feature_names_out()
    contribs_all = clf.get_booster().predict(xgboost.DMatrix(Xt), pred_contribs=True)

    # Sanity check: contributions (plus bias) must add up to the model's own log-odds
    p = model.predict_proba(sample)[:, 1]
    gap = np.abs(contribs_all.sum(axis=1) - np.log(p / (1 - p))).max()
    assert gap < 1e-3, f"SHAP values do not match the model's predictions (gap {gap:.3f})"
    contribs = contribs_all[:, :-1]

    def group_of(name):
        if name.startswith("num__"):
            return name[len("num__"):]
        if name.startswith("cat__subcategory_"):
            return "subcategory"
        if name.startswith("cat__category_"):
            return "category"
        return "state"

    groups = np.array([group_of(nm) for nm in names])
    rows = []
    for g in pd.unique(groups):
        g_shap = contribs[:, groups == g].sum(axis=1)
        direction = ""
        if g in num:
            rho = pd.Series(sample[g].values).corr(pd.Series(g_shap), method="spearman")
            direction = ("higher value -> higher chance" if rho > 0.1
                         else "higher value -> lower chance" if rho < -0.1 else "mixed")
        rows.append({"driver": g, "mean_abs_shap": round(float(np.abs(g_shap).mean()), 3),
                     "direction": direction})
    return sorted(rows, key=lambda r: -r["mean_abs_shap"])


def peer_table(train_success, display_col):
    out = {}
    for name, grp in train_success.groupby(display_col):
        out[name] = {
            "n": int(len(grp)),
            "goal": float(grp["goal"].median()),
            "duration": float(grp["duration"].median()),
            "levels": float(grp["levels"].median()),
            "updates": float(grp["updates"].median()),
        }
    return out


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Could not find {DATA_PATH}. Build it first (see the README, 'Build the Modeling Dataset'), "
            "and run this script from the repository root."
        )

    df = pd.read_csv(DATA_PATH)
    df["success"] = (df["status"] == "successful").astype(int)
    df["state"] = df["location"].str.rsplit(",", n=1).str[-1].str.strip()

    # Some labels appear under two spellings (for example 'Film & Video' and 'Film &amp; Video').
    # The models keep the original labels; the app shows the cleaned label and maps it back.
    df["category_display"] = df["category"].map(html.unescape)
    df["subcategory_display"] = df["subcategory"].map(html.unescape)

    train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=42, stratify=df["success"])
    train, test = df.loc[train_idx], df.loc[test_idx]

    model_a = make_model(NUM_A).fit(train[NUM_A + CAT], train["success"])
    model_b = make_model(NUM_B).fit(train[NUM_B + CAT], train["success"])

    metrics = {}
    for key, model, num in [("plan_a", model_a, NUM_A), ("plan_b", model_b, NUM_B)]:
        p = model.predict_proba(test[num + CAT])[:, 1]
        metrics[key] = {"accuracy": round(float(accuracy_score(test["success"], p > 0.5)), 3),
                        "roc_auc": round(float(roc_auc_score(test["success"], p)), 3)}
        print(f"{key}: accuracy={metrics[key]['accuracy']}  ROC-AUC={metrics[key]['roc_auc']}")

    def canonical(display_col, raw_col):
        counts = df.groupby([display_col, raw_col]).size().reset_index(name="n")
        counts = counts.sort_values("n", ascending=False).drop_duplicates(display_col)
        return dict(zip(counts[display_col], counts[raw_col]))

    canon_cat = canonical("category_display", "category")
    canon_sub = canonical("subcategory_display", "subcategory")

    categories = {}
    for disp in sorted(canon_cat):
        subs = sorted(df.loc[df["category_display"] == disp, "subcategory_display"].unique())
        categories[disp] = {"value": canon_cat[disp],
                            "subcategories": {s: canon_sub[s] for s in subs}}

    train_success = train[train["success"] == 1]
    lookups = {
        "categories": categories,
        "states": sorted(df["state"].dropna().unique().tolist()),
        "peers": {
            "subcategory": peer_table(train_success, "subcategory_display"),
            "category": peer_table(train_success, "category_display"),
        },
        "goal_floor": float(train["goal"].quantile(0.01)),
        "levels_cap": int(train["levels"].quantile(0.90)),
        "driver_importance": {
            "plan_a": driver_importance(model_a, NUM_A, test),
            "plan_b": driver_importance(model_b, NUM_B, test),
        },
        "metrics": metrics,
        "training_rows": int(len(train)),
        "xgboost_version": xgboost.__version__,
    }

    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model_a, OUT_DIR / "model_a.joblib", compress=3)
    joblib.dump(model_b, OUT_DIR / "model_b.joblib", compress=3)
    (OUT_DIR / "lookups.json").write_text(json.dumps(lookups, indent=1))
    print(f"Saved model_a.joblib, model_b.joblib and lookups.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
