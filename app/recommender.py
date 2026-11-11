"""
Recommendation logic for the Streamlit app.

This is the Milestone 4 notebook's recommender, packaged as a module. It loads the
models saved by scripts/export_artifacts.py and exposes one function:

    recommend_campaign(campaign: dict) -> dict

`campaign` uses the *display* labels shown in the app:
    {"category": "Film & Video", "subcategory": "Documentary", "state": "GA",
     "goal": 3500, "duration": 45, "levels": 8, "updates": 0}

The result follows the data contract described in app/README.md.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

CAT = ["category", "subcategory", "state"]
NUM_A = ["goal", "duration", "levels"]               # Plan A: no updates
NUM_B = ["goal", "duration", "levels", "updates"]    # Plan B: with updates

AT_RISK = 0.50      # predicted success chance below this -> AT RISK
HIGH_RISK = 0.30    # below this -> HIGH RISK
MIN_PEERS = 30      # fall back from subcategory peers to category peers below this size


class ArtifactsMissing(FileNotFoundError):
    """Raised when the exported model files are not found."""


@lru_cache(maxsize=1)
def load_artifacts():
    needed = [MODELS_DIR / "model_a.joblib", MODELS_DIR / "model_b.joblib", MODELS_DIR / "lookups.json"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise ArtifactsMissing(
            "Missing " + ", ".join(missing) + " in the models/ folder. "
            "Run `python scripts/export_artifacts.py` from the repository root first."
        )
    return {
        "model_a": joblib.load(MODELS_DIR / "model_a.joblib"),
        "model_b": joblib.load(MODELS_DIR / "model_b.joblib"),
        "lookups": json.loads((MODELS_DIR / "lookups.json").read_text()),
    }


def risk_level(p):
    if p < HIGH_RISK:
        return "HIGH RISK"
    if p < AT_RISK:
        return "AT RISK"
    return "likely to succeed"


def _candidate_changes(row, goal_floor, levels_cap):
    """List of (lever, description, {column: new_value}) for one campaign.
    Only levers a creator controls are changed; updates are never recommended as a lever."""
    out = []
    g = float(row["goal"])
    for f in [0.9, 0.8, 0.67, 0.5]:
        new = max(round(g * f, -1) if g >= 100 else round(g * f), goal_floor)
        if new < g:
            out.append(("goal", f"Lower the funding goal from {g:,.0f} to {new:,.0f} USD", {"goal": float(new)}))
    for d in [20, 30, 45, 60]:
        if abs(d - float(row["duration"])) >= 1:
            out.append(("duration", f"Set the campaign length to {d} days (currently {row['duration']:.0f})",
                        {"duration": float(d)}))
    for k in [2, 4, 6]:
        new = min(int(row["levels"]) + k, levels_cap)
        if new > row["levels"]:
            out.append(("levels", f"Offer {new} reward levels (currently {int(row['levels'])})", {"levels": new}))
    return out


def _score_plan(model, num, row, goal_floor, levels_cap, top_n=3, min_lift=0.005):
    cols = num + CAT
    base = pd.DataFrame([row])[cols].astype({c: float for c in num})
    base_p = float(model.predict_proba(base)[:, 1][0])

    cands = _candidate_changes(row, goal_floor, levels_cap)
    recs, combined_p = [], base_p
    if cands:
        rows = []
        for _, _, change in cands:
            r = base.copy()
            for k, v in change.items():
                r[k] = float(v)
            rows.append(r)
        probs = model.predict_proba(pd.concat(rows, ignore_index=True))[:, 1]
        table = pd.DataFrame({
            "lever": [c[0] for c in cands], "text": [c[1] for c in cands],
            "new_values": [c[2] for c in cands], "new_probability": probs,
        })
        table["lift"] = table["new_probability"] - base_p
        best = (table.sort_values("lift", ascending=False).drop_duplicates("lever")
                     .query("lift >= @min_lift").head(top_n))
        combo = base.copy()
        for nv in best["new_values"]:
            for k, v in nv.items():
                combo[k] = float(v)
        if len(best):
            combined_p = float(model.predict_proba(combo)[:, 1][0])
        recs = [{"lever": r.lever, "text": r.text,
                 "new_probability": float(r.new_probability), "lift": float(r.lift)}
                for r in best.itertuples()]

    return {"probability": base_p, "risk_level": risk_level(base_p),
            "recommendations": recs, "combined_probability": combined_p}


def _peer_benchmark(lookups, category, subcategory):
    peers = lookups["peers"]["subcategory"].get(subcategory)
    label = subcategory
    if peers is None or peers["n"] < MIN_PEERS:
        peers = lookups["peers"]["category"][category]
        label = category
    return {"label": label, "n": peers["n"], "goal": peers["goal"], "duration": peers["duration"],
            "levels": peers["levels"], "updates": peers["updates"]}


def recommend_campaign(campaign: dict) -> dict:
    art = load_artifacts()
    lk = art["lookups"]

    cat, sub = campaign["category"], campaign["subcategory"]
    if cat not in lk["categories"]:
        raise ValueError(f"Unknown category: {cat!r}")
    if sub not in lk["categories"][cat]["subcategories"]:
        raise ValueError(f"Unknown subcategory {sub!r} for category {cat!r}")

    row = {
        "category": lk["categories"][cat]["value"],              # original label the model was trained on
        "subcategory": lk["categories"][cat]["subcategories"][sub],
        "state": campaign["state"],
        "goal": float(campaign["goal"]),
        "duration": float(campaign["duration"]),
        "levels": int(campaign["levels"]),
        "updates": int(campaign["updates"]),
    }
    goal_floor, levels_cap = lk["goal_floor"], lk["levels_cap"]

    plan_a = _score_plan(art["model_a"], NUM_A, row, goal_floor, levels_cap)
    plan_b = _score_plan(art["model_b"], NUM_B, row, goal_floor, levels_cap)
    peers = _peer_benchmark(lk, cat, sub)

    note = (f"Successful campaigns like this one typically post about {peers['updates']:.0f} updates; "
            f"this campaign has {row['updates']}. This is an association, not a guaranteed fix.")

    return {"plan_a": plan_a, "plan_b": plan_b, "peer_benchmark": peers, "engagement_note": note}
