"""
Kickstarter Campaign Advisor - Streamlit front end for the Milestone 4 recommendation prototype.

Run from the repository root:

    streamlit run app/streamlit_app.py

Requires the model files in models/ (create them with `python scripts/export_artifacts.py`).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from recommender import AT_RISK, HIGH_RISK, ArtifactsMissing, load_artifacts, recommend_campaign

st.set_page_config(page_title="Kickstarter Campaign Advisor", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Loading models...")
def get_artifacts():
    return load_artifacts()


try:
    artifacts = get_artifacts()
except ArtifactsMissing as err:
    st.error(str(err))
    st.stop()

lookups = artifacts["lookups"]
categories = lookups["categories"]

# ---------------------------------------------------------------- example campaigns
PRESETS = {
    "Documentary: long campaign, no updates": dict(category="Film & Video", subcategory="Documentary", state="GA",
                                                    goal=3500, duration=45, levels=8, updates=0),
    "Theater: high goal, very long campaign": dict(category="Theater", subcategory="Theater", state="WI",
                                                    goal=7500, duration=61, levels=8, updates=1),
    "Comics: only one reward level": dict(category="Comics", subcategory="Comics", state="GA",
                                          goal=300, duration=30, levels=1, updates=0),
    "Country & Folk: very high goal": dict(category="Music", subcategory="Country & Folk", state="TN",
                                            goal=22000, duration=60, levels=9, updates=4),
    "Dance: well-prepared campaign": dict(category="Dance", subcategory="Dance", state="NY",
                                          goal=2000, duration=30, levels=9, updates=6),
}

DEFAULTS = PRESETS["Documentary: long campaign, no updates"]
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault(f"sub_{DEFAULTS['category']}", DEFAULTS["subcategory"])


def load_preset(name):
    p = PRESETS[name]
    for key in ("category", "state", "goal", "duration", "levels", "updates"):
        st.session_state[key] = p[key]
    st.session_state[f"sub_{p['category']}"] = p["subcategory"]


# ---------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("Your campaign")
    st.caption("Describe the campaign, or load an example below.")

    category = st.selectbox("Category", sorted(categories), key="category")
    subcategory = st.selectbox("Subcategory", sorted(categories[category]["subcategories"]),
                               key=f"sub_{category}")
    state = st.selectbox("State", lookups["states"], key="state")
    goal = st.number_input("Funding goal (USD)", min_value=100, max_value=1_000_000, step=100, key="goal")
    duration = st.slider("Campaign length (days)", min_value=1, max_value=90, key="duration")
    levels = st.number_input("Number of reward levels", min_value=0, max_value=60, step=1, key="levels")
    updates = st.number_input("Number of campaign updates", min_value=0, max_value=100, step=1, key="updates",
                              help="Updates posted on the campaign page. Used by Plan B only.")

    st.divider()
    st.subheader("Try an example")
    for name in PRESETS:
        st.button(name, on_click=load_preset, args=(name,), width="stretch")

result = recommend_campaign({
    "category": category, "subcategory": subcategory, "state": state,
    "goal": goal, "duration": duration, "levels": levels, "updates": updates,
})

# ---------------------------------------------------------------- header
st.title("Kickstarter Campaign Advisor")
st.markdown(
    "Estimates a campaign's chance of success from its setup, flags campaigns that look **at risk**, "
    "and suggests changes to the funding goal, campaign length and reward levels that the model "
    "associates with a better chance of success."
)
st.info(
    "**These are model estimates, not guarantees.** The suggestions come from patterns in past U.S. Kickstarter "
    "campaigns. They show what is *associated* with success, not what would *cause* it."
)

RISK_STYLE = {"HIGH RISK": st.error, "AT RISK": st.warning, "likely to succeed": st.success}


def render_plan(plan, heading, caption, key):
    st.subheader(heading)
    st.caption(caption)
    prob = plan["probability"]
    st.metric("Predicted chance of success", f"{prob:.0%}")
    st.progress(min(max(prob, 0.0), 1.0))
    level = plan["risk_level"]
    message = {"HIGH RISK": f"HIGH RISK: below {HIGH_RISK:.0%} predicted chance of success",
               "AT RISK": f"AT RISK: below {AT_RISK:.0%} predicted chance of success",
               "likely to succeed": f"Likely to succeed: {AT_RISK:.0%} or higher predicted chance"}[level]
    RISK_STYLE[level](message)

    recs = plan["recommendations"]
    if not recs:
        st.write("No change to the goal, length or reward levels raised the predicted chance meaningfully.")
        return
    st.markdown("**Top suggestions**" if level != "likely to succeed" else "**Optional improvements**")
    for i, r in enumerate(recs, 1):
        st.markdown(f"{i}. {r['text']}  \n&nbsp;&nbsp;&nbsp;&nbsp;→ predicted chance **{r['new_probability']:.0%}** "
                    f"({r['lift'] * 100:+.0f} points)")
    chart = pd.DataFrame({"Predicted lift (percentage points)": [r["lift"] * 100 for r in recs]},
                         index=[f"{i}. {r['lever']}" for i, r in enumerate(recs, 1)])
    st.bar_chart(chart, horizontal=True, height=40 + 45 * len(recs))
    st.caption(f"All suggestions applied together: about {plan['combined_probability']:.0%} "
               "(an optimistic upper bound, since the effects may not simply add up).")


col_a, col_b = st.columns(2, gap="large")
with col_a:
    render_plan(result["plan_a"], "Plan A: before launch",
                "Model without the updates feature. Use this when planning a campaign.", "a")
with col_b:
    render_plan(result["plan_b"], "Plan B: with updates",
                "Model that also knows the campaign's update count.", "b")

st.divider()

# ---------------------------------------------------------------- peers and engagement
peers = result["peer_benchmark"]
st.subheader("How your campaign compares with successful ones")
compare = pd.DataFrame({
    "Your campaign": [f"{goal:,.0f}", f"{duration:.0f}", f"{levels}", f"{updates}"],
    f"Typical successful '{peers['label']}' campaign (n={peers['n']:,})": [
        f"{peers['goal']:,.0f}", f"{peers['duration']:.0f}", f"{peers['levels']:.0f}", f"{peers['updates']:.0f}"],
}, index=["Funding goal (USD)", "Length (days)", "Reward levels", "Updates"])
st.dataframe(compare, width="stretch")
st.warning("**Engagement note (association, not a promise):** " + result["engagement_note"])

# ---------------------------------------------------------------- explanations
with st.expander("How to read these results"):
    st.markdown(
        f"""
- **Predicted chance of success** is the model's estimate. On held-out data, predicted chances matched
  actual success rates closely (for example, campaigns given about 30% did succeed about 30% of the time).
- **At risk** means a predicted chance below {AT_RISK:.0%}; **high risk** means below {HIGH_RISK:.0%}.
- **Plan A** uses only information a creator has when planning a campaign. **Plan B** also uses the number
  of updates, which makes it more accurate but means it depends on information that may only exist after launch.
- **Suggestions** are found by changing one thing at a time (the goal, the length, or the number of reward levels),
  re-scoring the campaign, and keeping the change with the biggest predicted improvement for each of the three.
- **Updates are not recommended as a fix.** Campaigns that are going well tend to post more updates *because*
  they are going well, so the update count is used as context and shown as an engagement note only.
        """
    )

with st.expander("What does the model rely on?"):
    imp_a = pd.DataFrame(lookups["driver_importance"]["plan_a"]).set_index("driver")["mean_abs_shap"]
    imp_b = pd.DataFrame(lookups["driver_importance"]["plan_b"]).set_index("driver")["mean_abs_shap"]
    imp = pd.concat([imp_a.rename("Plan A"), imp_b.rename("Plan B")], axis=1).fillna(0)
    imp = imp.sort_values("Plan B", ascending=False)
    st.bar_chart(imp, horizontal=True, height=320)
    st.caption("Average influence of each feature on predictions (mean absolute SHAP value, higher means more "
               "influence). Plan A has no updates feature.")

with st.expander("Limitations"):
    m = lookups["metrics"]
    st.markdown(
        f"""
- **Association, not causation.** A lower goal or more reward levels is associated with a higher predicted chance
  of success; this tool cannot prove it would change your campaign's outcome.
- **Estimated accuracy.** On a held-out test set, Plan A reached {m['plan_a']['accuracy']:.0%} accuracy
  (ROC-AUC {m['plan_a']['roc_auc']:.2f}) and Plan B {m['plan_b']['accuracy']:.0%} (ROC-AUC {m['plan_b']['roc_auc']:.2f}).
  Both models miss some failing campaigns, especially Plan A.
- **Combined estimates are optimistic.** Suggestions are scored one at a time; applying several together may help less.
- **Fixed attributes.** Category, subcategory and state are not changed by the suggestions.
- **Historical data.** The models were trained on past U.S. campaigns and may not reflect changes in how Kickstarter is used.
- **Not professional advice.** Use these estimates as one input to your own planning.
        """
    )
