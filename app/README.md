# Kickstarter Campaign Advisor (Streamlit app)

Front end for the Milestone 4 recommendation prototype. Enter a campaign's category, subcategory,
state, goal, length, reward levels and updates. The app shows the predicted chance of success (Plan A without
`updates`, Plan B with `updates`), an at-risk flag, the top 3 suggested changes with predicted lift, a peer
comparison and an engagement note.

## Run it (from the repository root)

```
python3 -m pip install -r requirements.txt
python3 scripts/export_artifacts.py
streamlit run app/streamlit_app.py
```

`export_artifacts.py` needs `data/kickstarter_us_filtered.csv` (see the main README) and writes
`models/model_a.joblib`, `models/model_b.joblib` and `models/lookups.json`. The `models/` folder is generated, so
do not commit it; re-run the script after changing the data or package versions.

## Files

- `app/streamlit_app.py` - the page (inputs, results, explanations)
- `app/recommender.py` - `recommend_campaign(campaign: dict) -> dict`, no Streamlit code
- `scripts/export_artifacts.py` - trains the two final XGBoost models and saves lookups

## Data contract (`recommend_campaign`)

Input: `category, subcategory, state, goal, duration, levels, updates` (display labels).
Output: `plan_a` and `plan_b` (each `probability`, `risk_level`, `recommendations[{lever,text,new_probability,lift}]`,
`combined_probability`), `peer_benchmark{label,n,goal,duration,levels,updates}`, `engagement_note`.

## Caveats

Suggestions are associations, not causal effects. `updates` is never recommended as a lever. Combined estimates are
an optimistic upper bound.
