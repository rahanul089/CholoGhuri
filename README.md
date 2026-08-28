# 🧭 CholoGhuri — Bangladesh Tourism AI Recommender

A Streamlit app that turns a 420-review traveler survey into a live, explainable
recommendation engine for destinations across Bangladesh.

- **Predicts** whether someone with your travel preferences would recommend the trip
  (hybrid soft-voting ensemble: Random Forest + Extra Trees + Gradient Boosting + XGBoost).
- **Explains** each prediction with SHAP (falls back to feature importance if SHAP fails to load).
- **Recommends** the top 5 matching destinations based on category, persona, budget and duration.
- **Visualizes** the underlying dataset (ratings, personas, seasons, correlations) with Plotly.

Built from an original research notebook (`tourfnal.ipynb`) that trained and evaluated the
same ensemble offline; this app retrains a lightweight version of that pipeline on startup
and caches it, so no pre-trained model files are needed.

## Project structure

```
bd-tourism-ai/
├── app.py                  # Streamlit UI (all pages)
├── src/
│   ├── data_processing.py  # loading, cleaning, feature engineering
│   ├── model.py            # ensemble training, prediction, SHAP explanation
│   ├── destinations.py     # destination database + recommendation scoring
│   └── styles.py           # custom CSS (green dashboard theme)
├── data/
│   └── ml.csv              # traveler review dataset
├── .streamlit/config.toml  # theme
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. First load takes a few seconds while the
ensemble trains — after that it's cached for the session.

## Deploy to Streamlit Community Cloud (free)

1. **Push this folder to GitHub** (see commands below).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"** → pick your repo → set:
   - **Main file path:** `app.py`
   - **Branch:** `main`
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically and gives
   you a public URL like `https://your-app-name.streamlit.app`.
5. Whenever you `git push` new commits, the app redeploys automatically.

## Push to GitHub

```bash
cd bd-tourism-ai
git init
git add .
git commit -m "Initial commit: CholoGhuri - Bangladesh Tourism AI Recommender"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

(Create the empty repo on GitHub first — no README/license, since this folder already has one.)

## Notes

- The dataset (`data/ml.csv`) is a small sample/synthetic survey used for demonstration —
  destination names, hotels and reviews are illustrative, not verified travel advice.
- `xgboost` and `shap` are optional at runtime: if either fails to install on a very
  constrained host, the app still works (falls back to 3-model ensemble / plain feature
  importances), but Streamlit Cloud installs both without issues.
