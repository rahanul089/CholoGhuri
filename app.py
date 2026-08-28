import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_processing import build_single_row, get_feature_columns, load_and_engineer
from src.destinations import build_destinations_db, recommend
from src.model import HAS_SHAP, explain_prediction, predict, train_ensemble
from src.styles import CUSTOM_CSS

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ml.csv")

st.set_page_config(
    page_title="CholoGhuri | Bangladesh Tourism AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- caching
@st.cache_data(show_spinner=False)
def get_data():
    return load_and_engineer(DATA_PATH)


@st.cache_resource(show_spinner=True)
def get_model_bundle(df_hash):
    feature_cols, cat_features, num_features = get_feature_columns()
    return train_ensemble(df, feature_cols, cat_features, num_features)


@st.cache_data(show_spinner=False)
def get_destinations(df_hash):
    return build_destinations_db(df)


df = get_data()
feature_cols, cat_features, num_features = get_feature_columns()
model_bundle = get_model_bundle(len(df))
destinations_db = get_destinations(len(df))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.2rem;">
            <div style="background:white;border-radius:10px;width:42px;height:42px;
                        display:flex;align-items:center;justify-content:center;font-size:1.4rem;">🧭</div>
            <div>
                <div style="font-weight:800;font-size:1.05rem;line-height:1.1;">CholoGhuri</div>
                <div style="font-size:0.72rem;opacity:0.75;">Bangladesh Tourism AI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🎯 Get My Recommendation", "📊 Insights Dashboard", "🤖 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Dataset: {len(df)} traveler reviews")
    st.caption(f"Destinations tracked: {sum(len(v) for v in destinations_db.values())}")
    st.caption("⚠️ Educational project — not official tourism advice.")


def hero(title, subtitle, badge="Model Live"):
    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="hero-badge"><span class="dot"></span>{badge}</span>
            <div class="hero-title">{title}</div>
            <div class="hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- OVERVIEW
if page == "🏠 Overview":
    hero(
        "CholoGhuri — Explore Bangladesh, Backed by Data 🇧🇩",
        "An AI-powered recommender trained on real traveler reviews — predicts whether you'll "
        "love a trip and surfaces the destinations that best match your travel persona.",
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Reviews analyzed", "Destinations", "Personas modeled", "Best model accuracy"],
        [
            len(df),
            sum(len(v) for v in destinations_db.values()),
            df["persona"].nunique(),
            f"{model_bundle['ensemble_results'][model_bundle['best_name']]['accuracy']*100:.1f}%",
        ],
    ):
        with col:
            st.markdown(
                f"""<div class="content-card" style="text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:#0b3d2e;">{value}</div>
                <div style="color:#6b8f7c;font-size:0.85rem;">{label}</div></div>""",
                unsafe_allow_html=True,
            )

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.subheader("What this app does")
        st.markdown(
            """
- **Predicts** whether a traveler with your preferences would recommend a similar trip,
  using a hybrid ensemble (Random Forest + Extra Trees + Gradient Boosting + XGBoost).
- **Explains** the prediction with the top contributing factors (SHAP where available).
- **Recommends** the top matching destinations in Bangladesh based on your persona, budget,
  season and trip length.
- **Visualizes** the underlying traveler dataset so you can explore patterns yourself.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.subheader("Popular personas")
        persona_counts = df["persona"].value_counts().reset_index()
        persona_counts.columns = ["persona", "count"]
        fig = px.bar(
            persona_counts, x="count", y="persona", orientation="h",
            color="count", color_continuous_scale=["#bfe6d2", "#0b3d2e"],
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=320,
                           margin=dict(l=0, r=10, t=10, b=0), yaxis_title="", xaxis_title="Travelers")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.info("👉 Head to **Get My Recommendation** in the sidebar to try the AI on your own travel preferences.")


# ---------------------------------------------------------------- RECOMMENDATION
elif page == "🎯 Get My Recommendation":
    hero(
        "Tell us how you like to travel ✈️",
        "Answer a few quick questions and the model will predict your satisfaction and rank the "
        "best-matching destinations for you.",
        badge="AI Recommendation",
    )

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    with st.form("preferences_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Your age", 18, 70, 26)
            duration = st.slider("Trip duration (days)", 1, 14, 3)
            rating = st.slider("How would you rate your last trip? (1-5)", 1.0, 5.0, 4.0, 0.5)
        with c2:
            budget = st.selectbox("Budget", ["low", "medium", "high"], index=1)
            season = st.selectbox("Preferred season", ["summer", "winter", "monsoon", "any"], index=3)
            companion = st.selectbox("Traveling with", ["Solo", "Partner", "Family", "Friends"], index=2)
        with c3:
            available_categories = sorted(destinations_db.keys())
            category = st.selectbox(
                "Interest category",
                available_categories,
                format_func=lambda c: c.replace("-", " ").title(),
            )
            personas = ["Cloud Chaser", "Beach Baddie", "Jungle Junkie", "Heritage Nerd",
                        "Waterfall Hunter", "Street Food Explorer", "Chai & Chill", "Off-Grid Wanderer"]
            persona = st.selectbox("Travel style / persona", personas)
            accommodation = st.selectbox(
                "Accommodation",
                ["Hotel", "Resort", "Homestay", "Cottage", "Budget Hotel", "Luxury Hotel", "Hostel", "Guest House"],
            )

        hotel_review = st.text_input(
            "Describe your last hotel stay in a few words (optional)",
            value="Clean and friendly staff",
        )
        submitted = st.form_submit_button("🔍 Get My Recommendation")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        user = {
            "age": age, "duration": duration, "budget": budget, "category": category,
            "persona": persona, "companion": companion, "season": season,
            "accommodation": accommodation, "rating": rating, "hotel_review": hotel_review,
        }
        row = build_single_row(user)
        pred, prob = predict(row, feature_cols, model_bundle["pipeline"])

        colL, colR = st.columns([1, 1.4])
        with colL:
            if pred == 1:
                st.markdown(
                    f"""<div class="verdict-yes">✅ Likely to recommend this kind of trip<br>
                    <span style="font-weight:400;font-size:0.85rem;">Model confidence: {prob*100:.1f}%</span></div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div class="verdict-no">⚠️ May not fully enjoy this kind of trip<br>
                    <span style="font-weight:400;font-size:0.85rem;">Model confidence: {(1-prob)*100:.1f}%</span></div>""",
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="content-card" style="margin-top:0.8rem;">', unsafe_allow_html=True)
            st.markdown("**Top factors behind this prediction**" + ("" if HAS_SHAP else " (feature importance)"))
            factors = explain_prediction(row, feature_cols, model_bundle)
            max_abs = max(abs(v) for _, v in factors) or 1
            for name, val in factors:
                bar_pct = int(abs(val) / max_abs * 100)
                direction = "↑ increases" if val >= 0 else "↓ decreases"
                color = "#1f9d68" if val >= 0 else "#c0392b"
                st.markdown(
                    f"""<div class="factor-row">
                        <span>{name.replace('_',' ')}</span>
                        <span style="color:{color};">{direction}</span>
                    </div>
                    <div style="background:#eef5f0;border-radius:6px;height:6px;margin-bottom:8px;">
                        <div style="background:{color};width:{bar_pct}%;height:6px;border-radius:6px;"></div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with colR:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.subheader("🏝️ Top destinations for you")
            recs = recommend(user, destinations_db, top_n=5)
            if not recs:
                st.warning("No matching destinations found — try a different category.")
            for i, r in enumerate(recs, 1):
                stars = "★" * int(round(r["rating"])) + "☆" * (5 - int(round(r["rating"])))
                st.markdown(
                    f"""<div class="destination-card">
                        <span class="match-pill">{r['score']}% match</span>
                        <h4><span class="destination-rank">#{i}</span>{r['name']}</h4>
                        <div style="color:#f5b301;letter-spacing:2px;">{stars}
                            <span style="color:#555;font-size:0.8rem;">({r['rating']}/5)</span></div>
                        <div style="color:#555;font-size:0.88rem;margin-top:4px;">{r['desc']}</div>
                        <div style="margin-top:6px;font-size:0.8rem;color:#0b3d2e;">
                            💰 {r['budget'].title()} budget &nbsp;•&nbsp; 🗓️ {r['duration']} days
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            budget_tips = {
                "low": "Travel off-season and use local/shared transport to stretch your budget.",
                "medium": "Mid-range hotels and local restaurants strike a good value balance.",
                "high": "Consider premium resorts, private guides and direct transfers.",
            }
            st.markdown(
                f"""<div class="content-card">
                <b>💡 Travel tip:</b> {budget_tips.get(budget, '')}<br>
                <b>Best season:</b> {season.title()} &nbsp;|&nbsp; <b>Style:</b> {persona}
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------- INSIGHTS
elif page == "📊 Insights Dashboard":
    hero("Traveler Data Insights 📈", "Explore patterns across 420 real traveler reviews used to train the model.",
         badge="Live Dataset")

    t1, t2, t3 = st.tabs(["Overview", "Ratings & Budget", "Correlations"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            vc = df["would_recommend"].value_counts().reset_index()
            vc.columns = ["would_recommend", "count"]
            fig = px.pie(vc, names="would_recommend", values="count", hole=0.5,
                         color_discrete_sequence=["#1f9d68", "#f5b301", "#c0392b"])
            fig.update_layout(title="Would Recommend?", margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            fig = px.histogram(df, x="age", nbins=15, color_discrete_sequence=["#1f9d68"])
            fig.update_layout(title="Age Distribution", margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        cat_counts = df["preferred_categories"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.bar(cat_counts, x="category", y="count", color="count",
                     color_continuous_scale=["#bfe6d2", "#0b3d2e"])
        fig.update_layout(title="Preferred Categories", coloraxis_showscale=False, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            fig = px.box(df, x="preferred_budget", y="rating",
                         category_orders={"preferred_budget": ["low", "medium", "high"]},
                         color="preferred_budget", color_discrete_sequence=["#f5b301", "#1f9d68", "#0b3d2e"])
            fig.update_layout(title="Rating by Budget", showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            season_df = df[df["visited_season"] != "Unknown"]
            fig = px.box(season_df, x="visited_season", y="rating", color="visited_season",
                         color_discrete_sequence=px.colors.sequential.Greens_r)
            fig.update_layout(title="Rating by Season", showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        persona_counts = df["persona"].value_counts().reset_index()
        persona_counts.columns = ["persona", "count"]
        fig = px.bar(persona_counts, x="persona", y="count", color="count",
                     color_continuous_scale=["#bfe6d2", "#0b3d2e"])
        fig.update_layout(title="Travelers by Persona", coloraxis_showscale=False, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        num_cols = ["rating", "age", "trip_duration_days", "hotel_sentiment_score",
                    "num_preferred_categories", "experience_score"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Greens", aspect="auto")
        fig.update_layout(title="Feature Correlation Heatmap", margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        fig = px.scatter(df, x="age", y="rating", color="preferred_budget", trendline=None,
                          color_discrete_sequence=["#f5b301", "#1f9d68", "#0b3d2e"])
        fig.update_layout(title="Age vs Rating", margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- MODEL PERFORMANCE
elif page == "🤖 Model Performance":
    hero("Under the Hood 🤖", "How the hybrid ensemble stacks up against individual models.",
         badge="Model Diagnostics")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("Base models")
    base_rows = [
        {"Model": name, "Accuracy": r["accuracy"], "F1": r["f1"], "Precision": r["precision"],
         "Recall": r["recall"], "AUC": r["auc"]}
        for name, r in model_bundle["base_results"].items()
    ]
    st.dataframe(pd.DataFrame(base_rows).style.format({c: "{:.3f}" for c in
                 ["Accuracy", "F1", "Precision", "Recall", "AUC"]}), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("Weighted voting ensembles")
    ens_rows = [
        {"Configuration": name, "Weights": str(r["weights"]), "Accuracy": r["accuracy"],
         "F1": r["f1"], "AUC": r["auc"], "CV Mean": r["cv_mean"]}
        for name, r in model_bundle["ensemble_results"].items()
    ]
    ens_df = pd.DataFrame(ens_rows)
    st.dataframe(
        ens_df.style.format({c: "{:.3f}" for c in ["Accuracy", "F1", "AUC", "CV Mean"]})
        .apply(lambda s: ["background-color:#e6f6ee" if s["Configuration"] == model_bundle["best_name"] else ""
                           for _ in s], axis=1),
        use_container_width=True, hide_index=True,
    )
    st.success(f"🏆 Best configuration: **{model_bundle['best_name']}** — used for live predictions.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    fig = px.bar(ens_df, x="Configuration", y="Accuracy", color="Accuracy",
                 color_continuous_scale=["#bfe6d2", "#0b3d2e"])
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- ABOUT
else:
    hero("About this project ℹ️", "A student/portfolio ML project turning a tourism survey into a live recommender.",
         badge="v1.0")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown(
        """
**Pipeline**
1. Load & clean 420 traveler reviews across 66 Bangladeshi destinations.
2. Engineer features: review sentiment, category counts, age groups, seasonality.
3. Train a hybrid ensemble (Random Forest, Extra Trees, Gradient Boosting, XGBoost) with
   soft-voting and weight-search, picking the best configuration on held-out accuracy.
4. Explain each prediction with SHAP (falls back to feature importance if SHAP isn't installed).
5. Score destinations against the user's category, persona, budget and trip length.

**Tech stack:** Python, scikit-learn, XGBoost, SHAP, Plotly, Streamlit.

**Data note:** all names, hotels and reviews are from a synthetic/sample survey dataset for
demonstration purposes — recommendations are illustrative, not verified travel advice.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="gu-footer">🧭 CholoGhuri — Bangladesh Tourism AI Recommender — built with Streamlit</div>',
            unsafe_allow_html=True)
