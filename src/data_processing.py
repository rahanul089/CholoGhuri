"""
Data loading, cleaning and feature engineering for the
CholoGhuri, a Bangladesh Tourism AI Recommender.

This mirrors the preprocessing logic from the original research
notebook (EnhancedDataPreprocessor) but is refactored into plain
functions so it can be cleanly cached inside Streamlit.
"""

import numpy as np
import pandas as pd

POSITIVE_WORDS = {
    "loved", "clean", "friendly", "great", "comfortable", "spacious", "good",
    "excellent", "wonderful", "amazing", "beautiful", "nice", "perfect",
    "awesome", "fantastic",
}
NEGATIVE_WORDS = {
    "poor", "bad", "noisy", "thin", "misled", "disappointing", "overpriced",
    "dirty", "unclean", "terrible", "awful", "horrible", "worst", "rude",
    "uncomfortable",
}

CATEGORICAL_FEATURES = [
    "visited_season", "travel_companion", "persona", "preferred_categories",
    "preferred_budget", "accommodation_type", "primary_category", "age_group",
]
NUMERICAL_FEATURES = [
    "age", "trip_duration_days", "rating", "num_preferred_categories",
    "hotel_sentiment_score", "experience_score", "review_month", "is_weekend",
]


def _sentiment_score(text):
    if pd.isna(text) or text == "No review provided":
        return 0
    t = str(text).lower()
    return sum(w in t for w in POSITIVE_WORDS) - sum(w in t for w in NEGATIVE_WORDS)


def _primary_category(x):
    if isinstance(x, str) and x not in ("nan", ""):
        return x.split("|")[0]
    return "Unknown"


def _num_categories(x):
    if isinstance(x, str) and x not in ("nan", ""):
        return len(x.split("|"))
    return 0


def _age_group(age):
    bins = [0, 20, 25, 30, 35, 40, 100]
    labels = ["Teen(<=20)", "Young(21-25)", "Adult(26-30)", "Older(31-35)", "Senior(36-40)", "Elder(40+)"]
    for i in range(len(bins) - 1):
        if bins[i] < age <= bins[i + 1]:
            return labels[i]
    return labels[-1]


def load_and_engineer(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and apply the same imputation + feature
    engineering steps used to train the original model."""
    df = pd.read_csv(csv_path)

    imputation_config = {
        "age": "median",
        "trip_duration_days": "median",
        "rating": "median",
        "preferred_categories": "mode",
        "preferred_budget": "mode",
        "visited_season": "Unknown",
        "travel_companion": "Not Specified",
        "accommodation_type": "Hotel",
        "hotel_review": "No review provided",
    }
    for col, strategy in imputation_config.items():
        if col in df.columns:
            if strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "mode":
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna(strategy)

    # sentiment features
    df["hotel_sentiment_score"] = df["hotel_review"].apply(_sentiment_score)
    df["hotel_review_length"] = df["hotel_review"].astype(str).str.len()
    df["hotel_review_words"] = df["hotel_review"].astype(str).str.split().str.len()

    # category features
    df["num_preferred_categories"] = df["preferred_categories"].apply(_num_categories)
    df["primary_category"] = df["preferred_categories"].apply(_primary_category)

    # derived features
    df["experience_score"] = df["rating"] + df["hotel_sentiment_score"]
    df["age_group"] = df["age"].apply(_age_group)

    # date features
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["review_month"] = df["review_date"].dt.month.fillna(7).astype(int)
    df["review_year"] = df["review_date"].dt.year.fillna(2024).astype(int)
    df["day_of_week"] = df["review_date"].dt.dayofweek.fillna(3).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # target: collapse Yes/Maybe -> 1 (positive), No -> 0
    df["would_recommend_encoded"] = (df["would_recommend"] != "No").astype(int)

    return df


def get_feature_columns():
    return CATEGORICAL_FEATURES + NUMERICAL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def build_single_row(user: dict) -> pd.DataFrame:
    """Turn a single user's form answers into a one-row DataFrame with
    the same engineered columns the model was trained on."""
    sentiment = _sentiment_score(user["hotel_review"])
    data = {
        "age": user["age"],
        "trip_duration_days": user["duration"],
        "preferred_budget": user["budget"],
        "preferred_categories": user["category"],
        "visited_season": user["season"],
        "travel_companion": user["companion"],
        "persona": user["persona"],
        "accommodation_type": user["accommodation"],
        "rating": user["rating"],
        "hotel_sentiment_score": sentiment,
        "experience_score": user["rating"] + sentiment,
        "num_preferred_categories": 1,
        "primary_category": _primary_category(user["category"]),
        "age_group": _age_group(user["age"]),
        "review_month": user.get("review_month", 7),
        "is_weekend": 0,
    }
    return pd.DataFrame([data])
