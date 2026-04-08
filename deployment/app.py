from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier


st.set_page_config(
    page_title="Crime Hotspot Predictor",
    page_icon="docs/icons8-detain-50.png",
    layout="wide",
)


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_FILE = Path(__file__).resolve()
DATA_FILE = ROOT_DIR / "data" / "chicago_crime_2015_2025.parquet"
MODEL_FILE = Path(__file__).resolve().with_name("hotspot_model.joblib")

BROAD_COLS = ["violent", "property", "drug", "public_order", "other"]
FEATURE_COLUMNS = [
    "total_crimes",
    "lag_1w",
    "lag_2w",
    "rolling_mean_4w",
    "week_of_year",
    "ratio_violent",
    "ratio_property",
    "ratio_drug",
    "ratio_public_order",
]

FALLBACK_DEFAULTS = {
    "total_crimes": 120.0,
    "lag_1w": 118.0,
    "lag_2w": 115.0,
    "rolling_mean_4w": 117.0,
    "week_of_year": 26.0,
    "ratio_violent": 0.24,
    "ratio_property": 0.49,
    "ratio_drug": 0.13,
    "ratio_public_order": 0.08,
}

REFERENCE_METRICS = {
    "model_name": "Gradient Boosting",
    "threshold": 0.40,
    "test_auc": 0.9143,
    "precision": 0.8088,
    "recall": 0.3595,
    "f1": 0.4977,
    "train_period": "2015-2023",
    "validation_period": "2024",
    "test_period": "2025",
}


def map_chicago_broad_category(primary_type: str) -> str:
    crime = str(primary_type).strip().lower()

    if crime in {
        "homicide",
        "assault",
        "battery",
        "robbery",
        "criminal sexual assault",
        "crim sexual assault",
        "sex offense",
        "kidnapping",
        "human trafficking",
    }:
        return "violent"

    if crime in {
        "theft",
        "burglary",
        "motor vehicle theft",
        "criminal damage",
        "deceptive practice",
        "arson",
        "criminal trespass",
        "possession of stolen property",
    }:
        return "property"

    if crime in {"narcotics", "other narcotic violation"}:
        return "drug"

    if crime in {
        "weapons violation",
        "public peace violation",
        "interference with public officer",
        "obscenity",
        "gambling",
        "liquor law violation",
        "other offense",
        "offense involving children",
    }:
        return "public_order"

    return "other"


@st.cache_data(show_spinner=False)
def load_raw_chicago_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing chicago_crime_2015_2025.parquet in the data directory."
        )

    return pd.read_parquet(DATA_FILE)


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["district"] = pd.to_numeric(frame["district"], errors="coerce")
    frame["primary_type"] = frame["primary_type"].astype(str).str.strip().str.upper()
    frame = frame.dropna(subset=["date", "district", "primary_type"])
    frame["district"] = frame["district"].astype(int)

    frame["week_start"] = frame["date"].dt.to_period("W").dt.start_time
    frame["broad_cat"] = frame["primary_type"].apply(map_chicago_broad_category)

    weekly_counts = (
        frame.groupby(["district", "week_start", "broad_cat"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for column in BROAD_COLS:
        if column not in weekly_counts.columns:
            weekly_counts[column] = 0

    weekly_counts["total_crimes"] = weekly_counts[BROAD_COLS].sum(axis=1)

    for column in BROAD_COLS:
        weekly_counts[f"ratio_{column}"] = (
            weekly_counts[column] / weekly_counts["total_crimes"].replace(0, 1)
        )

    weekly_counts["week_of_year"] = (
        weekly_counts["week_start"].dt.isocalendar().week.astype(int)
    )
    weekly_counts["month"] = weekly_counts["week_start"].dt.month.astype(int)
    weekly_counts["year"] = weekly_counts["week_start"].dt.year.astype(int)

    weekly_counts = weekly_counts.sort_values(["district", "week_start"]).copy()
    weekly_counts["lag_1w"] = (
        weekly_counts.groupby("district")["total_crimes"].shift(1)
    )
    weekly_counts["lag_2w"] = (
        weekly_counts.groupby("district")["total_crimes"].shift(2)
    )
    weekly_counts["rolling_mean_4w"] = (
        weekly_counts.groupby("district")["total_crimes"]
        .transform(lambda series: series.shift(1).rolling(4, min_periods=1).mean())
    )

    weekly_counts = weekly_counts.dropna(
        subset=["lag_1w", "lag_2w", "rolling_mean_4w"]
    ).copy()

    weekly_counts = weekly_counts.sort_values(
        ["week_start", "total_crimes"],
        ascending=[True, False],
    ).copy()
    weekly_counts["weekly_rank"] = (
        weekly_counts.groupby("week_start")["total_crimes"]
        .rank(method="first", ascending=False)
    )
    weekly_counts["is_hotspot"] = (weekly_counts["weekly_rank"] <= 3).astype(int)

    weekly_counts = weekly_counts.sort_values(["district", "week_start"]).copy()
    weekly_counts["target_next_week"] = (
        weekly_counts.groupby("district")["is_hotspot"].shift(-1)
    )

    modeling_frame = weekly_counts.dropna(subset=["target_next_week"]).copy()
    modeling_frame["target_next_week"] = modeling_frame["target_next_week"].astype(int)

    return modeling_frame


@st.cache_data(show_spinner=False)
def load_training_frame() -> pd.DataFrame:
    raw_data = load_raw_chicago_data()
    modeling_frame = build_modeling_frame(raw_data)
    return modeling_frame[modeling_frame["year"] <= 2023].copy()


def default_inputs() -> dict[str, float]:
    if not DATA_FILE.exists():
        return FALLBACK_DEFAULTS.copy()

    training_frame = load_training_frame()
    medians = training_frame[FEATURE_COLUMNS].median().to_dict()
    defaults = FALLBACK_DEFAULTS.copy()
    defaults.update({key: float(value) for key, value in medians.items()})
    return defaults


@st.cache_resource(show_spinner=False)
def load_or_train_artifact() -> tuple[dict, str]:
    if MODEL_FILE.exists():
        try:
            artifact = joblib.load(MODEL_FILE)
            loaded_metrics = artifact.get("metrics", {})
            loaded_model_name = loaded_metrics.get("model_name")
            if loaded_model_name == REFERENCE_METRICS["model_name"]:
                artifact.setdefault("features", FEATURE_COLUMNS)
                artifact.setdefault("threshold", REFERENCE_METRICS["threshold"])
                artifact.setdefault("defaults", default_inputs())
                artifact.setdefault("metrics", REFERENCE_METRICS)
                return artifact, "loaded"
        except Exception:
            pass

    training_frame = load_training_frame()
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    X_train = training_frame[FEATURE_COLUMNS].copy()
    y_train = training_frame["target_next_week"].astype(int)
    model.fit(X_train, y_train)

    artifact = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "threshold": REFERENCE_METRICS["threshold"],
        "defaults": default_inputs(),
        "metrics": REFERENCE_METRICS,
    }
    return artifact, "trained"


def make_prediction(
    model_bundle: dict,
    feature_values: dict[str, float],
) -> tuple[float, int]:
    feature_order = model_bundle["features"]
    input_frame = pd.DataFrame([feature_values])[feature_order]
    probability = float(model_bundle["model"].predict_proba(input_frame)[0, 1])
    prediction = int(probability >= model_bundle["threshold"])
    return probability, prediction


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR))
    except ValueError:
        return str(resolved)


artifact, artifact_source = load_or_train_artifact()
defaults = artifact.get("defaults", default_inputs())
metrics = artifact.get("metrics", REFERENCE_METRICS)

st.title("Chicago Crime Hotspot Predictor")
st.caption(
    "Proof-of-concept Streamlit app for IT5006 Milestone 3. "
    "This model predicts whether a district is likely to be a hotspot next week."
)

status_label = (
    "Loaded pre-trained artifact from deployment/hotspot_model.joblib"
    if artifact_source == "loaded"
    else "No saved artifact found. A fallback model was trained automatically from the local Chicago dataset."
)
st.info(status_label)

with st.sidebar:
    st.header("Model Summary")
    st.write(f"Model: `{metrics['model_name']}`")
    st.write(f"Decision threshold: `{metrics['threshold']}`")
    st.write(f"Training period: `{metrics['train_period']}`")
    st.write(f"Validation period: `{metrics['validation_period']}`")
    st.write(f"Reference test period: `{metrics['test_period']}`")
    st.metric("Reference Test AUC", f"{metrics['test_auc']:.3f}")
    st.metric("Reference Test F1", f"{metrics['f1']:.3f}")
    st.metric("Reference Test Recall", f"{metrics['recall']:.3f}")
    st.metric("Reference Test Precision", f"{metrics['precision']:.3f}")

    st.divider()
    st.write("Repository paths")
    st.code(f"App: {display_path(APP_FILE)}")
    st.code(f"Data: {display_path(DATA_FILE)}")
    st.code(f"Artifact: {display_path(MODEL_FILE)}")

st.markdown(
    """
    Enter weekly district-level features below. These are the same features used in the
    notebook model:

    - `total_crimes`: crime count in the current week
    - `lag_1w`, `lag_2w`: total crimes in the previous 1 and 2 weeks
    - `rolling_mean_4w`: average weekly crime count over the previous 4 weeks
    - `week_of_year`: ISO week number
    - `ratio_*`: share of current-week crimes in each category
    """
)

with st.form("prediction_form"):
    count_col, lag_col, ratio_col = st.columns(3)

    with count_col:
        total_crimes = st.number_input(
            "Current week total crimes",
            min_value=0.0,
            value=float(defaults["total_crimes"]),
            step=1.0,
        )
        week_of_year = st.slider(
            "Week of year",
            min_value=1,
            max_value=53,
            value=int(round(defaults["week_of_year"])),
        )

    with lag_col:
        lag_1w = st.number_input(
            "Lag 1 week",
            min_value=0.0,
            value=float(defaults["lag_1w"]),
            step=1.0,
        )
        lag_2w = st.number_input(
            "Lag 2 weeks",
            min_value=0.0,
            value=float(defaults["lag_2w"]),
            step=1.0,
        )
        rolling_mean_4w = st.number_input(
            "Rolling mean 4 weeks",
            min_value=0.0,
            value=float(defaults["rolling_mean_4w"]),
            step=1.0,
        )

    with ratio_col:
        ratio_violent = st.slider(
            "Violent ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["ratio_violent"]),
            step=0.01,
        )
        ratio_property = st.slider(
            "Property ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["ratio_property"]),
            step=0.01,
        )
        ratio_drug = st.slider(
            "Drug ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["ratio_drug"]),
            step=0.01,
        )
        ratio_public_order = st.slider(
            "Public-order ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(defaults["ratio_public_order"]),
            step=0.01,
        )

    submitted = st.form_submit_button("Predict Next Week Hotspot Risk")


if submitted:
    feature_values = {
        "total_crimes": float(total_crimes),
        "lag_1w": float(lag_1w),
        "lag_2w": float(lag_2w),
        "rolling_mean_4w": float(rolling_mean_4w),
        "week_of_year": float(week_of_year),
        "ratio_violent": float(ratio_violent),
        "ratio_property": float(ratio_property),
        "ratio_drug": float(ratio_drug),
        "ratio_public_order": float(ratio_public_order),
    }

    ratio_sum = (
        feature_values["ratio_violent"]
        + feature_values["ratio_property"]
        + feature_values["ratio_drug"]
        + feature_values["ratio_public_order"]
    )

    if ratio_sum > 1.0:
        st.warning(
            "The four ratio inputs add up to more than 1.0. "
            "Please double-check the category shares before using this output."
        )

    probability, prediction = make_prediction(artifact, feature_values)
    risk_percentage = int(round(probability * 100))

    result_col, summary_col = st.columns([1, 1])

    with result_col:
        st.subheader("Prediction")
        st.progress(min(max(risk_percentage, 0), 100))
        st.metric("Hotspot probability", f"{probability:.2%}")
        st.metric("Decision threshold", f"{artifact['threshold']:.2f}")

        if prediction == 1:
            st.error("Predicted outcome: Hotspot next week")
        else:
            st.success("Predicted outcome: Not a hotspot next week")

    with summary_col:
        st.subheader("Input Summary")
        st.dataframe(
            pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "value": [feature_values[column] for column in FEATURE_COLUMNS],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

st.divider()
with st.expander("Usage Notes"):
    st.markdown(
        """
        - This app is a proof-of-concept for coursework demonstration.
        - Predictions are based on weekly district-level aggregate features, not raw incident text.
        - The model should be used for analysis and demo purposes, NOT operational policing decisions.
        - Team members: LIN YIHAN, QI RUIXUAN, XU QIAOYANG
        """
    )
