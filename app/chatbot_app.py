"""Streamlit chatbot for student dropout risk (local version)."""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import (
    APP_EXPLANATION_FEATURES,
    APP_INPUT_FIELDS,
    APP_SECTIONS,
    INT_INPUT_FIELDS,
    field_options,
    pretty_name,
)
from src.predict import get_predictor

st.set_page_config(
    page_title="Student Dropout Risk Assistant",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    return get_predictor()


try:
    predictor = load_predictor()
except Exception as exc:
    st.error(
        "Could not load the model artifacts. Run the training notebook first "
        "so the models/ folder is populated."
    )
    st.exception(exc)
    st.stop()

DEFAULTS = predictor.default_input_values
ALL_FEATURES = predictor.raw_features
INPUT_FIELDS = list(APP_INPUT_FIELDS)
HIDDEN_FIELDS = [field for field in ALL_FEATURES if field not in INPUT_FIELDS]

FIELD_HELP = {
    "Course": "Select the course the student is enrolled in.",
    "Age at enrollment": "Age when the student started the course.",
    "Gender": "Recorded in the dataset as female or male.",
    "Displaced": "The student moved away from home to study.",
    "Admission grade": "Admission score on the 0-200 scale.",
    "Curricular units 1st sem (enrolled)": "Units enrolled in semester 1.",
    "Curricular units 1st sem (approved)": "Units approved in semester 1.",
    "Curricular units 1st sem (grade)": "Average grade in semester 1 (0-20).",
    "Curricular units 2nd sem (enrolled)": "Units enrolled in semester 2.",
    "Curricular units 2nd sem (approved)": "Units approved in semester 2.",
    "Curricular units 2nd sem (grade)": "Average grade in semester 2 (0-20).",
    "Tuition fees up to date": "Whether tuition fees are currently paid.",
    "Debtor": "Whether the student has outstanding debt.",
    "Scholarship holder": "Whether the student receives a scholarship.",
}

FIELD_RANGES = {
    "Age at enrollment": (16, 80, 1),
    "Admission grade": (0.0, 200.0, 0.1),
    "Curricular units 1st sem (enrolled)": (0, 30, 1),
    "Curricular units 1st sem (approved)": (0, 30, 1),
    "Curricular units 1st sem (grade)": (0.0, 20.0, 0.1),
    "Curricular units 2nd sem (enrolled)": (0, 30, 1),
    "Curricular units 2nd sem (approved)": (0, 30, 1),
    "Curricular units 2nd sem (grade)": (0.0, 20.0, 0.1),
}

HIGH_RISK_UPDATES = {
    "Course": 9500,
    "Age at enrollment": 21,
    "Gender": 0,
    "Displaced": 0,
    "Admission grade": 95.0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (approved)": 0,
    "Curricular units 1st sem (grade)": 0.0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (approved)": 0,
    "Curricular units 2nd sem (grade)": 0.0,
    "Tuition fees up to date": 0,
    "Debtor": 1,
    "Scholarship holder": 0,
}

LOW_RISK_UPDATES = {
    "Course": 9500,
    "Age at enrollment": 19,
    "Gender": 0,
    "Displaced": 0,
    "Admission grade": 150.0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 16.0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (approved)": 6,
    "Curricular units 2nd sem (grade)": 16.0,
    "Tuition fees up to date": 1,
    "Debtor": 0,
    "Scholarship holder": 1,
}


def field_key(field: str) -> str:
    return f"input_{field}"


def default_value(field: str):
    value = DEFAULTS[field]
    return int(value) if field in INT_INPUT_FIELDS else value


for field in INPUT_FIELDS:
    st.session_state.setdefault(field_key(field), default_value(field))


def reset_defaults():
    for field in INPUT_FIELDS:
        st.session_state[field_key(field)] = default_value(field)


def load_example(kind: str):
    reset_defaults()
    updates = HIGH_RISK_UPDATES if kind == "high" else LOW_RISK_UPDATES
    for field, value in updates.items():
        if field in INPUT_FIELDS:
            st.session_state[field_key(field)] = value


def current_values() -> dict:
    return {field: st.session_state[field_key(field)] for field in INPUT_FIELDS}


def full_input(values: dict) -> dict:
    """Combine the form values with typical values for background fields."""
    return {**DEFAULTS, **values}


def validation_errors(values: dict) -> list[str]:
    errors = []
    for semester, label in (("1st", "1"), ("2nd", "2")):
        enrolled = values[f"Curricular units {semester} sem (enrolled)"]
        approved = values[f"Curricular units {semester} sem (approved)"]
        if approved > enrolled:
            errors.append(
                f"Semester {label}: approved units cannot exceed enrolled units."
            )
    return errors


def render_field(field: str):
    label = pretty_name(field)
    key = field_key(field)
    help_text = FIELD_HELP.get(field)
    options = field_options(field)

    if options:
        values = [value for _, value in options]
        labels = {value: text for text, value in options}
        if len(options) <= 3:
            st.segmented_control(
                label,
                options=values,
                format_func=lambda value: labels[value],
                key=key,
                help=help_text,
                required=True,
                width="stretch",
            )
        else:
            st.selectbox(
                label,
                options=values,
                format_func=lambda value: labels[value],
                key=key,
                help=help_text,
            )
        return

    minimum, maximum, step = FIELD_RANGES.get(field, (0.0, None, 0.1))
    if field in INT_INPUT_FIELDS:
        st.number_input(
            label,
            min_value=int(minimum),
            max_value=int(maximum) if maximum is not None else None,
            step=int(step),
            key=key,
            help=help_text,
        )
    else:
        st.number_input(
            label,
            min_value=float(minimum),
            max_value=float(maximum) if maximum is not None else None,
            step=float(step),
            format="%.2f",
            key=key,
            help=help_text,
        )


def show_result(result: dict):
    probability = result["dropout_probability"]
    risk = result["risk_level"]

    with st.container(border=True):
        left, right = st.columns([2, 1])
        with left:
            st.metric(
                "Estimated dropout probability",
                f"{result['dropout_probability_percent']}%",
            )
            st.progress(min(max(probability, 0.0), 1.0))
        with right:
            st.metric("Risk level", risk)
            st.caption(
                "Flagged for review"
                if result["predicted_dropout"]
                else "Not flagged"
            )
        st.caption(
            f"Decision threshold: {result['threshold_used']:.2f}. "
            "The estimate uses the form fields plus typical values for "
            "background details that are not requested."
        )

    st.subheader("Key factors")
    reasons = pd.DataFrame(
        [
            {
                "Reason": item["reason"],
                "Area": item["category"],
                "Share": f"{item['percent_contribution']:.1f}%",
                "Direction": item["direction"],
            }
            for item in result["top_reasons"][:5]
        ]
    )
    st.dataframe(reasons, width="stretch", hide_index=True)
    st.caption(
        "Only the fields in the form are shown here. Background fields are held "
        "at typical values and are excluded from this list."
    )

    st.subheader("Where the estimate comes from")
    category_df = pd.DataFrame(result["category_contributions"])
    if not category_df.empty:
        figure = px.bar(
            category_df,
            x="percent_contribution",
            y="category",
            orientation="h",
            text="percent_contribution",
            color="percent_contribution",
            color_continuous_scale="Blues",
            labels={
                "percent_contribution": "Share of explanation (%)",
                "category": "",
            },
        )
        figure.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        figure.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=300,
            margin=dict(l=10, r=40, t=20, b=20),
            xaxis_range=[0, max(60, category_df["percent_contribution"].max() * 1.2)],
        )
        st.plotly_chart(
            figure,
            width="stretch",
            alt="Bar chart of explanation share by category",
        )

    st.subheader("Recommended actions")
    for recommendation in result["recommendations"][:4]:
        st.markdown(f"- {recommendation}")

    st.caption(result["disclaimer"])


st.title("Student Dropout Risk Assistant")
st.markdown(
    "Enter the core details below to estimate dropout risk and receive "
    "supportive recommendations. This tool is designed for early support, "
    "not punishment."
)
st.info(
    "This prediction is a decision-support tool. Human review is required before "
    "any academic or financial action is taken.",
    title="Use with human review",
)

with st.container(horizontal=True):
    st.button("Use typical student", on_click=reset_defaults)
    st.button(
        "High-risk example",
        on_click=functools.partial(load_example, "high"),
    )
    st.button(
        "Low-risk example",
        on_click=functools.partial(load_example, "low"),
    )

with st.form("assessment_form"):
    for section, fields in APP_SECTIONS.items():
        st.markdown(f"#### {section}")
        columns = st.columns(2)
        for index, field in enumerate(fields):
            with columns[index % 2]:
                render_field(field)

    st.caption(
        f"Only these {len(INPUT_FIELDS)} fields are requested. Marital status, "
        "nationality, family occupation, application details, and macro "
        "indicators are not needed for the estimate; they use typical training "
        "values."
    )
    submitted = st.form_submit_button(
        "Predict risk", type="primary", width="stretch"
    )

if submitted:
    values = current_values()
    errors = validation_errors(values)
    if errors:
        for error in errors:
            st.error(error)
    else:
        try:
            st.session_state["result"] = predictor.predict(
                full_input(values),
                explanation_features=APP_EXPLANATION_FEATURES,
            )
        except Exception as exc:
            st.error("The prediction could not be generated.")
            st.exception(exc)

result = st.session_state.get("result")
if result:
    show_result(result)

st.divider()
st.subheader("Ask the assistant")
st.caption(
    "Commands: `predict`, `use typical student`, `explain risk`, "
    "`show recommendations`."
)

if "chat" not in st.session_state:
    st.session_state["chat"] = []

for message in st.session_state["chat"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about dropout risk or type a command")
if prompt:
    st.session_state["chat"].append({"role": "user", "content": prompt})
    command = prompt.strip().lower()

    if command in {"predict", "predict risk"}:
        values = current_values()
        errors = validation_errors(values)
        if errors:
            reply = "Please fix the form first: " + " ".join(errors)
        else:
            result = predictor.predict(
                full_input(values),
                explanation_features=APP_EXPLANATION_FEATURES,
            )
            st.session_state["result"] = result
            reply = (
                f"Estimated dropout probability is "
                f"**{result['dropout_probability_percent']}%** "
                f"({result['risk_level']} risk)."
            )
    elif command in {"use typical student", "use default student", "default"}:
        reset_defaults()
        result = predictor.predict(
            dict(DEFAULTS),
            explanation_features=APP_EXPLANATION_FEATURES,
        )
        st.session_state["result"] = result
        reply = (
            "Loaded the typical student. "
            f"Estimated dropout probability is "
            f"**{result['dropout_probability_percent']}%** "
            f"({result['risk_level']} risk)."
        )
    elif command in {"explain risk", "explain"}:
        current = st.session_state.get("result")
        if current:
            lines = [
                f"- {item['reason']} ({item['category']}, "
                f"{item['percent_contribution']:.1f}%)"
                for item in current["top_reasons"][:5]
            ]
            reply = "Top factors:\n" + "\n".join(lines)
        else:
            reply = "Please run a prediction first."
    elif command in {"show recommendations", "recommendations"}:
        current = st.session_state.get("result")
        if current:
            reply = "Recommended actions:\n" + "\n".join(
                f"- {item}" for item in current["recommendations"]
            )
        else:
            reply = "Please run a prediction first."
    else:
        reply = (
            "I can help with: `predict`, `use typical student`, "
            "`explain risk`, `show recommendations`."
        )

    st.session_state["chat"].append({"role": "assistant", "content": reply})
    st.rerun()

with st.sidebar:
    st.header("Model information")
    st.metric("Decision threshold", f"{predictor.threshold:.2f}")
    st.metric("Form fields", len(INPUT_FIELDS))
    st.caption(
        f"The model uses {len(predictor.feature_columns)} internal features. "
        f"The other {len(HIDDEN_FIELDS)} background fields use typical values."
    )
    st.caption("Artifacts are loaded from models/. The app never retrains.")
    st.divider()
    st.markdown("**Risk bands**")
    for level, band in predictor.risk_bands.items():
        upper = min(band[1], 1.0)
        st.markdown(f"- {level}: {band[0]:.0%} - {upper:.0%}")
    st.divider()
    st.caption(
        "Privacy: do not enter names, student IDs, or other direct identifiers."
    )
