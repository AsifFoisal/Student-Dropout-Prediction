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
    BINARY_FLAG_COLS,
    CATEGORICAL_CODE_COLS,
    binary_options,
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
RAW_FEATURES = predictor.raw_features

SECTIONS = {
    "Basic Information": [
        "Marital status",
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Nacionality",
        "Gender",
        "Age at enrollment",
        "International",
        "Displaced",
        "Educational special needs",
    ],
    "Family Background": [
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ],
    "Academic Entry": [
        "Previous qualification (grade)",
        "Admission grade",
    ],
    "Financial Information": [
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
    ],
    "Semester 1 Performance": [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
    ],
    "Semester 2 Performance": [
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
    ],
    "Economic Indicators": [
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ],
}

RISK_COLORS = {
    "Low": "#2e7d32",
    "Medium": "#f9a825",
    "High": "#ef6c00",
    "Critical": "#c62828",
}

HIGH_RISK_UPDATES = {
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (approved)": 0,
    "Curricular units 1st sem (grade)": 0.0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (approved)": 0,
    "Curricular units 2nd sem (grade)": 0.0,
    "Debtor": 1,
    "Tuition fees up to date": 0,
    "Scholarship holder": 0,
    "Admission grade": 95.0,
}

LOW_RISK_UPDATES = {
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 16.0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (approved)": 6,
    "Curricular units 2nd sem (grade)": 16.0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Scholarship holder": 1,
    "Admission grade": 150.0,
}


def field_key(field: str) -> str:
    return f"input_{field}"


for field in RAW_FEATURES:
    st.session_state.setdefault(field_key(field), DEFAULTS[field])


def reset_defaults():
    for field in RAW_FEATURES:
        st.session_state[field_key(field)] = DEFAULTS[field]


def load_example(kind: str):
    reset_defaults()
    updates = HIGH_RISK_UPDATES if kind == "high" else LOW_RISK_UPDATES
    for field, value in updates.items():
        st.session_state[field_key(field)] = value


def render_field(field: str):
    label = pretty_name(field)
    key = field_key(field)

    if field in BINARY_FLAG_COLS:
        labels = {value: text for text, value in binary_options(field)}
        st.selectbox(
            label,
            options=[0, 1],
            format_func=lambda value: labels[value],
            key=key,
        )
    elif field in CATEGORICAL_CODE_COLS or field == "Age at enrollment":
        st.number_input(label, min_value=0, step=1, key=key)
    else:
        st.number_input(label, step=0.1, format="%.2f", key=key)


def show_result(result: dict):
    probability = result["dropout_probability"]
    risk = result["risk_level"]
    color = RISK_COLORS.get(risk, "#555555")

    st.markdown(
        f"""
        <div style="border:1px solid #e0e0e0;border-radius:14px;padding:20px;margin:10px 0;
                    background:linear-gradient(135deg,#ffffff,#f7f9fc)">
            <div style="font-size:14px;color:#666">Estimated dropout probability</div>
            <div style="font-size:46px;font-weight:700;line-height:1.1">{result['dropout_probability_percent']}%</div>
            <div style="font-size:20px;font-weight:600;color:{color};margin-top:6px">Risk level: {risk}</div>
            <div style="font-size:13px;color:#777;margin-top:6px">
                Decision threshold: {result['threshold_used']:.2f} -
                Classification: {'higher risk' if result['predicted_dropout'] else 'not flagged'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(probability, 0.0), 1.0))

    if result["missing_fields_filled"]:
        labels = [pretty_name(field) for field in result["missing_fields_filled"]]
        st.warning(
            "Some missing values were filled using typical training data values: "
            + ", ".join(labels)
        )

    st.subheader("Key factors")
    reasons = pd.DataFrame(
        [
            {
                "Reason": item["reason"],
                "Category": item["category"],
                "Contribution": f"{item['percent_contribution']:.1f}%",
                "Direction": item["direction"],
            }
            for item in result["top_reasons"]
        ]
    )
    st.dataframe(reasons, width="stretch", hide_index=True)

    st.subheader("Category contribution")
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
            height=340,
            margin=dict(l=10, r=40, t=20, b=20),
            xaxis_range=[0, max(60, category_df["percent_contribution"].max() * 1.2)],
        )
        st.plotly_chart(figure, width="stretch")

    st.subheader("Recommended actions")
    for recommendation in result["recommendations"]:
        st.markdown(f"- {recommendation}")

    st.caption(result["disclaimer"])


st.title("Student Dropout Risk Assistant")
st.markdown(
    "Enter student information to estimate dropout risk and receive supportive "
    "recommendations. This tool is designed for early support, not punishment."
)
st.info(
    "This prediction is a decision-support tool. Human review is required before "
    "any academic or financial action is taken."
)

button_col1, button_col2, button_col3 = st.columns(3)
button_col1.button(
    "Use Default Sample Student",
    on_click=reset_defaults,
    width="stretch",
)
button_col2.button(
    "Load High-Risk Example",
    on_click=functools.partial(load_example, "high"),
    width="stretch",
)
button_col3.button(
    "Load Low-Risk Example",
    on_click=functools.partial(load_example, "low"),
    width="stretch",
)

with st.form("assessment_form"):
    for section, fields in SECTIONS.items():
        st.markdown(f"#### {section}")
        columns = st.columns(2)
        for index, field in enumerate(fields):
            with columns[index % 2]:
                render_field(field)
    submitted = st.form_submit_button(
        "Predict Risk", type="primary", width="stretch"
    )

if submitted:
    values = {field: st.session_state[field_key(field)] for field in RAW_FEATURES}
    try:
        st.session_state["result"] = predictor.predict(values)
    except Exception as exc:
        st.error("The prediction could not be generated.")
        st.exception(exc)

result = st.session_state.get("result")
if result:
    show_result(result)

st.divider()
st.subheader("Ask the assistant")
st.caption(
    "Commands: `predict`, `use default student`, `explain risk`, "
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
        values = {
            field: st.session_state[field_key(field)] for field in RAW_FEATURES
        }
        result = predictor.predict(values)
        st.session_state["result"] = result
        reply = (
            f"Estimated dropout probability is **{result['dropout_probability_percent']}%** "
            f"({result['risk_level']} risk)."
        )
    elif command in {"use default student", "default"}:
        reset_defaults()
        result = predictor.predict(dict(DEFAULTS))
        st.session_state["result"] = result
        reply = (
            "Loaded the default sample student. "
            f"Estimated dropout probability is **{result['dropout_probability_percent']}%** "
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
            "I can help with: `predict`, `use default student`, "
            "`explain risk`, `show recommendations`."
        )

    st.session_state["chat"].append({"role": "assistant", "content": reply})
    st.rerun()

with st.sidebar:
    st.header("Model information")
    st.metric("Decision threshold", f"{predictor.threshold:.2f}")
    st.metric("Features used", len(predictor.feature_columns))
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
