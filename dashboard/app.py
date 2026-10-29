import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Locate project directories.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "attendance"

ANALYTICS_FILE = DATA_DIR / "session_analytics.json"
ATTENDANCE_FILE = DATA_DIR / "attendance.csv"
FUSED_FILE = DATA_DIR / "fused_signals.csv"
SCORES_FILE = DATA_DIR / "engagement_scores.csv"
SUGGESTIONS_FILE = DATA_DIR / "teaching_suggestions.csv"

st.set_page_config(
    page_title="Multimodal Attendance Agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Multimodal Attendance & Engagement Agent")

st.caption(
    "A prototype for analyzing visual presence, audio activity, "
    "attendance-track estimates, and session-level patterns."
)


@st.cache_data
def load_json(file_path):
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data
def load_csv(file_path):
    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def create_demo_data():
    """Create synthetic data for a public demonstration."""

    intervals = list(range(10))

    fused = pd.DataFrame({
        "start_time_seconds": intervals,
        "end_time_seconds": [i + 1 for i in intervals],
        "max_faces_detected": [3, 4, 4, 3, 2, 3, 4, 4, 3, 4],
        "visual_presence_ratio": [
            0.80, 0.90, 0.85, 0.75, 0.50,
            0.70, 0.90, 0.85, 0.65, 0.90
        ],
        "rms_energy": [
            0.025, 0.030, 0.018, 0.008, 0.006,
            0.020, 0.035, 0.028, 0.012, 0.032
        ],
        "audio_activity": [
            "Audio Activity", "Audio Activity",
            "Audio Activity", "Quiet", "Quiet",
            "Audio Activity", "Audio Activity",
            "Audio Activity", "Audio Activity",
            "Audio Activity"
        ],
    })

    fused["visual_presence"] = fused["visual_presence_ratio"].apply(
        lambda value: "Detected" if value > 0 else "Not detected"
    )

    fused["combined_observation"] = fused.apply(
        lambda row: (
            "Visual presence and audio activity"
            if row["visual_presence_ratio"] > 0
            and row["audio_activity"] == "Audio Activity"
            else "Review interval"
        ),
        axis=1,
    )

    scores = fused[
        ["start_time_seconds", "visual_presence_ratio", "audio_activity"]
    ].copy()

    scores["visual_score"] = scores["visual_presence_ratio"] * 100

    scores["audio_score"] = scores["audio_activity"].apply(
        lambda value: 100 if value == "Audio Activity" else 0
    )

    scores["activity_indicator"] = (
        0.5 * scores["visual_score"]
        + 0.5 * scores["audio_score"]
    ).round(2)

    scores["activity_category"] = scores["activity_indicator"].apply(
        lambda value: (
            "Low detected activity"
            if value < 25
            else "Moderate detected activity"
            if value < 60
            else "High detected activity"
        )
    )

    attendance = pd.DataFrame({
        "track_id": [1, 2, 3, 4],
        "frames_seen": [240, 220, 180, 150],
        "duration_seconds": [8.0, 7.3, 6.0, 5.0],
        "attendance_status": [
            "Present", "Present", "Present", "Present"
        ],
    })

    suggestions = scores[
        ["start_time_seconds", "visual_score", "audio_score"]
    ].copy()

    def make_suggestion(row):
        if row["visual_score"] < 30:
            return (
                "Review camera placement, lighting, and face visibility."
            )
        if row["audio_score"] >= 100:
            return (
                "Review whether the audio represents speech, discussion, "
                "or background noise."
            )
        if row["visual_score"] >= 70 and row["audio_score"] == 0:
            return (
                "Check whether this is a quiet lesson segment or an "
                "audio capture issue."
            )
        return "Review classroom conditions and lesson context."

    suggestions["teaching_suggestion"] = suggestions.apply(
        make_suggestion,
        axis=1,
    )

    analytics = {
        "session_information": {
            "video_duration_seconds": 10.0
        },
        "attendance_statistics": {
            "attendance_track_records": 4
        },
        "activity_indicator_statistics": {
            "average_activity_indicator": round(
                float(scores["activity_indicator"].mean()), 2
            )
        },
    }

    return analytics, attendance, fused, scores, suggestions


# Load actual output files.
analytics = load_json(ANALYTICS_FILE)
attendance = load_csv(ATTENDANCE_FILE)
fused = load_csv(FUSED_FILE)
scores = load_csv(SCORES_FILE)
suggestions = load_csv(SUGGESTIONS_FILE)

# Use synthetic demo data only if none of the main output files exist.
output_files = [
    ANALYTICS_FILE,
    ATTENDANCE_FILE,
    FUSED_FILE,
    SCORES_FILE,
    SUGGESTIONS_FILE,
]

demo_mode = not any(file_path.exists() for file_path in output_files)

if demo_mode:
    (
        analytics,
        attendance,
        fused,
        scores,
        suggestions,
    ) = create_demo_data()

    st.info(
        "DEMO MODE: The information displayed below is synthetic sample "
        "data created for demonstration. It is not real classroom data."
    )
else:
    st.success("Showing available local processing results.")

# 1. Session overview
st.header("1. Session Overview")

if analytics is None:
    st.warning("Session analytics are unavailable.")
else:
    session_info = analytics.get("session_information", {})
    attendance_info = analytics.get("attendance_statistics", {})
    activity_info = analytics.get("activity_indicator_statistics", {})

    duration = session_info.get("video_duration_seconds")
    track_records = attendance_info.get("attendance_track_records")
    average_activity = activity_info.get("average_activity_indicator")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Video Duration",
        f"{duration:.2f} seconds"
        if isinstance(duration, (int, float))
        else "Unavailable",
    )

    col2.metric(
        "Attendance Track Records",
        track_records if track_records is not None else "Unavailable",
    )

    col3.metric(
        "Average Activity Indicator",
        f"{average_activity:.2f}/100"
        if isinstance(average_activity, (int, float))
        else "Unavailable",
    )

    st.info(
        "Track records are not verified student identities. The activity "
        "indicator is experimental and is not a validated measure of "
        "attention, understanding, or learning."
    )

# 2. Visual and audio analysis
st.header("2. Visual and Audio Analysis")

if fused is None or fused.empty:
    st.warning("Fused signals are unavailable.")
else:
    if "start_time_seconds" in fused.columns:
        fused = fused.sort_values("start_time_seconds")

    if {
        "start_time_seconds",
        "visual_presence_ratio",
    }.issubset(fused.columns):
        st.subheader("Visual Presence Ratio Over Time")

        visual_chart = fused[
            ["start_time_seconds", "visual_presence_ratio"]
        ].dropna().set_index("start_time_seconds")

        st.line_chart(visual_chart)

        st.caption(
            "The ratio represents the fraction of processed frames in an "
            "interval where at least one face was detected. It does not "
            "represent the percentage of students present."
        )

    if {"start_time_seconds", "rms_energy"}.issubset(fused.columns):
        st.subheader("Audio Signal Energy Over Time")

        audio_chart = fused[
            ["start_time_seconds", "rms_energy"]
        ].dropna().set_index("start_time_seconds")

        st.line_chart(audio_chart)

        st.caption(
            "RMS energy measures signal strength. It cannot, by itself, "
            "distinguish speech from noise or silence."
        )

    if "audio_activity" in fused.columns:
        st.subheader("Audio Activity Summary")

        audio_counts = (
            fused["audio_activity"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Activity")
            .reset_index(name="Intervals")
        )

        st.bar_chart(audio_counts.set_index("Activity"))

# 3. Activity indicator analysis
st.header("3. Activity Indicator Analysis")

if scores is None or scores.empty:
    st.warning("Activity scores are unavailable.")
else:
    if "activity_category" in scores.columns:
        category_counts = (
            scores["activity_category"]
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Category")
            .reset_index(name="Intervals")
        )

        st.subheader("Activity Category Distribution")
        st.bar_chart(category_counts.set_index("Category"))

    score_columns = [
        column
        for column in [
            "start_time_seconds",
            "visual_score",
            "audio_score",
            "activity_indicator",
            "activity_category",
        ]
        if column in scores.columns
    ]

    if score_columns:
        st.subheader("Activity Score Records")

        st.dataframe(
            scores[score_columns],
            use_container_width=True,
            hide_index=True,
        )

# 4. Teaching suggestions
st.header("4. Teaching Suggestions")

if suggestions is None or suggestions.empty:
    st.info("No teaching suggestions are available.")
elif "teaching_suggestion" in suggestions.columns:
    unique_suggestions = (
        suggestions["teaching_suggestion"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(unique_suggestions) == 0:
        st.info("No suggestions were generated.")
    else:
        for number, suggestion in enumerate(unique_suggestions, start=1):
            st.markdown(f"**{number}.** {suggestion}")

    st.caption(
        "Suggestions are generated by predefined rules. Review the lesson "
        "context before acting on them."
    )

# 5. Detailed output tables
st.header("5. Detailed Output Tables")

table_options = {
    "Attendance Records": attendance,
    "Fused Signals": fused,
    "Activity Scores": scores,
    "Teaching Suggestions": suggestions,
}

available_tables = {
    name: dataframe
    for name, dataframe in table_options.items()
    if dataframe is not None
}

if available_tables:
    selected_table = st.selectbox(
        "Choose a dataset to inspect",
        list(available_tables.keys()),
    )

    st.dataframe(
        available_tables[selected_table],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("No output tables are available.")

# Footer
st.divider()

if demo_mode:
    st.caption(
        "This public demo uses synthetic data. No real classroom recordings "
        "or student attendance records are included in the demo."
    )
else:
    st.caption(
        "Use only recordings you are authorized to process. Avoid uploading "
        "identifiable classroom videos or attendance records to public "
        "repositories."
    )