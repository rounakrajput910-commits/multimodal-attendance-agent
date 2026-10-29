import json
import pandas as pd
import cv2

from pathlib import Path
from datetime import datetime


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

VIDEO_FILE = PROJECT_DIR / "data" / "videos" / "classroom.mp4"

ATTENDANCE_FILE = (
    PROJECT_DIR / "data" / "attendance" / "attendance.csv"
)

FUSED_FILE = (
    PROJECT_DIR / "data" / "attendance" / "fused_signals.csv"
)

SCORES_FILE = (
    PROJECT_DIR / "data" / "attendance" / "engagement_scores.csv"
)

SUGGESTIONS_FILE = (
    PROJECT_DIR / "data" / "attendance" / "teaching_suggestions.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "attendance"

JSON_REPORT = OUTPUT_DIR / "session_analytics.json"
CSV_REPORT = OUTPUT_DIR / "session_analytics.csv"


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def load_csv(file_path):
    """Load a CSV file and return its DataFrame."""

    if not file_path.exists():
        print(f"Warning: File not found: {file_path.name}")
        return None

    try:
        return pd.read_csv(file_path)
    except Exception as error:
        print(f"Warning: Could not read {file_path.name}: {error}")
        return None


def get_video_duration():
    """Return video duration in seconds, or None if unavailable."""

    if not VIDEO_FILE.exists():
        return None

    video = cv2.VideoCapture(str(VIDEO_FILE))

    if not video.isOpened():
        return None

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    video.release()

    if fps <= 0 or frame_count < 0:
        return None

    return round(frame_count / fps, 2)


def safe_mean(dataframe, column):
    """Calculate a column mean when valid data is available."""

    if dataframe is None or column not in dataframe.columns:
        return None

    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()

    if values.empty:
        return None

    return round(float(values.mean()), 2)


def safe_sum(dataframe, column):
    """Calculate a numeric column sum when valid data is available."""

    if dataframe is None or column not in dataframe.columns:
        return None

    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()

    if values.empty:
        return None

    return int(values.sum())


# --------------------------------------------------
# MAIN ANALYTICS FUNCTION
# --------------------------------------------------

def generate_analytics():

    print("\nGenerating session analytics...")

    # Load available output files
    attendance_df = load_csv(ATTENDANCE_FILE)
    fused_df = load_csv(FUSED_FILE)
    scores_df = load_csv(SCORES_FILE)
    suggestions_df = load_csv(SUGGESTIONS_FILE)

    # Calculate video duration
    duration = get_video_duration()

    # Attendance statistics
    attendance_records = (
        len(attendance_df)
        if attendance_df is not None
        else None
    )

    # Count records meeting the existing attendance threshold,
    # if the expected column exists.
    qualifying_records = None

    if attendance_df is not None:
        if "attendance_status" in attendance_df.columns:
            qualifying_records = int(
                attendance_df["attendance_status"]
                .astype(str)
                .str.lower()
                .isin(["present", "true", "1"])
                .sum()
            )
        elif "status" in attendance_df.columns:
            qualifying_records = int(
                attendance_df["status"]
                .astype(str)
                .str.lower()
                .isin(["present", "true", "1"])
                .sum()
            )

    # Visual and audio statistics
    average_visual_presence = safe_mean(
        fused_df,
        "visual_presence_ratio"
    )

    average_audio_energy = safe_mean(
        fused_df,
        "rms_energy"
    )

    # Experimental activity indicator statistics
    average_activity_indicator = safe_mean(
        scores_df,
        "activity_indicator"
    )

    category_distribution = {}

    if (
        scores_df is not None
        and "activity_category" in scores_df.columns
    ):
        category_distribution = {
            str(category): int(count)
            for category, count in
            scores_df["activity_category"]
            .value_counts(dropna=False)
            .items()
        }

    # Audio activity statistics
    active_audio_intervals = None
    total_audio_intervals = None

    if (
        fused_df is not None
        and "audio_activity" in fused_df.columns
    ):
        total_audio_intervals = len(fused_df)

        active_audio_intervals = int(
            (
                fused_df["audio_activity"].astype(str)
                == "Audio Activity"
            ).sum()
        )

    # Collect unique suggestion texts
    suggestions = []

    if (
        suggestions_df is not None
        and "teaching_suggestion" in suggestions_df.columns
    ):
        suggestions = [
            str(suggestion)
            for suggestion in suggestions_df["teaching_suggestion"]
            .dropna()
            .unique()
        ]

    # Build the final report
    report = {
        "project": "Multimodal Attendance and Engagement Agent",
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "session_information": {
            "input_video": str(
                VIDEO_FILE.relative_to(PROJECT_DIR)
            ),
            "video_duration_seconds": duration
        },
        "attendance_statistics": {
            "attendance_track_records": attendance_records,
            "records_meeting_existing_threshold": qualifying_records,
            "note": (
                "Track records are estimates and may not represent "
                "unique or verified student identities."
            )
        },
        "visual_and_audio_statistics": {
            "average_visual_presence_ratio": average_visual_presence,
            "average_audio_energy_rms": average_audio_energy,
            "audio_activity_intervals": active_audio_intervals,
            "total_audio_intervals": total_audio_intervals
        },
        "activity_indicator_statistics": {
            "average_activity_indicator": average_activity_indicator,
            "category_distribution": category_distribution,
            "note": (
                "This is an experimental signal-based activity indicator, "
                "not a validated measure of attention or learning."
            )
        },
        "teaching_suggestions": suggestions
    }

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    with open(JSON_REPORT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    # Save a readable two-column CSV summary
    csv_rows = []

    def flatten(data, prefix=""):
        for key, value in data.items():
            current_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flatten(value, current_key)
            elif isinstance(value, list):
                csv_rows.append({
                    "metric": current_key,
                    "value": " | ".join(map(str, value))
                })
            else:
                csv_rows.append({
                    "metric": current_key,
                    "value": value
                })

    flatten(report)

    pd.DataFrame(csv_rows).to_csv(CSV_REPORT, index=False)

    # Display summary
    print("\nSession analytics generated successfully!")
    print("-----------------------------------------")
    print(f"Video duration: {duration} seconds")
    print(f"Attendance track records: {attendance_records}")
    print(f"Average visual presence ratio: {average_visual_presence}")
    print(f"Average audio energy: {average_audio_energy}")
    print(f"Average activity indicator: {average_activity_indicator}")
    print(f"\nJSON report: {JSON_REPORT}")
    print(f"CSV report: {CSV_REPORT}")


if __name__ == "__main__":
    generate_analytics()