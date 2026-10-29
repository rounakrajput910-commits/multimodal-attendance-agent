import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_DIR / "data" / "attendance" / "engagement_scores.csv"
)

OUTPUT_CSV = (
    PROJECT_DIR / "data" / "attendance" / "teaching_suggestions.csv"
)


def generate_suggestion(row):
    """
    Generate a teaching suggestion based on observable signals.

    These rules identify patterns worth checking.
    They do not establish student attention or understanding.
    """

    visual_score = row["visual_signal_score"]
    audio_score = row["audio_activity_score"]

    # Rule 1: Few frames contain a detected face.
    if visual_score < 30:
        return (
            "Check camera placement, lighting, and visibility. "
            "Participants may be outside the camera view."
        )

    # Rule 2: Audio activity is detected.
    if audio_score >= 100:
        return (
            "Audio activity was detected. Check whether it reflects "
            "class discussion, the instructor's speech, or background noise."
        )

    # Rule 3: Faces are visible but audio activity is low.
    if visual_score >= 70 and audio_score == 0:
        return (
            "Faces were frequently visible, but little audio activity "
            "was detected. Check whether this was a quiet lesson segment "
            "or an audio capture issue."
        )

    # Rule 4: Intermediate visual presence.
    if visual_score < 70:
        return (
            "Review the video quality and classroom layout. "
            "Some faces may not be visible in every frame."
        )

    # Rule 5: Default recommendation.
    return (
        "Continue monitoring the session and review the lesson context "
        "before deciding whether any teaching change is needed."
    )


def main():
    # Step 1: Check the input file.
    if not INPUT_CSV.exists():
        print(f"Error: Input file not found: {INPUT_CSV}")
        print("Run engagement_scoring.py first.")
        return

    # Step 2: Load activity scores.
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as error:
        print(f"Error reading activity scores: {error}")
        return

    if df.empty:
        print("Error: No activity records found.")
        return

    required_columns = {
        "start_time_seconds",
        "end_time_seconds",
        "visual_signal_score",
        "audio_activity_score",
        "activity_indicator",
        "activity_category"
    }

    if not required_columns.issubset(df.columns):
        print("Error: Required columns are missing.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Step 3: Generate suggestions for each interval.
    df["teaching_suggestion"] = df.apply(
        generate_suggestion,
        axis=1
    )

    # Step 4: Save the results.
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # Step 5: Print a summary.
    print("\nTeaching suggestion generation completed!")
    print("-----------------------------------------")
    print(f"Intervals analyzed: {len(df)}")
    print(f"Suggestions generated: {df['teaching_suggestion'].notna().sum()}")
    print(f"Output saved to: {OUTPUT_CSV}")

    print("\nSample teaching suggestions:\n")

    for _, row in df.head(5).iterrows():
        start = row["start_time_seconds"]
        end = row["end_time_seconds"]

        print(f"Time: {start:.1f} - {end:.1f} seconds")
        print(f"Activity indicator: {row['activity_indicator']}/100")
        print(f"Suggestion: {row['teaching_suggestion']}")
        print("-" * 50)


if __name__ == "__main__":
    main()