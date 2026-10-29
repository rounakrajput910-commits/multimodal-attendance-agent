import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_DIR / "data" / "attendance" / "fused_signals.csv"
)

OUTPUT_CSV = (
    PROJECT_DIR / "data" / "attendance" / "engagement_scores.csv"
)

# Scoring weights
VISUAL_WEIGHT = 0.5
AUDIO_WEIGHT = 0.5


def calculate_engagement_scores():
    # Step 1: Check input file
    if not INPUT_CSV.exists():
        print(f"Error: Input file not found: {INPUT_CSV}")
        print("Run signal_fusion.py first.")
        return

    # Step 2: Load fused signals
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as error:
        print(f"Error reading input CSV: {error}")
        return

    if df.empty:
        print("Error: No fused signal records found.")
        return

    required_columns = {
        "start_time_seconds",
        "end_time_seconds",
        "visual_presence_ratio",
        "rms_energy",
        "visual_presence",
        "audio_activity"
    }

    if not required_columns.issubset(df.columns):
        print("Error: Required columns are missing.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Step 3: Normalize visual presence
    # The ratio is already between 0 and 1.
    visual_score = (
        df["visual_presence_ratio"]
        .fillna(0)
        .clip(0, 1)
    )

    # Step 4: Convert audio activity into a simple indicator
    # This reflects the earlier energy threshold, not verified speech.
    audio_score = (
        df["audio_activity"]
        .eq("Audio Activity")
        .astype(float)
    )

    # Step 5: Calculate the experimental activity indicator
    total_weight = VISUAL_WEIGHT + AUDIO_WEIGHT

    if total_weight <= 0:
        print("Error: Scoring weights must have a positive sum.")
        return

    score = (
        (visual_score * VISUAL_WEIGHT)
        + (audio_score * AUDIO_WEIGHT)
    ) / total_weight

    df["visual_signal_score"] = (visual_score * 100).round(2)
    df["audio_activity_score"] = (audio_score * 100).round(2)

    df["activity_indicator"] = (score * 100).round(2)

    # Step 6: Add descriptive categories
    def classify_score(value):
        if value < 25:
            return "Low detected activity"
        elif value < 60:
            return "Moderate detected activity"
        else:
            return "High detected activity"

    df["activity_category"] = df["activity_indicator"].apply(
        classify_score
    )

    # Step 7: Save results
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # Step 8: Print summary
    print("\nActivity indicator calculation completed!")
    print("------------------------------------------")
    print(f"Intervals analyzed: {len(df)}")
    print(
        "Average activity indicator:",
        f"{df['activity_indicator'].mean():.2f}/100"
    )
    print(
        "Low detected activity intervals:",
        int((df["activity_category"] == "Low detected activity").sum())
    )
    print(
        "Moderate detected activity intervals:",
        int(
            (
                df["activity_category"]
                == "Moderate detected activity"
            ).sum()
        )
    )
    print(
        "High detected activity intervals:",
        int((df["activity_category"] == "High detected activity").sum())
    )

    print(f"\nResults saved to: {OUTPUT_CSV}")

    print("\nFirst five results:")
    print(
        df[
            [
                "start_time_seconds",
                "end_time_seconds",
                "visual_signal_score",
                "audio_activity_score",
                "activity_indicator",
                "activity_category"
            ]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    calculate_engagement_scores()