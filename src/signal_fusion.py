import cv2
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = PROJECT_DIR / "data" / "videos" / "classroom.mp4"

AUDIO_ANALYSIS_CSV = (
    PROJECT_DIR / "data" / "attendance" / "audio_analysis.csv"
)

OUTPUT_CSV = (
    PROJECT_DIR / "data" / "attendance" / "fused_signals.csv"
)

# Analyze the signals in one-second intervals
INTERVAL_SECONDS = 1.0


def fuse_signals():
    # Step 1: Check that the required files exist
    if not INPUT_VIDEO.exists():
        print(f"Error: Video not found: {INPUT_VIDEO}")
        return

    if not AUDIO_ANALYSIS_CSV.exists():
        print(f"Error: Audio analysis file not found: {AUDIO_ANALYSIS_CSV}")
        print("Run audio_analysis.py first.")
        return

    # Step 2: Load the audio analysis results
    try:
        audio_df = pd.read_csv(AUDIO_ANALYSIS_CSV)
    except Exception as error:
        print(f"Error reading audio analysis CSV: {error}")
        return

    required_columns = {
        "start_time_seconds",
        "end_time_seconds",
        "rms_energy",
        "activity"
    }

    if not required_columns.issubset(audio_df.columns):
        print("Error: Audio analysis CSV has unexpected columns.")
        print(f"Available columns: {list(audio_df.columns)}")
        return

    if audio_df.empty:
        print("Error: Audio analysis CSV contains no records.")
        return

    # Step 3: Create a face detector
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        print("Error: Could not load the face detector.")
        return

    # Step 4: Open the classroom video
    video = cv2.VideoCapture(str(INPUT_VIDEO))

    if not video.isOpened():
        print("Error: Could not open the input video.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("Error: Invalid video frame rate.")
        video.release()
        return

    # Store visual observations for each second
    visual_data = {}

    frame_number = 0

    print("Analyzing video and collecting visual signals...")

    # Step 5: Process the video frame by frame
    while True:
        success, frame = video.read()

        if not success:
            break

        timestamp = frame_number / fps
        interval_index = int(timestamp // INTERVAL_SECONDS)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_count = len(faces)

        if interval_index not in visual_data:
            visual_data[interval_index] = {
                "max_faces_detected": 0,
                "frames_processed": 0,
                "frames_with_faces": 0
            }

        visual_data[interval_index]["max_faces_detected"] = max(
            visual_data[interval_index]["max_faces_detected"],
            face_count
        )

        visual_data[interval_index]["frames_processed"] += 1

        if face_count > 0:
            visual_data[interval_index]["frames_with_faces"] += 1

        frame_number += 1

    video.release()

    if not visual_data:
        print("Error: No video frames were processed.")
        return

    # Step 6: Combine visual and audio observations
    fused_results = []

    for _, audio_row in audio_df.iterrows():

        start_time = float(audio_row["start_time_seconds"])
        end_time = float(audio_row["end_time_seconds"])

        interval_index = int(start_time // INTERVAL_SECONDS)

        visual = visual_data.get(
            interval_index,
            {
                "max_faces_detected": 0,
                "frames_processed": 0,
                "frames_with_faces": 0
            }
        )

        max_faces = visual["max_faces_detected"]
        frames_processed = visual["frames_processed"]
        frames_with_faces = visual["frames_with_faces"]

        # Estimate how frequently faces were visible in this interval
        if frames_processed > 0:
            visual_presence_ratio = (
                frames_with_faces / frames_processed
            )
        else:
            visual_presence_ratio = 0.0

        audio_activity = str(audio_row["activity"])
        rms_energy = float(audio_row["rms_energy"])

        visual_presence = (
            "Detected" if max_faces > 0 else "Not detected"
        )

        audio_is_active = audio_activity == "Audio Activity"

        # Combine observations without claiming actual engagement
        if max_faces > 0 and audio_is_active:
            combined_observation = "Visual and audio activity"

        elif max_faces > 0:
            combined_observation = "Visual presence only"

        elif audio_is_active:
            combined_observation = "Audio activity only"

        else:
            combined_observation = "Low detected activity"

        fused_results.append({
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "max_faces_detected": max_faces,
            "visual_presence_ratio": round(
                visual_presence_ratio, 3
            ),
            "visual_presence": visual_presence,
            "rms_energy": rms_energy,
            "audio_activity": audio_activity,
            "combined_observation": combined_observation
        })

    # Step 7: Save the fused results
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fused_df = pd.DataFrame(fused_results)
    fused_df.to_csv(OUTPUT_CSV, index=False)

    # Step 8: Display a summary
    print("\nMultimodal signal fusion completed!")
    print("------------------------------------")
    print(f"Video frames processed: {frame_number}")
    print(f"Intervals analyzed: {len(fused_df)}")
    print(
        "Intervals with visual and audio activity:",
        int(
            (
                fused_df["combined_observation"]
                == "Visual and audio activity"
            ).sum()
        )
    )
    print(f"Output saved to: {OUTPUT_CSV}")

    print("\nFirst five fused observations:")
    print(fused_df.head().to_string(index=False))


if __name__ == "__main__":
    fuse_signals()