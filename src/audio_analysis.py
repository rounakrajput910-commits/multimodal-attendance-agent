import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_AUDIO = PROJECT_DIR / "data" / "videos" / "classroom_audio.wav"
OUTPUT_CSV = PROJECT_DIR / "data" / "attendance" / "audio_analysis.csv"

# Analysis settings
INTERVAL_SECONDS = 1.0
QUIET_THRESHOLD = 0.01


def analyze_audio():
    # Step 1: Check whether the audio file exists
    if not INPUT_AUDIO.exists():
        print(f"Error: Audio file not found at {INPUT_AUDIO}")
        print("Run audio_extraction.py first.")
        return

    # Step 2: Load the audio
    try:
        audio, sample_rate = sf.read(
            str(INPUT_AUDIO),
            dtype="float32"
        )
    except Exception as error:
        print(f"Error reading audio: {error}")
        return

    if audio.size == 0:
        print("Error: Audio file is empty.")
        return

    # Step 3: Convert stereo audio to mono if necessary
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Step 4: Calculate interval size
    samples_per_interval = int(sample_rate * INTERVAL_SECONDS)

    if samples_per_interval <= 0:
        print("Error: Invalid sample rate.")
        return

    # Step 5: Analyze each interval
    results = []

    for start in range(0, len(audio), samples_per_interval):
        end = min(start + samples_per_interval, len(audio))
        segment = audio[start:end]

        if len(segment) == 0:
            continue

        # Calculate RMS energy
        rms = float(np.sqrt(np.mean(np.square(segment))))

        # Classify based on signal energy
        if rms < QUIET_THRESHOLD:
            activity = "Quiet"
        else:
            activity = "Audio Activity"

        start_time = start / sample_rate
        end_time = end / sample_rate

        results.append({
            "start_time_seconds": round(start_time, 2),
            "end_time_seconds": round(end_time, 2),
            "rms_energy": round(rms, 6),
            "activity": activity
        })

    # Step 6: Save the results
    if not results:
        print("No audio intervals were analyzed.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    # Step 7: Display a summary
    quiet_count = int((df["activity"] == "Quiet").sum())
    active_count = int((df["activity"] == "Audio Activity").sum())

    print("\nAudio analysis completed!")
    print("--------------------------------")
    print(f"Audio duration: {len(audio) / sample_rate:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Intervals analyzed: {len(df)}")
    print(f"Quiet intervals: {quiet_count}")
    print(f"Audio activity intervals: {active_count}")
    print(f"Results saved to: {OUTPUT_CSV}")

    print("\nFirst five intervals:")
    print(df.head())


if __name__ == "__main__":
    analyze_audio()