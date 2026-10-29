import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = PROJECT_DIR / "data" / "videos" / "classroom.mp4"
OUTPUT_DIR = PROJECT_DIR / "data" / "attendance"

MODULES = [
    "face_detection.py",
    "attendance.py",
    "audio_extraction.py",
    "audio_analysis.py",
    "signal_fusion.py",
    "engagement_scoring.py",
    "teaching_suggestions.py",
    "session_analytics.py",
]

OUTPUT_FILES = {
    "face_detection_video": PROJECT_DIR / "data" / "videos" / "faces_detected.mp4",
    "extracted_audio": PROJECT_DIR / "data" / "videos" / "classroom_audio.wav",
    "attendance_records": OUTPUT_DIR / "attendance.csv",
    "audio_analysis": OUTPUT_DIR / "audio_analysis.csv",
    "fused_signals": OUTPUT_DIR / "fused_signals.csv",
    "engagement_scores": OUTPUT_DIR / "engagement_scores.csv",
    "teaching_suggestions": OUTPUT_DIR / "teaching_suggestions.csv",
    "analytics_json": OUTPUT_DIR / "session_analytics.json",
    "analytics_csv": OUTPUT_DIR / "session_analytics.csv",
}


def run_module(module_name):
    """Run one Python module and stop if it fails."""

    module_path = PROJECT_DIR / "src" / module_name

    if not module_path.exists():
        print(f"\nERROR: Module not found: {module_path}")
        return False

    print("\n" + "=" * 60)
    print(f"Running: {module_name}")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, str(module_path)],
            cwd=str(PROJECT_DIR),
            check=False
        )
    except OSError as error:
        print(f"ERROR: Could not start {module_name}.")
        print(f"Details: {error}")
        return False

    if result.returncode != 0:
        print(f"\nERROR: {module_name} failed.")
        print("Fix this module's error before continuing.")
        return False

    print(f"\nSUCCESS: {module_name} completed.")
    return True


def create_session_summary():
    """Record which project output files were generated."""

    results = {}

    for name, file_path in OUTPUT_FILES.items():
        results[name] = {
            "exists": file_path.exists(),
            "path": str(file_path.relative_to(PROJECT_DIR))
        }

    summary = {
        "project": "Multimodal Attendance and Engagement Agent",
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "input_video": str(INPUT_VIDEO.relative_to(PROJECT_DIR)),
        "outputs": results
    }

    summary_path = OUTPUT_DIR / "session_summary.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("FINAL OUTPUT SUMMARY")
    print("=" * 60)

    for name, details in results.items():
        status = "AVAILABLE" if details["exists"] else "MISSING"
        print(f"{name}: {status}")

    print(f"\nSummary saved to: {summary_path}")

    missing_files = [
        name for name, details in results.items()
        if not details["exists"]
    ]

    if missing_files:
        print("\nWARNING: Some expected output files are missing.")
        for name in missing_files:
            print(f"- {name}")
    else:
        print("\nAll expected output files are available!")


def main():
    print("\nMULTIMODAL ATTENDANCE AND ENGAGEMENT AGENT")
    print("=" * 60)

    if not INPUT_VIDEO.exists():
        print(f"ERROR: Input video not found:\n{INPUT_VIDEO}")
        print("Place your test video at data/videos/classroom.mp4.")
        return

    for module_name in MODULES:
        success = run_module(module_name)

        if not success:
            print("\nApplication stopped because a module failed.")
            return

    create_session_summary()

    print("\n" + "=" * 60)
    print("CORE APPLICATION RUN COMPLETED")
    print("=" * 60)
    print("Your processing pipeline has finished.")


if __name__ == "__main__":
    main()