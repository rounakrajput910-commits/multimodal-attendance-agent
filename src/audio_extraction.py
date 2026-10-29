import subprocess
import cv2
import numpy as np
import soundfile as sf
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = PROJECT_DIR / "data" / "videos" / "classroom.mp4"
OUTPUT_AUDIO = PROJECT_DIR / "data" / "videos" / "classroom_audio.wav"

# FFmpeg executable path on your computer
FFMPEG_PATH = (
    r"C:\Users\princ\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
)


def extract_audio():
    # Step 1: Check whether the input video exists
    if not INPUT_VIDEO.exists():
        print(f"Error: Video not found at {INPUT_VIDEO}")
        return

    # Step 2: Check whether FFmpeg exists
    if not Path(FFMPEG_PATH).is_file():
        print("Error: FFmpeg executable was not found.")
        print(f"Expected location: {FFMPEG_PATH}")
        return

    # Step 3: Check whether OpenCV can open the video
    video = cv2.VideoCapture(str(INPUT_VIDEO))

    if not video.isOpened():
        print("Error: Could not open the input video.")
        return

    video.release()

    # Step 4: Prepare the FFmpeg command
    command = [
        FFMPEG_PATH,
        "-y",
        "-i", str(INPUT_VIDEO),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(OUTPUT_AUDIO)
    ]

    # Step 5: Extract audio
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

    except subprocess.CalledProcessError as error:
        print("Error: Audio extraction failed.")
        print(error.stderr[-3000:])
        return

    except OSError as error:
        print("Error: Could not start FFmpeg.")
        print(f"Details: {error}")
        return

    # Step 6: Read the extracted WAV file
    try:
        audio, sample_rate = sf.read(
            str(OUTPUT_AUDIO),
            dtype="float32"
        )

    except Exception as error:
        print("Error: Could not read the extracted audio.")
        print(f"Details: {error}")
        return

    # Step 7: Check whether audio samples exist
    if audio.size == 0:
        print("Error: The extracted audio is empty.")
        return

    # Step 8: Calculate basic audio statistics
    duration = audio.shape[0] / sample_rate
    rms = float(np.sqrt(np.mean(np.square(audio))))

    # Step 9: Display results
    print("\nAudio extraction completed successfully!")
    print("----------------------------------------")
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Number of audio samples: {audio.shape[0]}")
    print(f"Average signal level (RMS): {rms:.6f}")
    print(f"Extracted audio saved to: {OUTPUT_AUDIO}")


if __name__ == "__main__":
    extract_audio()