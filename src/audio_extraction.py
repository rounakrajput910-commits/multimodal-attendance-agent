import subprocess
import shutil
import cv2
import numpy as np
import soundfile as sf

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_VIDEO = PROJECT_DIR / "data" / "videos" / "classroom.mp4"
OUTPUT_AUDIO = PROJECT_DIR / "data" / "videos" / "classroom_audio.wav"


def find_ffmpeg():
    """Find FFmpeg in PATH or a standard Windows WinGet installation."""

    # First, check whether FFmpeg is available in the system PATH.
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path:
        return ffmpeg_path

    # If not found, check the usual Windows WinGet installation directory.
    winget_packages = (
        Path.home()
        / "AppData"
        / "Local"
        / "Microsoft"
        / "WinGet"
        / "Packages"
    )

    if winget_packages.exists():
        matches = list(
            winget_packages.glob(
                "Gyan.FFmpeg.Essentials_*/"
                "ffmpeg-*/bin/ffmpeg.exe"
            )
        )

        if matches:
            return str(matches[0])

    return None


def extract_audio():
    """Extract mono, 16 kHz audio from the input video."""

    if not INPUT_VIDEO.exists():
        print(f"Error: Video not found at {INPUT_VIDEO}")
        return

    ffmpeg_path = find_ffmpeg()

    if ffmpeg_path is None:
        print("Error: FFmpeg could not be found.")
        print("Install FFmpeg and make it accessible to the application.")
        return

    # Check whether OpenCV can open the input video.
    video = cv2.VideoCapture(str(INPUT_VIDEO))

    if not video.isOpened():
        print("Error: Could not open the input video.")
        return

    video.release()

    # Extract mono audio sampled at 16,000 Hz.
    command = [
        ffmpeg_path,
        "-y",
        "-i", str(INPUT_VIDEO),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(OUTPUT_AUDIO)
    ]

    try:
        subprocess.run(
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

    # Verify that the extracted audio can be read.
    try:
        audio, sample_rate = sf.read(
            str(OUTPUT_AUDIO),
            dtype="float32"
        )

    except Exception as error:
        print("Error: Could not read the extracted audio.")
        print(f"Details: {error}")
        return

    if audio.size == 0:
        print("Error: The extracted audio is empty.")
        return

    duration = audio.shape[0] / sample_rate
    rms = float(np.sqrt(np.mean(np.square(audio))))

    print("\nAudio extraction completed successfully!")
    print("----------------------------------------")
    print(f"FFmpeg location: {ffmpeg_path}")
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Number of audio samples: {audio.shape[0]}")
    print(f"Average signal level (RMS): {rms:.6f}")
    print(f"Extracted audio saved to: {OUTPUT_AUDIO}")


if __name__ == "__main__":
    extract_audio()