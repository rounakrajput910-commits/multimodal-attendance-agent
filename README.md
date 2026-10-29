# Multimodal Attendance & Engagement Agent

A Python-based prototype that combines computer vision and audio signal analysis to generate classroom session statistics, estimated attendance tracks, activity indicators, and rule-based teaching suggestions.

## Overview

Traditional attendance systems often rely on manual recording and provide limited information about classroom activity.

This project explores how video and audio signals can be processed together to summarize a recorded classroom session. It uses face detection to estimate visual presence, analyzes audio signal energy, combines these measurements into time-based observations, and generates session analytics.

The application runs locally on the user's computer.

## Features

- **Face Detection:** Detects visible faces in a recorded video using OpenCV's Haar Cascade detector.
- **Attendance Estimation:** Tracks detected face locations across frames and generates estimated attendance records.
- **Audio Extraction:** Extracts mono, 16 kHz audio from a video using FFmpeg.
- **Audio Activity Analysis:** Calculates RMS energy over time intervals to classify intervals as quiet or audio-active.
- **Multimodal Signal Fusion:** Combines visual presence and audio activity into a time-based dataset.
- **Activity Indicator:** Calculates an experimental indicator from visual and audio features.
- **Teaching Suggestions:** Generates rule-based suggestions using the computed signals.
- **Session Analytics:** Exports session statistics in JSON and CSV formats.
- **Streamlit Dashboard:** Displays charts, metrics, suggestions, and output tables in a local web interface.

## Technology Stack

- Python
- OpenCV
- NumPy
- Pandas
- SoundFile
- FFmpeg
- Streamlit

## Project Structure

```text
multimodal-attendance-agent/
├── dashboard/
│   └── app.py
├── data/
│   ├── videos/
│   └── attendance/
├── src/
│   ├── face_detection.py
│   ├── attendance.py
│   ├── audio_extraction.py
│   ├── audio_analysis.py
│   ├── signal_fusion.py
│   ├── engagement_scoring.py
│   ├── teaching_suggestions.py
│   ├── session_analytics.py
│   └── multimodal_agent.py
├── requirements.txt
├── .gitignore
└── README.md
```

Generated files and input recordings are stored in the local `data/` directory.

## Prerequisites

- Python 3.10 or another Python version supported by the installed dependencies.
- Git
- Visual Studio Code or another Python editor
- FFmpeg installed and accessible to the audio extraction module
- A compatible test video saved as `data/videos/classroom.mp4`

The actual Python version and dependency compatibility should be verified on the target computer.

## Installation

### 1. Clone the repository

After cloning this project from GitHub, open a terminal in the project directory.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd multimodal-attendance-agent
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your repository.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install FFmpeg

Install FFmpeg for your operating system and ensure the executable is accessible to the application.

The audio extraction module must be configured with a valid FFmpeg executable path.

### 5. Add a test video

Place a video you are authorized to process at:

```text
data/videos/classroom.mp4
```

The project does not include classroom recordings or generated attendance data in the public repository.

## Running the Application

From the project root, activate your virtual environment and run:

```bash
python src/multimodal_agent.py
```

The core application runs the processing modules sequentially and generates output files under `data/`.

## Launching the Dashboard

Run:

```bash
python -m streamlit run dashboard/app.py
```

Open the local address printed by Streamlit, usually:

```text
http://localhost:8501
```

The dashboard displays session metrics, visual and audio charts, activity indicators, teaching suggestions, and available output tables.

## Output Files

The application generates outputs such as:

| File | Description |
|---|---|
| `faces_detected.mp4` | Video with detected face bounding boxes |
| `classroom_audio.wav` | Extracted audio |
| `attendance.csv` | Estimated attendance-track records |
| `audio_analysis.csv` | Audio energy and activity by interval |
| `fused_signals.csv` | Combined visual and audio signals |
| `engagement_scores.csv` | Experimental activity indicators |
| `teaching_suggestions.csv` | Rule-based teaching suggestions |
| `session_analytics.json` | Detailed session analytics |
| `session_analytics.csv` | Tabular session analytics |
| `session_summary.json` | Summary of generated output files |

These files are generated locally and should not be committed if they contain recordings or identifiable information.

## Limitations

This project is an educational prototype, not a production-ready attendance or student assessment system.

- Face detection does not identify a person's identity.
- Face tracking can lose track of people or assign new track IDs.
- Attendance records are estimates and are not verified against a student roster.
- Audio RMS energy cannot reliably distinguish speech from noise.
- The activity indicator is experimental and has not been validated as a measure of attention, understanding, or learning.
- Teaching suggestions use predefined rules and require human interpretation.
- Detection accuracy depends on recording quality, lighting, camera placement, and other environmental conditions.

The output should not be used as the sole basis for grading, discipline, or other consequential decisions.

## Privacy and Responsible Use

- Process recordings only when you have appropriate permission.
- Prefer consented, anonymized, or synthetic test data.
- Keep recordings and attendance records local unless sharing is explicitly authorized.
- Do not publish identifiable classroom recordings or personal attendance information.
- Review generated recommendations before acting on them.

## Future Improvements

- Use more robust multi-object tracking.
- Add speech detection to distinguish speech-like activity from general sound energy.
- Improve attendance estimation and track continuity.
- Calibrate the activity indicator using clearly defined, ethically collected evaluation data.
- Add configuration options for input files, thresholds, and FFmpeg paths.
- Add automated tests for missing files, corrupted videos, silent audio, and other edge cases.
- Improve dashboard reporting and export options.

## Project Status

Educational prototype. The core processing pipeline and local Streamlit dashboard have been implemented and tested with a local video.

## Author

Add your name and academic details here before publishing.