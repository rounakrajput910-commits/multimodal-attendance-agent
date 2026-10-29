import cv2
import csv
from pathlib import Path
from math import hypot

# Find the main project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Input video
INPUT_VIDEO = PROJECT_DIR / "data" / "videos" / "classroom.mp4"

# Attendance report
OUTPUT_CSV = PROJECT_DIR / "data" / "attendance" / "attendance.csv"

# Minimum observed duration to mark as present
MINIMUM_SECONDS = 2

# Maximum distance for matching a face between frames
MAX_DISTANCE = 100

# Number of frames a track can be missed before it expires
MAX_MISSING_FRAMES = 10


def generate_attendance():

    # Check whether the video exists
    if not INPUT_VIDEO.exists():
        print("Error: Input video not found.")
        return

    # Load the face detector
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        print("Error: Could not load face detector.")
        return

    # Open the video
    video = cv2.VideoCapture(str(INPUT_VIDEO))

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("Error: Invalid video frame rate.")
        video.release()
        return

    # Store temporary face tracks
    tracks = {}

    next_track_id = 1
    frame_number = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        frame_number += 1

        # Convert frame to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Find the center of each detected face
        detections = []

        for (x, y, w, h) in faces:

            center_x = x + w // 2
            center_y = y + h // 2

            detections.append(
                (center_x, center_y)
            )

        # Mark existing tracks as potentially missed
        for track in tracks.values():
            track["missing"] += 1

        # Calculate distances between existing tracks
        # and newly detected faces
        possible_matches = []

        for track_id, track in tracks.items():

            if track["missing"] > MAX_MISSING_FRAMES:
                continue

            for detection_index, center in enumerate(detections):

                distance = hypot(
                    track["x"] - center[0],
                    track["y"] - center[1]
                )

                if distance <= MAX_DISTANCE:

                    possible_matches.append(
                        (
                            distance,
                            track_id,
                            detection_index
                        )
                    )

        # Match the closest pairs first
        possible_matches.sort()

        matched_tracks = set()
        matched_detections = set()

        for distance, track_id, detection_index in possible_matches:

            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            center_x, center_y = detections[detection_index]

            tracks[track_id]["x"] = center_x
            tracks[track_id]["y"] = center_y

            tracks[track_id]["frames_seen"] += 1
            tracks[track_id]["last_frame"] = frame_number
            tracks[track_id]["missing"] = 0

            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        # Create a new track for every unmatched detection
        for detection_index, center in enumerate(detections):

            if detection_index in matched_detections:
                continue

            track_id = next_track_id
            next_track_id += 1

            tracks[track_id] = {
                "x": center[0],
                "y": center[1],
                "frames_seen": 1,
                "first_frame": frame_number,
                "last_frame": frame_number,
                "missing": 0
            }

    video.release()

    # Create the output folder if it doesn't exist
    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save attendance results
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Track_ID",
            "Frames_Detected",
            "Observed_Duration_Seconds",
            "First_Seen_Seconds",
            "Last_Seen_Seconds",
            "Attendance_Status"
        ])

        for track_id, track in tracks.items():

            duration = track["frames_seen"] / fps

            first_seen = track["first_frame"] / fps
            last_seen = track["last_frame"] / fps

            status = (
                "Present"
                if duration >= MINIMUM_SECONDS
                else "Below Threshold"
            )

            writer.writerow([
                f"Person_{track_id}",
                track["frames_seen"],
                round(duration, 2),
                round(first_seen, 2),
                round(last_seen, 2),
                status
            ])

    print("\nAttendance processing completed!")
    print(f"Frames processed: {frame_number}")
    print(f"Temporary tracks created: {len(tracks)}")
    print(f"Attendance report saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_attendance()