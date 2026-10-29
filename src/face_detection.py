import cv2
from pathlib import Path

# Locate the project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Input and output video paths
INPUT_VIDEO = PROJECT_DIR / "data" / "videos" / "classroom.mp4"
OUTPUT_VIDEO = PROJECT_DIR / "data" / "videos" / "faces_detected.mp4"


def detect_faces():
    # Check whether the input video exists
    if not INPUT_VIDEO.exists():
        print(f"Error: Video not found at {INPUT_VIDEO}")
        return

    # Load OpenCV's pre-trained face detector
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        print("Error: Could not load the face detector.")
        return

    # Open the input video
    video = cv2.VideoCapture(str(INPUT_VIDEO))

    if not video.isOpened():
        print("Error: Could not open the input video.")
        return

    # Read video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        print("Error: Invalid video properties.")
        video.release()
        return

    # Prepare the output video
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not output.isOpened():
        print("Error: Could not create the output video.")
        video.release()
        return

    frame_number = 0
    total_faces_detected = 0
    max_faces_in_frame = 0

    while True:
        # Read one frame
        success, frame = video.read()

        if not success:
            break

        frame_number += 1

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_count = len(faces)
        total_faces_detected += face_count
        max_faces_in_frame = max(max_faces_in_frame, face_count)

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        # Display the number of faces in this frame
        cv2.putText(
            frame,
            f"Faces detected: {face_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Save the processed frame
        output.write(frame)

    video.release()
    output.release()

    print("\nFace detection completed!")
    print(f"Frames processed: {frame_number}")
    print(f"Total face detections across frames: {total_faces_detected}")
    print(f"Maximum faces detected in one frame: {max_faces_in_frame}")
    print(f"Processed video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    detect_faces()