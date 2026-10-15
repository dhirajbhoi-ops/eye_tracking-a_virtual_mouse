import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# -----------------------------
# SETTINGS
# -----------------------------

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Cursor sensitivity
CURSOR_SMOOTHING = 7

# Blink detection
BLINK_THRESHOLD = 0.20
BLINK_COOLDOWN = 0.8

# Scroll sensitivity
SCROLL_THRESHOLD = 0.08

# -----------------------------
# INITIALIZE
# -----------------------------

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

screen_width, screen_height = pyautogui.size()

# Previous cursor position
prev_x = screen_width // 2
prev_y = screen_height // 2

last_blink_time = 0

# -----------------------------
# LANDMARKS
# -----------------------------

# Left eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Iris
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# -----------------------------
# FUNCTIONS
# -----------------------------

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, width, height):

    points = []

    for index in eye_indices:
        x = int(landmarks[index].x * width)
        y = int(landmarks[index].y * height)
        points.append((x, y))

    # Vertical distances
    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])

    # Horizontal distance
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2 * horizontal)


def get_iris_position(landmarks, iris_indices, width, height):

    points = []

    for index in iris_indices:
        x = landmarks[index].x * width
        y = landmarks[index].y * height

        points.append((x, y))

    return np.mean(points, axis=0)


# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # -----------------------------
        # BLINK DETECTION
        # -----------------------------

        left_ear = eye_aspect_ratio(
            landmarks,
            LEFT_EYE,
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        )

        right_ear = eye_aspect_ratio(
            landmarks,
            RIGHT_EYE,
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        )

        average_ear = (left_ear + right_ear) / 2

        current_time = time.time()

        # Blink detected
        if average_ear < BLINK_THRESHOLD:

            if current_time - last_blink_time > BLINK_COOLDOWN:

                pyautogui.click()

                last_blink_time = current_time

                cv2.putText(
                    frame,
                    "BLINK - CLICK",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        # -----------------------------
        # IRIS POSITION
        # -----------------------------

        left_iris = get_iris_position(
            landmarks,
            LEFT_IRIS,
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        )

        right_iris = get_iris_position(
            landmarks,
            RIGHT_IRIS,
            CAMERA_WIDTH,
            CAMERA_HEIGHT
        )

        # Average both eyes
        iris_x = (left_iris[0] + right_iris[0]) / 2
        iris_y = (left_iris[1] + right_iris[1]) / 2

        # -----------------------------
        # FACE POSITION
        # -----------------------------

        left_eye = landmarks[33]
        right_eye = landmarks[263]

        eye_center_x = (
            left_eye.x + right_eye.x
        ) / 2 * CAMERA_WIDTH

        eye_center_y = (
            left_eye.y + right_eye.y
        ) / 2 * CAMERA_HEIGHT

        # Relative iris position
        relative_x = iris_x - eye_center_x
        relative_y = iris_y - eye_center_y

        # Normalize
        normalized_x = relative_x / 80
        normalized_y = relative_y / 50

        # -----------------------------
        # CURSOR MOVEMENT
        # -----------------------------

        target_x = screen_width / 2 + normalized_x * screen_width / 2
        target_y = screen_height / 2 + normalized_y * screen_height / 2

        # Limit cursor
        target_x = np.clip(target_x, 0, screen_width - 1)
        target_y = np.clip(target_y, 0, screen_height - 1)

        # Smooth movement
        new_x = prev_x + (target_x - prev_x) / CURSOR_SMOOTHING
        new_y = prev_y + (target_y - prev_y) / CURSOR_SMOOTHING

        pyautogui.moveTo(
            int(new_x),
            int(new_y),
            duration=0
        )

        prev_x = new_x
        prev_y = new_y

        # -----------------------------
        # SCROLLING
        # -----------------------------

        if normalized_y < -SCROLL_THRESHOLD:

            pyautogui.scroll(2)

            cv2.putText(
                frame,
                "SCROLL UP",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

        elif normalized_y > SCROLL_THRESHOLD:

            pyautogui.scroll(-2)

            cv2.putText(
                frame,
                "SCROLL DOWN",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

        # -----------------------------
        # DRAW IRIS
        # -----------------------------

        cv2.circle(
            frame,
            (int(left_iris[0]), int(left_iris[1])),
            5,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (int(right_iris[0]), int(right_iris[1])),
            5,
            (0, 255, 0),
            -1
        )

        # Display EAR
        cv2.putText(
            frame,
            f"EAR: {average_ear:.2f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # -----------------------------
    # DISPLAY
    # -----------------------------

    cv2.imshow("Eye Tracking Mouse", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()