"""
Eye + neck controlled mouse (version 3)

This version deliberately does not map raw iris pixels directly to the screen.
Head/neck pose is the main control signal; iris motion is a small fine-control
signal.  Press C while looking straight at the camera to calibrate your neutral
pose.  Press P to pause/resume and Q or ESC to exit.
"""

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


# Camera and filtering -------------------------------------------------------
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
MEDIAN_WINDOW = 7                 # removes individual bad landmark frames
EMA_ALPHA = 0.18                  # lower = steadier, higher = faster
MAX_FEATURE_JUMP = 0.12           # reject implausible one-frame changes
DEAD_ZONE = 0.018                 # ignore tiny movement around neutral

# Cursor response. Head/neck is intentionally much stronger than iris motion.
HEAD_GAIN_X, HEAD_GAIN_Y = 5.5, 5.0
IRIS_GAIN_X, IRIS_GAIN_Y = 1.4, 1.2
X_DIRECTION, Y_DIRECTION = 1, 1   # change either to -1 if an axis feels reversed

# Click and scroll -----------------------------------------------------------
BLINK_THRESHOLD = 0.19
WINK_HOLD_SECONDS = 0.14
CLICK_COOLDOWN = 0.70
SCROLL_COOLDOWN = 0.18
SCROLL_HEAD_THRESHOLD = 0.16      # neck pitch needed before scrolling
SCROLL_AMOUNT = 2

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]


def point(lms, index):
    """Return a MediaPipe landmark as a small float numpy vector."""
    return np.array([lms[index].x, lms[index].y], dtype=float)


def eye_aspect_ratio(lms, indices):
    pts = [point(lms, i) for i in indices]
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal < 1e-6:
        return 1.0
    return (np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])) / (2 * horizontal)


def iris_center(lms, indices):
    return np.mean([point(lms, i) for i in indices], axis=0)


def features(lms):
    """Return (neck_x, neck_y, iris_x, iris_y), all scale-independent.

    The nose position relative to the eye line and chin is a stable proxy for
    head/neck yaw and pitch.  Unlike absolute image coordinates, it remains
    usable if the user leans slightly nearer to or farther from the camera.
    """
    left_eye, right_eye = point(lms, 33), point(lms, 263)
    nose, chin = point(lms, 1), point(lms, 152)
    eye_mid = (left_eye + right_eye) / 2
    eye_width = max(np.linalg.norm(right_eye - left_eye), 1e-4)
    face_height = max(np.linalg.norm(chin - eye_mid), 1e-4)

    # Head / neck pose proxy.  These are intentionally the primary inputs.
    neck_x = (nose[0] - eye_mid[0]) / eye_width
    neck_y = (nose[1] - eye_mid[1]) / face_height

    # Iris position relative to each eye, then averaged.  This cancels most
    # camera translation and lets eyes make small corrections to head motion.
    li, ri = iris_center(lms, LEFT_IRIS), iris_center(lms, RIGHT_IRIS)
    left_top, left_bottom = point(lms, 159), point(lms, 145)
    right_top, right_bottom = point(lms, 386), point(lms, 374)
    left_w = max(abs(left_eye[0] - point(lms, 133)[0]), 1e-4)
    right_w = max(abs(point(lms, 362)[0] - right_eye[0]), 1e-4)
    left_h = max(abs(left_bottom[1] - left_top[1]), 1e-4)
    right_h = max(abs(right_bottom[1] - right_top[1]), 1e-4)
    iris_x = ((li[0] - (left_eye[0] + point(lms, 133)[0]) / 2) / left_w +
              (ri[0] - (point(lms, 362)[0] + right_eye[0]) / 2) / right_w) / 2
    iris_y = ((li[1] - (left_top[1] + left_bottom[1]) / 2) / left_h +
              (ri[1] - (right_top[1] + right_bottom[1]) / 2) / right_h) / 2
    return np.array([neck_x, neck_y, iris_x, iris_y], dtype=float)


def dead_zone(value):
    """Keep the cursor still near centre, without a sudden jump at the edge."""
    if abs(value) <= DEAD_ZONE:
        return 0.0
    return np.sign(value) * (abs(value) - DEAD_ZONE) / (1 - DEAD_ZONE)


def main():
    pyautogui.PAUSE = 0
    screen_w, screen_h = pyautogui.size()
    cursor = np.array([screen_w / 2, screen_h / 2], dtype=float)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.7, min_tracking_confidence=0.7,
    )
    history = deque(maxlen=MEDIAN_WINDOW)
    neutral = None
    calibration = []
    filtered = None
    last_raw = None
    paused = False
    left_started = right_started = None
    left_fired = right_fired = False
    last_click = last_scroll = 0.0
    status = "Press C, then look straight ahead for calibration"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                status = "Camera frame unavailable"
                break
            frame = cv2.flip(frame, 1)
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
                raw = features(lms)
                # Never let an isolated webcam/landmark glitch enter the filter.
                if last_raw is None or np.max(np.abs(raw - last_raw)) <= MAX_FEATURE_JUMP:
                    last_raw = raw
                    history.append(raw)
                if history:
                    median = np.median(np.asarray(history), axis=0)
                    filtered = median if filtered is None else EMA_ALPHA * median + (1 - EMA_ALPHA) * filtered

                left_closed = eye_aspect_ratio(lms, LEFT_EYE) < BLINK_THRESHOLD
                right_closed = eye_aspect_ratio(lms, RIGHT_EYE) < BLINK_THRESHOLD
                now = time.time()

                # Calibrate from many stable frames, not a single noisy sample.
                if calibration:
                    calibration.append(filtered.copy())
                    if len(calibration) >= 60:
                        neutral = np.median(np.asarray(calibration), axis=0)
                        calibration = []
                        status = "Calibrated: head/neck primary, iris fine control"
                    else:
                        status = f"Calibrating... {len(calibration)}/60"

                if neutral is not None and not paused and not calibration:
                    delta = filtered - neutral
                    move_x = dead_zone(delta[0]) * HEAD_GAIN_X + dead_zone(delta[2]) * IRIS_GAIN_X
                    move_y = dead_zone(delta[1]) * HEAD_GAIN_Y + dead_zone(delta[3]) * IRIS_GAIN_Y
                    target = np.array([
                        screen_w / 2 + X_DIRECTION * move_x * screen_w / 2,
                        screen_h / 2 + Y_DIRECTION * move_y * screen_h / 2,
                    ])
                    target = np.clip(target, [0, 0], [screen_w - 1, screen_h - 1])
                    cursor += 0.22 * (target - cursor)
                    pyautogui.moveTo(int(cursor[0]), int(cursor[1]), duration=0)

                    # Scrolling is based on deliberate neck pitch, not raw iris Y.
                    if abs(delta[1]) > SCROLL_HEAD_THRESHOLD and now - last_scroll > SCROLL_COOLDOWN:
                        pyautogui.scroll(SCROLL_AMOUNT if delta[1] < 0 else -SCROLL_AMOUNT)
                        last_scroll = now
                        status = "Scroll up" if delta[1] < 0 else "Scroll down"
                    elif not status.startswith("Scroll"):
                        status = "Tracking"

                # A held wink is required, preventing a one-frame grain spike click.
                if left_closed and not right_closed:
                    left_started = now if left_started is None else left_started
                    if not left_fired and now - left_started >= WINK_HOLD_SECONDS and now - last_click >= CLICK_COOLDOWN:
                        pyautogui.click(); last_click = now; left_fired = True; status = "Left click"
                else:
                    left_started = None; left_fired = False
                if right_closed and not left_closed:
                    right_started = now if right_started is None else right_started
                    if not right_fired and now - right_started >= WINK_HOLD_SECONDS and now - last_click >= CLICK_COOLDOWN:
                        pyautogui.rightClick(); last_click = now; right_fired = True; status = "Right click"
                else:
                    right_started = None; right_fired = False

                cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
                cv2.putText(frame, "C: calibrate  P: pause  Q/ESC: quit", (12, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                status = "Face not detected - cursor held"
                cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)

            cv2.imshow("Neck + Iris Mouse (V3)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                status = "Paused" if paused else "Tracking"
            if key == ord("c") and filtered is not None:
                calibration = [filtered.copy()]
                history.clear()
                status = "Hold your natural straight-ahead position"
    finally:
        cap.release()
        mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
