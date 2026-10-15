import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import time
import math
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

SMOOTHING = 7

# How many recent frames of raw gaze to average.
# Higher = steadier cursor but slightly more lag.
GAZE_HISTORY_SIZE = 8

# If the raw gaze ratio jumps more than this in a single
# frame, it's almost certainly landmark noise (a blink,
# a brief detection glitch, etc), so we ignore that frame
# instead of letting the cursor snap.
MAX_GAZE_JUMP = 0.15

BLINK_THRESHOLD = 0.19

# How long an eye must remain closed to count as a wink
WINK_TIME = 0.12

# Long blink duration
LONG_BLINK_TIME = 0.45

# Cooldown between clicks
CLICK_COOLDOWN = 0.7

# Scroll zone
SCROLL_ZONE = 0.15

# Scroll speed
SCROLL_SPEED = 2


# ============================================================
# SCREEN
# ============================================================

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# EYE LANDMARKS
# ============================================================

LEFT_EYE = [
    33, 160, 158,
    133, 153, 144
]

RIGHT_EYE = [
    362, 385, 387,
    263, 373, 380
]

LEFT_IRIS = [
    468, 469, 470, 471
]

RIGHT_IRIS = [
    473, 474, 475, 476
]


# ============================================================
# GLOBAL VARIABLES
# ============================================================

running = True
tracking_enabled = True
calibrating = False

camera = None

# Calibration data
calibration_points = []
calibration_samples = []

calibration_index = 0

# Cursor
previous_x = SCREEN_WIDTH // 2
previous_y = SCREEN_HEIGHT // 2

# Rolling buffer of recent raw gaze readings, used to
# smooth out iris-detection jitter before it ever reaches
# the calibration mapping.
gaze_x_history = deque(maxlen=GAZE_HISTORY_SIZE)
gaze_y_history = deque(maxlen=GAZE_HISTORY_SIZE)

# Last accepted (post-smoothing) gaze reading, used to
# reject single-frame spikes.
last_gaze_x = None
last_gaze_y = None

# Blink state
left_eye_closed = False
right_eye_closed = False

left_eye_start = 0
right_eye_start = 0

both_eye_start = 0

last_click = 0

# Text selection
selection_mode = False

# Status
current_status = "Initializing..."


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def distance(p1, p2):
    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


def get_point(landmark, width, height):
    return (
        int(landmark.x * width),
        int(landmark.y * height)
    )


def eye_aspect_ratio(landmarks, indices, width, height):

    points = [
        get_point(landmarks[i], width, height)
        for i in indices
    ]

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])

    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0

    return (
        vertical1 + vertical2
    ) / (2 * horizontal)


def get_iris_center(landmarks, indices):

    x = np.mean(
        [landmarks[i].x for i in indices]
    )

    y = np.mean(
        [landmarks[i].y for i in indices]
    )

    return x, y


# ============================================================
# GAZE FEATURE
# ============================================================

def calculate_gaze(landmarks):

    """
    Calculates normalized gaze coordinates.

    Instead of using absolute camera coordinates,
    we calculate iris position relative to the eye.
    """

    # Left eye corners
    left_inner = landmarks[133]
    left_outer = landmarks[33]

    # Right eye corners
    right_inner = landmarks[362]
    right_outer = landmarks[263]

    # Eye vertical landmarks
    left_top = landmarks[159]
    left_bottom = landmarks[145]

    right_top = landmarks[386]
    right_bottom = landmarks[374]

    # Iris
    left_iris_x, left_iris_y = get_iris_center(
        landmarks,
        LEFT_IRIS
    )

    right_iris_x, right_iris_y = get_iris_center(
        landmarks,
        RIGHT_IRIS
    )

    # -----------------------------
    # Horizontal position
    # -----------------------------

    left_width = left_outer.x - left_inner.x

    right_width = right_inner.x - right_outer.x

    if abs(left_width) < 0.001:
        left_width = 0.001

    if abs(right_width) < 0.001:
        right_width = 0.001

    left_ratio = (
        left_iris_x - left_inner.x
    ) / left_width

    right_ratio = (
        right_iris_x - right_outer.x
    ) / right_width

    horizontal = (
        left_ratio + right_ratio
    ) / 2

    # -----------------------------
    # Vertical position
    # -----------------------------

    left_height = left_bottom.y - left_top.y
    right_height = right_bottom.y - right_top.y

    if abs(left_height) < 0.001:
        left_height = 0.001

    if abs(right_height) < 0.001:
        right_height = 0.001

    left_vertical = (
        left_iris_y - left_top.y
    ) / left_height

    right_vertical = (
        right_iris_y - right_top.y
    ) / right_height

    vertical = (
        left_vertical + right_vertical
    ) / 2

    return horizontal, vertical


# ============================================================
# SMOOTH GAZE (reduce iris-jitter noise)
# ============================================================

def smooth_gaze(gaze_x, gaze_y):
    """
    Reduces frame-to-frame jitter in the raw gaze reading
    before it gets mapped to screen coordinates.

    Two things happen here:
      1. Spike rejection - if this frame's raw reading is
         wildly different from the last accepted one, it's
         probably noise (blink, momentary bad detection),
         so we substitute the last good value instead of
         letting it through.
      2. Rolling average - we then average the last several
         frames together, which cancels out the small
         continuous wobble in iris landmark detection.
    """

    global last_gaze_x, last_gaze_y

    if last_gaze_x is not None:

        if abs(gaze_x - last_gaze_x) > MAX_GAZE_JUMP:
            gaze_x = last_gaze_x

        if abs(gaze_y - last_gaze_y) > MAX_GAZE_JUMP:
            gaze_y = last_gaze_y

    last_gaze_x = gaze_x
    last_gaze_y = gaze_y

    gaze_x_history.append(gaze_x)
    gaze_y_history.append(gaze_y)

    smoothed_x = sum(gaze_x_history) / len(gaze_x_history)
    smoothed_y = sum(gaze_y_history) / len(gaze_y_history)

    return smoothed_x, smoothed_y


# ============================================================
# MAP GAZE TO SCREEN
# ============================================================

def map_gaze(gaze_x, gaze_y):

    if len(calibration_samples) < 4:

        # Default mapping
        screen_x = gaze_x * SCREEN_WIDTH
        screen_y = gaze_y * SCREEN_HEIGHT

    else:

        gaze_points = np.array(
            [sample[0] for sample in calibration_samples]
        )

        screen_points = np.array(
            [sample[1] for sample in calibration_samples]
        )

        min_x = np.min(gaze_points[:, 0])
        max_x = np.max(gaze_points[:, 0])

        min_y = np.min(gaze_points[:, 1])
        max_y = np.max(gaze_points[:, 1])

        if max_x - min_x < 0.001:
            max_x = min_x + 0.001

        if max_y - min_y < 0.001:
            max_y = min_y + 0.001

        screen_x = np.interp(
            gaze_x,
            [min_x, max_x],
            [0, SCREEN_WIDTH]
        )

        screen_y = np.interp(
            gaze_y,
            [min_y, max_y],
            [0, SCREEN_HEIGHT]
        )

    screen_x = np.clip(
        screen_x,
        0,
        SCREEN_WIDTH - 1
    )

    screen_y = np.clip(
        screen_y,
        0,
        SCREEN_HEIGHT - 1
    )

    return screen_x, screen_y


# ============================================================
# CLICK FUNCTIONS
# ============================================================

def left_click():

    global last_click

    if time.time() - last_click < CLICK_COOLDOWN:
        return

    pyautogui.click()
    last_click = time.time()


def right_click():

    global last_click

    if time.time() - last_click < CLICK_COOLDOWN:
        return

    pyautogui.rightClick()
    last_click = time.time()


# ============================================================
# TEXT SELECTION
# ============================================================

def start_selection():

    global selection_mode

    if not selection_mode:

        pyautogui.mouseDown()

        selection_mode = True

        update_status(
            "TEXT SELECTION: ON"
        )


def stop_selection():

    global selection_mode

    if selection_mode:

        pyautogui.mouseUp()

        selection_mode = False

        update_status(
            "TEXT SELECTION: OFF"
        )


# ============================================================
# CALIBRATION
# ============================================================

def create_calibration_points():

    margin_x = SCREEN_WIDTH * 0.10
    margin_y = SCREEN_HEIGHT * 0.10

    return [

        (margin_x, margin_y),

        (SCREEN_WIDTH / 2, margin_y),

        (SCREEN_WIDTH - margin_x, margin_y),

        (margin_x, SCREEN_HEIGHT / 2),

        (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2),

        (SCREEN_WIDTH - margin_x, SCREEN_HEIGHT / 2),

        (margin_x, SCREEN_HEIGHT - margin_y),

        (SCREEN_WIDTH / 2, SCREEN_HEIGHT - margin_y),

        (
            SCREEN_WIDTH - margin_x,
            SCREEN_HEIGHT - margin_y
        )
    ]


def start_calibration():

    global calibrating
    global calibration_points
    global calibration_samples
    global calibration_index

    if calibrating:
        return

    calibrating = True

    calibration_points = create_calibration_points()

    calibration_samples = []

    calibration_index = 0

    update_status(
        "Calibration started"
    )


def collect_calibration_sample(gaze):

    global calibration_index
    global calibrating

    if not calibrating:
        return

    if calibration_index >= len(calibration_points):

        calibrating = False

        update_status(
            "Calibration complete"
        )

        return

    target = calibration_points[
        calibration_index
    ]

    # Store gaze and corresponding screen position
    calibration_samples.append(
        (
            gaze,
            target
        )
    )

    calibration_index += 1

    if calibration_index >= len(
        calibration_points):

        calibrating = False

        update_status(
            "Calibration complete"
        )


# ============================================================
# STATUS
# ============================================================

def update_status(text):

    global current_status

    current_status = text

    try:
        status_label.config(
            text=text
        )
    except:
        pass


# ============================================================
# CAMERA PROCESSING
# ============================================================

def camera_loop():

    global camera
    global previous_x
    global previous_y

    global left_eye_closed
    global right_eye_closed

    global left_eye_start
    global right_eye_start

    global both_eye_start

    camera = cv2.VideoCapture(0)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    while running:

        ret, frame = camera.read()

        if not ret:
            update_status(
                "Camera error"
            )
            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(
            rgb
        )

        if results.multi_face_landmarks:

            landmarks = (
                results.multi_face_landmarks[0]
                .landmark
            )

            # ==========================================
            # EAR
            # ==========================================

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

            left_closed = (
                left_ear < BLINK_THRESHOLD
            )

            right_closed = (
                right_ear < BLINK_THRESHOLD
            )

            now = time.time()

            # ==========================================
            # LEFT EYE WINK
            # ==========================================

            if left_closed and not right_closed:

                if not left_eye_closed:

                    left_eye_start = now

                elif (
                    now - left_eye_start
                    > WINK_TIME
                ):

                    left_click()

                    left_eye_closed = True

                    update_status(
                        "LEFT CLICK"
                    )

            else:

                left_eye_closed = False

            # ==========================================
            # RIGHT EYE WINK
            # ==========================================

            if right_closed and not left_closed:

                if not right_eye_closed:

                    right_eye_start = now

                elif (
                    now - right_eye_start
                    > WINK_TIME
                ):

                    right_click()

                    right_eye_closed = True

                    update_status(
                        "RIGHT CLICK"
                    )

            else:

                right_eye_closed = False

            # ==========================================
            # BOTH EYES
            # ==========================================

            if left_closed and right_closed:

                if both_eye_start == 0:

                    both_eye_start = now

                duration = (
                    now - both_eye_start
                )

                # Long blink
                if duration > LONG_BLINK_TIME:

                    # Start selection
                    if not selection_mode:

                        start_selection()

                    # Prevent repeatedly triggering
                    both_eye_start = now + 100

            else:

                # If both eyes were closed
                # and selection is active,
                # another long blink can be used
                # after the next long closure.

                if both_eye_start != 0:

                    both_eye_start = 0

            # ==========================================
            # GAZE
            # ==========================================

            gaze_x, gaze_y = calculate_gaze(
                landmarks
            )

            gaze_x, gaze_y = smooth_gaze(
                gaze_x,
                gaze_y
            )

            # ==========================================
            # CALIBRATION
            # ==========================================

            if calibrating:

                target = calibration_points[
                    calibration_index
                ]

                # Show calibration target
                cv2.putText(
                    frame,
                    "Look at the calibration point",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # Collect a few frames for stability
                collect_calibration_sample(
                    (gaze_x, gaze_y)
                )

            # ==========================================
            # NORMAL TRACKING
            # ==========================================

            elif tracking_enabled:

                target_x, target_y = map_gaze(
                    gaze_x,
                    gaze_y
                )

                # Smooth cursor
                cursor_x = (
                    previous_x
                    + (
                        target_x - previous_x
                    ) / SMOOTHING
                )

                cursor_y = (
                    previous_y
                    + (
                        target_y - previous_y
                    ) / SMOOTHING
                )

                previous_x = cursor_x
                previous_y = cursor_y

                pyautogui.moveTo(
                    int(cursor_x),
                    int(cursor_y),
                    duration=0
                )

                # ======================================
                # SCROLL
                # ======================================

                if gaze_y < SCROLL_ZONE:

                    pyautogui.scroll(
                        SCROLL_SPEED
                    )

                    update_status(
                        "SCROLL UP"
                    )

                elif gaze_y > (
                    1 - SCROLL_ZONE
                ):

                    pyautogui.scroll(
                        -SCROLL_SPEED
                    )

                    update_status(
                        "SCROLL DOWN"
                    )

                else:

                    if not selection_mode:

                        update_status(
                            "Tracking"
                        )

            # ==========================================
            # DRAW INFORMATION
            # ==========================================

            cv2.putText(
                frame,
                f"Left EAR: {left_ear:.2f}",
                (10, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right EAR: {right_ear:.2f}",
                (10, 455),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            if selection_mode:

                cv2.putText(
                    frame,
                    "SELECTING TEXT",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        else:

            update_status(
                "Face not detected"
            )

        # ==========================================
        # CALIBRATION WINDOW
        # ==========================================

        if calibrating:

            if calibration_index < len(
                calibration_points
            ):

                target = calibration_points[
                    calibration_index
                ]

                # Convert screen target to camera
                # display coordinates
                tx = int(
                    (
                        target[0]
                        / SCREEN_WIDTH
                    )
                    * CAMERA_WIDTH
                )

                ty = int(
                    (
                        target[1]
                        / SCREEN_HEIGHT
                    )
                    * CAMERA_HEIGHT
                )

                cv2.circle(
                    frame,
                    (tx, ty),
                    15,
                    (0, 255, 255),
                    -1
                )

        cv2.imshow(
            "Eye Tracking Camera",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            stop_program()

    if camera:
        camera.release()

    cv2.destroyAllWindows()


# ============================================================
# GUI FUNCTIONS
# ============================================================

def toggle_tracking():

    global tracking_enabled

    tracking_enabled = not tracking_enabled

    if tracking_enabled:

        tracking_button.config(
            text="Pause Tracking"
        )

        update_status(
            "Tracking enabled"
        )

    else:

        tracking_button.config(
            text="Resume Tracking"
        )

        update_status(
            "Tracking paused"
        )


def stop_program():

    global running

    running = False

    try:

        if selection_mode:

            pyautogui.mouseUp()

    except:
        pass

    try:
        root.destroy()
    except:
        pass


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Eye Tracking Mouse V2"
)

root.geometry(
    "500x600"
)

root.resizable(
    False,
    False
)


# Title

title = ttk.Label(
    root,
    text="Eye Tracking Mouse V2",
    font=("Arial", 22, "bold")
)

title.pack(
    pady=20
)


subtitle = ttk.Label(
    root,
    text="Control your computer using your eyes",
    font=("Arial", 11)
)

subtitle.pack(
    pady=5
)


# Status

status_label = ttk.Label(
    root,
    text="Initializing...",
    font=("Arial", 13)
)

status_label.pack(
    pady=20
)


# Buttons

calibration_button = ttk.Button(
    root,
    text="Start 9-Point Calibration",
    command=start_calibration
)

calibration_button.pack(
    pady=10,
    ipadx=20,
    ipady=8
)


tracking_button = ttk.Button(
    root,
    text="Pause Tracking",
    command=toggle_tracking
)

tracking_button.pack(
    pady=10,
    ipadx=20,
    ipady=8
)


stop_button = ttk.Button(
    root,
    text="EXIT",
    command=stop_program
)

stop_button.pack(
    pady=20,
    ipadx=40,
    ipady=8
)


# Instructions

instructions = """
CONTROLS

Look around          → Move cursor

Left-eye wink        → Left click

Right-eye wink       → Right click

Long both-eye blink  → Start text selection

During selection     → Look to move mouse

Long both-eye blink  → Release selection

Look at top          → Scroll up

Look at bottom       → Scroll down

ESC                  → Emergency stop
"""

instruction_label = ttk.Label(
    root,
    text=instructions,
    font=("Arial", 10),
    justify="left"
)

instruction_label.pack(
    pady=10
)


# ============================================================
# START CAMERA THREAD
# ============================================================

camera_thread = threading.Thread(
    target=camera_loop,
    daemon=True
)

camera_thread.start()


# ============================================================
# CLOSE WINDOW
# ============================================================

root.protocol(
    "WM_DELETE_WINDOW",
    stop_program
)

root.mainloop()