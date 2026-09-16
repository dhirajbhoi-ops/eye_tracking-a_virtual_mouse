# 👁️ Eye Tracking Mouse

Control your mouse cursor using only your eyes — powered by real-time facial landmark detection. Move the cursor by looking around, click by winking, scroll by looking up or down, and select text with a long blink. No extra hardware required, just a webcam.

## Features

- **Gaze-based cursor control** — tracks iris position relative to your eyes and maps it to screen coordinates
- **Wink-to-click** — left-eye wink for left click, right-eye wink for right click
- **Blink-based text selection** — a long blink with both eyes toggles click-and-drag text selection
- **Scroll zones** — look near the top or bottom of the screen to scroll up/down
- **9-point calibration** — maps your natural gaze range to your actual screen for accurate tracking
- **Jitter-smoothing filter** — rolling average + spike rejection keeps the cursor stable even with natural iris-detection noise
- **Simple GUI** — start calibration, pause/resume tracking, and emergency-stop, all from a small control panel

## How it works

The app uses [MediaPipe Face Mesh](https://developers.google.com/mediapipe) with iris refinement to track 478 facial landmarks in real time from your webcam feed. It calculates the iris position relative to the eye corners to estimate gaze direction, smooths that signal to remove frame-to-frame jitter, and maps it to screen coordinates (using a calibration pass for accuracy). Blinks are detected using the Eye Aspect Ratio (EAR) technique to trigger clicks and other actions.

## Requirements

- Python 3.11 (MediaPipe's iris/solutions API isn't reliably available on 3.13+ yet)
- A webcam (built-in, USB, or even your phone via an app like Iriun/DroidCam)
- pre-install cv2, mediapipe 

## Installation

```bash
git clone https://github.com/dhirajbhoi-ops/eye_tracking-a_virtual_mouse.git
cd eye-tracking-mouse
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install opencv-python mediapipe pyautogui numpy
```

## Usage

```bash
python version_2.py
```

1. Click **Start 9-Point Calibration** and look at each highlighted point as it appears.
2. Once calibrated, click **Pause/Resume Tracking** to toggle control.
3. Use the controls below to interact with your screen.

```bash
python version_3.py
```

1. Click **Start 9-Point Calibration** and look at each highlighted point as it appears.
2. Once calibrated, click **Pause/Resume Tracking** to toggle control.
3. Use the controls below to interact with your screen.
4. Coordinates tracking

## Controls

| Action | Trigger |
|---|---|
| Move cursor | Look around |
| Left click | Wink left eye |
| Right click | Wink right eye |
| Start/stop text selection | Long blink (both eyes) |
| Scroll up | Look near top of screen |
| Scroll down | Look near bottom of screen |
| Emergency stop | `ESC` key |

## Project structure

- `version_1.py` — initial prototype: basic gaze tracking, blink-to-click, and scrolling
- `version_2.py` — full version: adds calibration, a Tkinter control GUI, wink-based left/right clicks, scrolling feature, text selection, text selection mode, and gaze smoothing 
- `version_3.py` — full version: adds calibration, a Tkinter control GUI, wink-based left/right clicks, text selection mode, and gaze smoothing for a stable cursor

## Known limitations

- Accuracy depends on lighting and camera quality — a well-lit, front-facing setup works best
- Recalibrate if you change your seating position or distance from the camera
- Currently single-monitor only
- the cursor might be unstable due to the webcam picture quality, but it demonstration ready

## License
dhirajbhoi-ops
