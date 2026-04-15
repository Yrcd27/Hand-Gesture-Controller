import cv2
import mediapipe as mp
import time
import numpy as np
import ctypes
import urllib.request
import urllib.parse
import ssl
import os
from pynput.mouse import Button, Controller as MouseController
from gestures import fingers_up, handle_gesture, GestureDebouncer, pinch_distance

# ── Screen dimensions ────────────────────────────────────
mouse    = MouseController()
screen_w = ctypes.windll.user32.GetSystemMetrics(0)
screen_h = ctypes.windll.user32.GetSystemMetrics(1)

# ── Cursor smoothing (EMA) ───────────────────────────────
SMOOTH_ALPHA   = 0.18   # lower = smoother but more lag, higher = more responsive
CURSOR_DEAD_ZONE = 4    # pixels — ignore micro-jitter below this threshold
prev_x, prev_y = 0, 0

# ── Click state ──────────────────────────────────────────
left_clicking  = False
right_clicking = False
CLICK_THRESH   = 0.045
CLICK_COOLDOWN = 0.4    # seconds between clicks to prevent accidental double-clicks
last_left_click_t  = 0
last_right_click_t = 0

# ── Scroll state (velocity-based) ───────────────────────
SCROLL_SENSITIVITY = 18   # multiplier for per-frame wrist velocity
SCROLL_DEAD_ZONE   = 0.004 # ignore tiny hand tremors
SCROLL_SMOOTHING   = 0.3   # EMA alpha for scroll velocity
scroll_vel         = 0.0   # smoothed scroll velocity
prev_wrist_y       = None  # previous frame wrist y for velocity calc
last_scroll_t      = 0
SCROLL_COOLDOWN    = 0.016  # ~60fps cap on scroll events

# ── MediaPipe setup ──────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    parsed = urllib.parse.urlparse(MODEL_URL)
    if parsed.scheme != "https":
        raise ValueError(f"Refusing to download from non-HTTPS URL: {MODEL_URL}")
    print("Downloading hand landmark model (~9MB)...")
    ssl_ctx = ssl.create_default_context()
    with urllib.request.urlopen(MODEL_URL, context=ssl_ctx) as response, \
         open(MODEL_PATH, "wb") as out_file:
        out_file.write(response.read())
    print("Done.")


def draw_ui(frame, state, action_label, mode_label, label_time, scroll_vel=None):
    h, w = frame.shape[:2]

    # Finger indicator boxes
    names = ["T", "I", "M", "R", "P"]
    for i, (n, up) in enumerate(zip(names, state)):
        color = (0, 220, 100) if up else (60, 60, 60)
        cv2.rectangle(frame, (10 + i * 46, 10), (50 + i * 46, 42), color, -1)
        cv2.putText(frame, n, (22 + i * 46, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Mode label
    if mode_label:
        cv2.putText(frame, mode_label, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

    # Scroll velocity bar on right side
    if scroll_vel is not None:
        bar_x = w - 30
        bar_top, bar_bot = 80, h - 80
        cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 16, bar_bot), (40, 40, 40), -1)
        mid_y = (bar_top + bar_bot) // 2
        # Clamp indicator within bar
        indicator_y = int(mid_y - np.clip(scroll_vel * 300, -(mid_y - bar_top), mid_y - bar_top))
        color = (0, 200, 255) if scroll_vel > 0 else (255, 100, 0)
        cv2.rectangle(frame, (bar_x, mid_y), (bar_x + 16, indicator_y), color, -1)
        arrow = "^" if scroll_vel > 0 else "v"
        label_y = indicator_y - 8 if scroll_vel > 0 else indicator_y + 18
        cv2.putText(frame, arrow, (bar_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Action label at bottom (shown for 1.5s)
    if time.time() - label_time < 1.5:
        cv2.putText(frame, action_label, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 2)


def run():
    global prev_x, prev_y, left_clicking, right_clicking
    global last_left_click_t, last_right_click_t
    global scroll_vel, prev_wrist_y, last_scroll_t

    ensure_model()

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    db         = GestureDebouncer()
    last_label = ""
    label_time = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Try index 1.")
        return

    # Lock camera to 640x480 @ 30fps for consistent performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Gesture Controller running. ESC = quit.")
    print("─────────────────────────────────────────")
    print("  Index only          → Move cursor")
    print("  Pinch (index+thumb) → Left click")
    print("  Index+mid + pinch   → Right click")
    print("  Peace sign + move   → Scroll up/down")
    print("  Fist                → Play / Pause")
    print("  Open hand           → Next track")
    print("  Thumb + pinky       → Prev track")
    print("  Mid+ring+pinky      → Volume up")
    print("  Ring+pinky          → Volume down")
    print("─────────────────────────────────────────")

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            mode_label      = ""
            scroll_vis      = None
            hand_detected   = bool(result.hand_landmarks)

            if hand_detected:
                lm         = result.hand_landmarks[0]
                handedness = result.handedness[0][0].display_name
                state      = fingers_up(lm, handedness)

                # Draw landmark dots
                for l in lm:
                    cx, cy = int(l.x * w), int(l.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 220, 100), -1)

                wrist_y = lm[0].y

                # ── SCROLL MODE: peace sign (index+middle only) ──────────
                if state == [False, True, True, False, False]:
                    mode_label = "SCROLL MODE  (move hand up/down)"

                    if prev_wrist_y is not None:
                        # Per-frame velocity: positive = hand moved up = scroll up
                        raw_vel = (prev_wrist_y - wrist_y)

                        # Dead zone: suppress micro-jitter
                        if abs(raw_vel) < SCROLL_DEAD_ZONE:
                            raw_vel = 0.0

                        # EMA smooth the velocity to remove spikes
                        scroll_vel = scroll_vel + SCROLL_SMOOTHING * (raw_vel - scroll_vel)
                    else:
                        scroll_vel = 0.0

                    prev_wrist_y = wrist_y
                    scroll_vis   = scroll_vel

                    now = time.time()
                    if abs(scroll_vel) > 0.001 and now - last_scroll_t > SCROLL_COOLDOWN:
                        amount = int(scroll_vel * SCROLL_SENSITIVITY)
                        if amount != 0:
                            mouse.scroll(0, amount)
                            last_scroll_t = now
                            mode_label = "SCROLLING UP" if scroll_vel > 0 else "SCROLLING DOWN"

                else:
                    # Reset scroll state when gesture changes
                    prev_wrist_y = None
                    scroll_vel   = 0.0

                # ── CURSOR MODE: index finger only ───────────────────────
                if state[1] and not state[2] and not state[3] and not state[4]:
                    thumb_pinch = pinch_distance(lm, 4, 8)

                    if thumb_pinch < CLICK_THRESH:
                        now = time.time()
                        if not left_clicking and now - last_left_click_t > CLICK_COOLDOWN:
                            mouse.press(Button.left)
                            mouse.release(Button.left)
                            left_clicking     = True
                            last_left_click_t = now
                            last_label        = "Left Click"
                            label_time        = now
                        mode_label = "LEFT CLICK"
                    else:
                        left_clicking = False
                        mode_label    = "CURSOR"

                        tip   = lm[8]
                        # Map index fingertip from camera FOV to screen coords
                        raw_x = np.interp(tip.x, [0.1, 0.9], [0, screen_w])
                        raw_y = np.interp(tip.y, [0.1, 0.9], [0, screen_h])

                        # EMA smoothing
                        sx = prev_x + SMOOTH_ALPHA * (raw_x - prev_x)
                        sy = prev_y + SMOOTH_ALPHA * (raw_y - prev_y)

                        # Dead zone: don't move cursor for sub-pixel jitter
                        if abs(sx - prev_x) > CURSOR_DEAD_ZONE or abs(sy - prev_y) > CURSOR_DEAD_ZONE:
                            prev_x, prev_y = sx, sy
                            mouse.position = (int(sx), int(sy))

                        fx, fy = int(tip.x * w), int(tip.y * h)
                        cv2.circle(frame, (fx, fy), 14, (0, 200, 255), 2)
                        # Draw a crosshair at fingertip for precision feedback
                        cv2.line(frame, (fx - 10, fy), (fx + 10, fy), (0, 200, 255), 1)
                        cv2.line(frame, (fx, fy - 10), (fx, fy + 10), (0, 200, 255), 1)

                # ── RIGHT CLICK: index+middle + thumb pinch ──────────────
                elif state[1] and state[2] and not state[3] and not state[4] and state[0]:
                    dist = pinch_distance(lm, 4, 8)
                    mode_label = "RIGHT CLICK READY"
                    if dist < CLICK_THRESH:
                        now = time.time()
                        if not right_clicking and now - last_right_click_t > CLICK_COOLDOWN:
                            mouse.press(Button.right)
                            mouse.release(Button.right)
                            right_clicking     = True
                            last_right_click_t = now
                            last_label         = "Right Click"
                            label_time         = now
                        mode_label = "RIGHT CLICK"
                    else:
                        right_clicking = False

                # ── MEDIA / VOLUME GESTURES ──────────────────────────────
                elif state != [False, True, True, False, False]:
                    left_clicking  = False
                    right_clicking = False
                    if db.should_fire(state):
                        lbl = handle_gesture(state)
                        if lbl:
                            last_label = lbl
                            label_time = time.time()

                draw_ui(frame, state, last_label, mode_label, label_time, scroll_vis)

            else:
                # No hand: decay scroll velocity naturally
                scroll_vel   *= 0.7
                prev_wrist_y  = None

            cv2.imshow("Gesture Controller  |  ESC = quit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    run()
