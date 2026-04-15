import pyautogui
import time
from collections import deque
from pynput.keyboard import Key, Controller
from pynput.mouse import Button, Controller as MouseController

keyboard = Controller()
mouse    = MouseController()

pyautogui.FAILSAFE = False


def vol_up():
    keyboard.press(Key.media_volume_up)
    keyboard.release(Key.media_volume_up)

def vol_down():
    keyboard.press(Key.media_volume_down)
    keyboard.release(Key.media_volume_down)


def fingers_up(hand_landmarks, handedness="Right"):
    if not hand_landmarks or len(hand_landmarks) < 21:
        return [False] * 5
    lm = hand_landmarks
    thumb = lm[4].x < lm[3].x if handedness == "Right" else lm[4].x > lm[3].x
    # Compare fingertip to MCP joint (tip-3) for robustness against hand tilt
    others = [lm[tip].y < lm[tip - 3].y for tip in [8, 12, 16, 20]]
    return [thumb] + others


def pinch_distance(lm, a, b):
    # 3D distance including z-depth for accuracy at angles
    return ((lm[a].x - lm[b].x)**2 + (lm[a].y - lm[b].y)**2 + (lm[a].z - lm[b].z)**2) ** 0.5


class GestureDebouncer:
    def __init__(self, cooldown=1.0, history=5):
        self.cooldown  = cooldown
        self.last_time = 0
        self.history   = deque(maxlen=history)

    def should_fire(self, state):
        self.history.append(tuple(state))
        if len(set(self.history)) != 1:
            return False
        if time.time() - self.last_time < self.cooldown:
            return False
        self.last_time = time.time()
        return True


def handle_gesture(state):
    # Fist = play/pause
    if state == [False, False, False, False, False]:
        pyautogui.press('space')
        return "Play / Pause"

    # Open hand = next track
    elif state == [True, True, True, True, True]:
        pyautogui.hotkey('ctrl', 'right')
        return "Next Track"

    # Thumb + pinky = prev track
    elif state == [True, False, False, False, True]:
        pyautogui.hotkey('ctrl', 'left')
        return "Prev Track"

    # Middle + ring + pinky = volume up
    elif state == [False, False, True, True, True]:
        vol_up()
        return "Volume Up"

    # Ring + pinky = volume down
    elif state == [False, False, False, True, True]:
        vol_down()
        return "Volume Down"

    return None
