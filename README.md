# Hand Gesture Controller

Control your PC using hand gestures via webcam — no mouse or keyboard needed.

## Requirements

```bash
pip install opencv-python mediapipe pyautogui pynput numpy
```

## Run

```bash
python main.py
```

## Gestures

| Gesture | Action |
|---|---|
| Index finger only | Move cursor |
| Index + thumb pinch | Left click |
| Index + middle + thumb pinch | Right click |
| Peace sign + move up/down | Scroll |
| Fist | Play / Pause |
| Open hand | Next track |
| Thumb + pinky | Previous track |
| Middle + ring + pinky | Volume up |
| Ring + pinky | Volume down |

## Notes

- Press `ESC` to quit
- Requires a webcam (index 0 by default)
- The hand landmark model (~9MB) is auto-downloaded on first run
