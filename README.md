# Hand Gesture Control

Control your mouse cursor using hand gestures via webcam.

## Requirements

- Python 3.8+
- Webcam
- Dependencies: `pip install -r requirements.txt`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Controls

| Gesture | Action |
|---------|--------|
| Open palm | Move cursor |
| Pinch (thumb + index) | Click |
| Double pinch | Double click |
| Pinch + hold | Drag |
| Middle finger | 靠北 😂 |

## Build .exe

```bash
pip install pyinstaller
pyinstaller hand-gesture.spec
```

Output: `dist/HandGesture/HandGesture.exe`

## Quit

Press `q` to exit.

## Troubleshooting

- **No camera?**: Make sure webcam is not used by another app
- **Not responding?**: Adjust `PINCH_THRESHOLD` in `config.py` (lower = more strict)
- **Cursor jumpy?**: Increase `SMOOTHING_WINDOW` in `config.py`
