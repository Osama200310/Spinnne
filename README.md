# Raspberry Pi Timelapse Capture

This project captures still images on a schedule and can optionally run object detection on each capture. It supports:
- Raspberry Pi Camera via Picamera2
- Generic webcams via OpenCV (VideoCapture)

Images are saved to a folder at a fixed interval. If detection is enabled, a sidecar `.txt` with detections is written and (optionally) an annotated preview image `_det.jpg` is saved. You can now also keep images only when specific labels are detected using `--save-on`.

---
## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# On Raspberry Pi, also install Picamera2 via apt (see below)

python3 main.py
```

- Default save folder: `./timelapse`
- Default interval: `5` seconds
- Stop with `Ctrl+C`

---
## Object detection and conditional saving (Ultralytics only)

This project now uses a single detection backend: Ultralytics YOLO (e.g., YOLOv8). OpenCV-based detectors have been removed for simplicity.

Common flags:
- `--detect` / `--no-detect` to turn detection on/off (default: on)
- `--model` to choose the Ultralytics model (default: `yolov8n.pt`)
- `--conf` and `--imgsz` to tune inference
- `--yolo-classes <names/ids...>` to restrict which classes are considered/written
- `--save-annotated` to also write an annotated image with boxes (`_det.jpg`)
- `--save-on <labels...>` keep the image only if at least one of these labels is detected (case-insensitive). You can pass multiple labels (space separated).

Behavior with `--save-on`:
- The app runs YOLO on the raw camera frame in memory first and only saves the JPEG if a required label is detected. This avoids writing and then deleting images, saving I/O and SD-card wear.

Examples:
```bash
# Keep only images that contain an orange (YOLO model with that class)
python3 main.py --model yolov8n.pt --save-on orange

# Keep only oranges OR apples (multiple labels)
python3 main.py --model yolov8n.pt --save-on orange apple

# Restrict YOLO classes and keep only when a sports ball is found
python3 main.py --model yolov8n.pt --yolo-classes "sports ball" 32 --save-on "sports ball"
```
Notes:
- Matching is case-insensitive; YOLO class names accept spaces/underscores interchangeably. You may also pass numeric class IDs in `--save-on`.
- `--yolo-classes` still works to restrict which YOLO classes are considered/written.
- When detection is disabled or fails to initialize, `--save-on` has no effect and all images will be kept (a warning is logged).

---
        ## Prerequisites

On Raspberry Pi (Picamera2 backend):
- Raspberry Pi OS Bullseye/Bookworm with the libcamera stack enabled
- Raspberry Pi camera connected and working
- Picamera2 library (install via apt):

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

Make sure your user is in the `video` group (log out/in after):

```bash
sudo usermod -aG video $USER
```

Optional dependencies for detection:
- OpenCV (Python wheels): installed via `pip install -r requirements.txt`
- Ultralytics/YOLO: `pip install ultralytics` (already listed in `requirements.txt` if you plan to use it)
- SSD model files: MobileNet-SSD `deploy.prototxt` and `*.caffemodel`

---
## Usage and configuration

Basic run:
```bash
python3 main.py --save-dir ./timelapse --interval 5
```

Exposure and white balance (Picamera2 only; ignored for generic webcams):
- Use auto-exposure (default): `--ae`
- Manual exposure: `--no-ae --shutter-us <microseconds> [--iso <100..800>]`
  - Example: `--no-ae --shutter-us 1000 --iso 400` → 1/1000s shutter, ISO≈400
- Auto white balance on/off: `--awb` / `--no-awb`

Ultralytics-specific:
- `--model yolov8n.pt` (path or name)
- `--conf 0.25` and `--imgsz 640`
- Optionally restrict classes with `--yolo-classes` (names or IDs). Example: `--yolo-classes "sports ball" 32`


---
## Freeze motion / reduce blur

Motion blur happens when shutter speed is too slow relative to subject motion. To freeze faster movement, use a shorter exposure (smaller `--shutter-us`). Shorter exposures make the image darker, so you may need to raise `--iso` and/or add more light.

Suggested starting points:
- Bright daylight, walking people: `--no-ae --shutter-us 1000 --iso 200` (≈1/1000s)
- Daylight, fast motion (cars/birds): `--no-ae --shutter-us 500 --iso 400` (≈1/2000s)
- Indoor bright room: `--no-ae --shutter-us 2000 --iso 800` (≈1/500s)

---

The legacy notes below are kept for reference and may mention older file names or defaults. The above sections supersede them.

## Freeze motion / reduce blur

Motion blur happens when shutter speed is too slow relative to subject motion. To freeze faster movement, use a shorter exposure (smaller `--shutter-us`). Shorter exposures make the image darker, so you may need to raise `--iso` and/or add more light.

Suggested starting points:

- Bright daylight, walking people: `--no-ae --shutter-us 1000 --iso 200` (≈1/1000s)
- Daylight, fast motion (cars/birds): `--no-ae --shutter-us 500 --iso 400` (≈1/2000s)
- Indoor bright room: `--no-ae --shutter-us 2000 --iso 800` (≈1/500s) — may still be dark; add light if needed.

Examples:

```bash
# 1/1000s shutter, ISO 400, save every 5s
python3 main.py --no-ae --shutter-us 1000 --iso 400

# 1/2000s shutter for very fast motion
python3 main.py --no-ae --shutter-us 500 --iso 400

# Keep auto exposure but lock white balance off
python3 main.py --ae --no-awb
```

Tips:
- If images are too dark, try a higher `--iso` (up to ~800) and add light.
- Extremely fast shutters (e.g., `--shutter-us 200`) may be too dark except in strong sunlight.
- For consistent look, keep `--no-ae` so brightness doesn’t fluctuate between shots.

## Quick alternative (no Python)

If you prefer a pure shell loop using libcamera:

```bash
mkdir -p ~/timelapse
while true; do libcamera-still -o "~/timelapse/image_$(date +%F_%H-%M-%S).jpg"; sleep 5; done
```

## Notes

- If `picamera2` is not found in your Python environment, install it with `sudo apt install python3-picamera2` (on Raspberry Pi OS). On other OSes, Picamera2 may not be available via pip.
- On autofocus cameras, Picamera2 typically manages focus automatically. You can experiment with controls if needed.

# Raspberry Pi Timelapse Capture

This project contains a simple Python script that captures a still image from a Raspberry Pi camera every 5 seconds and saves it into a folder.

## Prerequisites

- Raspberry Pi OS Bullseye/Bookworm with the libcamera stack enabled
- Raspberry Pi camera connected and working
- Picamera2 library (install via apt):

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

Make sure your user is in the `video` group (log out/in after):

```bash
sudo usermod -aG video $USER
```

## Usage

1. Run the script:

```bash
python3 main.py
```

By default, images are saved to `./timelapse` every 5 seconds.

2. Stop with `Ctrl+C`.

## Configuration

You can configure everything via command-line options (see below).

- Save folder: `--save-dir /path/to/folder`
- Interval seconds: `--interval 5`

Optional exposure controls to reduce motion blur:

- Use auto-exposure (default): `--ae`
- Manual exposure: `--no-ae --shutter-us <microseconds> [--iso <100..800>]`
  - Example: `--no-ae --shutter-us 1000 --iso 400` → 1/1000s shutter, ISO≈400
- Auto white balance on/off: `--awb` / `--no-awb`

Advanced Picamera2 tuning:
- Most users should stick to the CLI flags above. If you need deeper control (sensor mode, resolution, special Picamera2 controls), extend `camera.py` where the Picamera2 configuration is created and `set_controls` is called.
- The OpenCV webcam backend ignores most exposure-related controls; use the Raspberry Pi camera for full control.

## Freeze motion / reduce blur

Motion blur happens when shutter speed is too slow relative to subject motion. To freeze faster movement, use a shorter exposure (smaller `--shutter-us`). Shorter exposures make the image darker, so you may need to raise `--iso` and/or add more light.

Suggested starting points:

- Bright daylight, walking people: `--no-ae --shutter-us 1000 --iso 200` (≈1/1000s)
- Daylight, fast motion (cars/birds): `--no-ae --shutter-us 500 --iso 400` (≈1/2000s)
- Indoor bright room: `--no-ae --shutter-us 2000 --iso 800` (≈1/500s) — may still be dark; add light if needed.

Examples:

```bash
# 1/1000s shutter, ISO 400, save every 5s
python3 main.py --no-ae --shutter-us 1000 --iso 400

# 1/2000s shutter for very fast motion
python3 main.py --no-ae --shutter-us 500 --iso 400

# Keep auto exposure but lock white balance off
python3 main.py --ae --no-awb
```

Tips:
- If images are too dark, try a higher `--iso` (up to ~800) and add light.
- Extremely fast shutters (e.g., `--shutter-us 200`) may be too dark except in strong sunlight.
- For consistent look, keep `--no-ae` so brightness doesn’t fluctuate between shots.

## Quick alternative (no Python)

If you prefer a pure shell loop using libcamera:

```bash
mkdir -p ~/timelapse
while true; do libcamera-still -o "~/timelapse/image_$(date +%F_%H-%M-%S).jpg"; sleep 5; done
```

## Notes

- If `picamera2` is not found in your Python environment, install it with `sudo apt install python3-picamera2` (on Raspberry Pi OS). On other OSes, Picamera2 may not be available via pip.
- On autofocus cameras, Picamera2 typically manages focus automatically. You can experiment with controls if needed.

---

## Object Detection (YOLOv8 Nano) and Logging to Text

This project includes optional object detection using Ultralytics YOLOv8 Nano (`yolov8n.pt`). After each captured image, the script can run detection and write a sidecar text file next to the image containing the detections.

- Default model: `yolov8n.pt` (YOLO v8 nano)
- Output file: a `.txt` with the same base name as the image
- Each detection line format:
  - `class_name confidence x1 y1 x2 y2` (pixel coordinates)
  - Example: `person 0.842 123.0 45.0 320.0 400.0`
- If no detections: the file contains `no_detections`
- Optional: save an annotated image with bounding boxes by passing `--save-annotated` (adds `_det.jpg`)

### Install detection dependencies

On Raspberry Pi OS, keep using `picamera2` from apt. Install the Python packages for detection via pip in your environment (venv recommended):

```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# If OpenCV headless wheel is not available for your Pi and you want annotated images:
sudo apt install -y python3-opencv
```

Notes:
- `ultralytics` installs PyTorch as a dependency. On some Pi models, prebuilt wheels may not be available and building from source is slow. Consider running detection on a more capable device, or lower the image size with `--imgsz`.
- `picamera2` stays installed via apt and is not in `requirements.txt`.

### Run with detection

Detection is ON by default. To run with defaults (YOLOv8n, conf=0.25, imgsz=640):

```bash
python3 main.py --save-dir /home/pi/timelapse
```

Useful flags:
- `--no-detect` to disable detection entirely.
- `--model yolov8n.pt` to choose a different model or a local path.
- `--conf 0.3` to raise/lower the confidence threshold.
- `--imgsz 416` to speed up inference on slower devices (lower is faster but less accurate).
- `--save-annotated` to store `*_det.jpg` images with drawn boxes (requires OpenCV; see above).

### Performance tips for slow devices (e.g., Raspberry Pi)

- Use the nano model (`yolov8n.pt`, default).
- Reduce inference size: try `--imgsz 416` or even `--imgsz 320`.
- Increase the capture interval, e.g., `--interval 10` or `--interval 30`.
- Avoid saving annotated images unless needed (`--save-annotated` costs extra CPU).
- Consider reducing camera resolution in the still configuration if you primarily rely on detection.
- Keep the device cool; thermal throttling will slow inference.

---

## Legacy: Detection backends (deprecated; project now uses Ultralytics only)

The project previously supported several OpenCV-based detectors (`opencv-motion`, `opencv-dnn-ssd`, `opencv-ball`). These have been removed. If you need similar functionality, consider implementing it as a separate module or fork based on earlier commits. Installation now focuses on Picamera2 for capture and Ultralytics for detection:

- Base requirements (in venv recommended):
  ```bash
  python3 -m venv .venv --system-site-packages   # so Picamera2 from apt is visible
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Raspberry Pi packages (system):
  ```bash
  sudo apt update
  sudo apt install -y python3-picamera2    # camera library (mandatory)
  # Optional if pip wheel isn’t available or you prefer system OpenCV for webcam/annotation
  sudo apt install -y python3-opencv
  ```

### Virtualenv and Picamera2 on Raspberry Pi
Picamera2 is installed via apt into system site‑packages. Create your venv with `--system-site-packages` so the venv can import it:
```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```
If you instead install OpenCV via apt (`python3-opencv`), this also makes the apt OpenCV visible inside the venv.
