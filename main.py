#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
import time
import signal
import argparse
from typing import Optional, List, Tuple, Iterable
import logging

from camera import Camera as CamExt  # local module
from detector import Detector as DetectorExt  # local module

# Configure basic logging (keeps print statements intact for CLI simplicity)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("timelapse")


# Lazy imports for detection (ultralytics / cv2) are inside Detector

# Configure where to save and the interval
SAVE_DIR = Path("./timelapse")  # change this path if you like
INTERVAL_SEC = 5                        # seconds between shots

# Optional exposure controls (set via CLI too)
AE_ENABLE = True          # auto exposure on by default
SHUTTER_US = None         # e.g. 1000 = 1 ms (1/1000 s). Use with AE_ENABLE=False
ISO = None                # e.g. 400. Roughly ISO ≈ AnalogueGain*100
AWB_ENABLE = True         # auto white balance

# Detection defaults
DETECT_ENABLE = True
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.25
SAVE_ANNOTATED = False

running = True

def handle_sigint(sig, frame):
    """Signal handler to stop the timelapse loop gracefully."""
    global running
    LOGGER.info("Stopping timelapse…")
    running = False






def apply_exposure_controls(camera: CamExt, args) -> None:
    """Apply exposure and white balance controls.

    Notes:
    - On Picamera2, both auto and manual exposure are supported.
    - On OpenCV webcam backend, these controls are mostly ignored and a debug
      log is emitted when exposure-specific fields are provided.
    """
    # Decide AE/AWB
    controls = {"AeEnable": args.ae, "AwbEnable": args.awb}

    # If manual exposure requested, disable AE and set ExposureTime/AnalogueGain
    if not args.ae:
        if args.shutter_us is None:
            raise ValueError("Manual exposure selected but no --shutter-us provided")
        controls["ExposureTime"] = int(args.shutter_us)
        if args.iso is not None:
            # ISO ~ AnalogueGain*100 (approx.). Use at your own discretion.
            controls["AnalogueGain"] = max(1.0, float(args.iso) / 100.0)
        # Optionally limit frame duration to help the camera keep short exposures
        # (min_us, max_us). Here we set max close to shutter to avoid AE changing frame time.
        controls["FrameDurationLimits"] = (int(args.shutter_us), int(max(args.shutter_us, args.shutter_us)))

    camera.set_controls(controls)


def parse_args():
    p = argparse.ArgumentParser(description="Raspberry Pi timelapse with optional manual exposure and multiple detection backends (OpenCV or Ultralytics).")
    p.add_argument("--save-dir", type=Path, default=SAVE_DIR, help="Folder to save images")
    p.add_argument("--interval", type=int, default=INTERVAL_SEC, help="Seconds between shots")

    # Exposure-related
    p.add_argument("--ae", action="store_true", default=AE_ENABLE, help="Enable auto exposure (default: on)")
    p.add_argument("--no-ae", dest="ae", action="store_false", help="Disable auto exposure for manual shutter/ISO")
    p.add_argument("--shutter-us", type=int, default=SHUTTER_US, help="Manual shutter in microseconds (e.g., 1000 = 1/1000s). Requires --no-ae")
    p.add_argument("--iso", type=int, default=ISO, help="Approximate ISO (100..800); sets analogue gain. Used with --no-ae")

    # White balance
    p.add_argument("--awb", action="store_true", default=AWB_ENABLE, help="Enable auto white balance (default: on)")
    p.add_argument("--no-awb", dest="awb", action="store_false", help="Disable auto white balance")

    # Detection-related
    p.add_argument("--detect", action="store_true", default=DETECT_ENABLE, help="Enable object detection for each image (default: on)")
    p.add_argument("--no-detect", dest="detect", action="store_false", help="Disable object detection")
    p.add_argument("--detector", type=str, default="opencv-motion", choices=["opencv-motion", "opencv-dnn-ssd", "opencv-ball", "ultralytics"], help="Detection backend to use")
    # For ultralytics
    p.add_argument("--model", type=str, default=YOLO_MODEL, help="Ultralytics model path or name (default: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=YOLO_CONF, help="Detection confidence threshold (ultralytics) (default: 0.25)")
    p.add_argument("--imgsz", type=int, default=640, help="Ultralytics inference image size (short side). Lower is faster but less accurate (e.g., 416)")
    p.add_argument("--yolo-classes", nargs="+", help="Restrict Ultralytics to these classes (names or IDs). Example: --yolo-classes 'sports ball' 32")
    # For OpenCV motion
    p.add_argument("--min-area", type=int, default=500, help="Minimum bounding box area (pixels) for motion detections (opencv-motion)")
    # For OpenCV SSD
    p.add_argument("--ssd-prototxt", type=Path, help="Path to MobileNet-SSD deploy.prototxt (opencv-dnn-ssd)")
    p.add_argument("--ssd-weights", type=Path, help="Path to MobileNet-SSD caffemodel (opencv-dnn-ssd)")
    p.add_argument("--ssd-conf", type=float, default=0.4, help="Confidence threshold for SSD (opencv-dnn-ssd)")
    # For OpenCV ball (HoughCircles)
    p.add_argument("--ball-dp", type=float, default=1.2, help="Inverse ratio of accumulator resolution to image resolution (HoughCircles)")
    p.add_argument("--ball-min-dist", type=int, default=20, help="Minimum distance between detected circle centers")
    p.add_argument("--ball-canny", type=int, default=100, help="Higher Canny edge threshold passed to HoughCircles (param1)")
    p.add_argument("--ball-accum", type=int, default=30, help="Accumulator threshold for circle centers in HoughCircles (param2). Lower → more detections")
    p.add_argument("--ball-min-radius", type=int, default=5, help="Minimum circle radius in pixels (set 0 to auto)")
    p.add_argument("--ball-max-radius", type=int, default=80, help="Maximum circle radius in pixels (set 0 to auto)")
    p.add_argument("--ball-resize", type=float, default=1.0, help="Optional downscale factor for processing (e.g., 0.75 or 0.5); mapped back to original coords")
    p.add_argument("--ball-hsv-lower", nargs=3, type=int, help="Optional HSV lower bound (H S V) for color mask, e.g., 20 50 50")
    p.add_argument("--ball-hsv-upper", nargs=3, type=int, help="Optional HSV upper bound (H S V) for color mask, e.g., 35 255 255")

    p.add_argument("--save-annotated", action="store_true", default=SAVE_ANNOTATED, help="Also save annotated image with boxes (adds _det.jpg)")

    # Conditional saving: keep captures only if at least one of these labels is detected
    p.add_argument("--save-on", nargs="+", help="Only keep the image if one of these labels is detected (case-insensitive). Examples: --save-on orange person 'sports ball'")

    return p.parse_args()


def main():
    args = parse_args()

    save_dir = args.save_dir
    interval = args.interval

    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare detector (may disable itself if deps are missing)
    detector = DetectorExt(
        enabled=args.detect,
        model_path=args.model,
        conf=args.conf,
        save_annotated=args.save_annotated,
        imgsz=args.imgsz,
        detector=args.detector,
        min_area=args.min_area,
        ssd_prototxt=args.ssd_prototxt,
        ssd_weights=args.ssd_weights,
        ssd_conf=args.ssd_conf,
        yolo_classes=args.yolo_classes,
        save_on=args.save_on,
        ball_dp=args.ball_dp,
        ball_min_dist=args.ball_min_dist,
        ball_canny=args.ball_canny,
        ball_accum=args.ball_accum,
        ball_min_radius=args.ball_min_radius,
        ball_max_radius=args.ball_max_radius,
        ball_resize=args.ball_resize,
        ball_hsv_lower=tuple(args.ball_hsv_lower) if args.ball_hsv_lower else None,
        ball_hsv_upper=tuple(args.ball_hsv_upper) if args.ball_hsv_upper else None,
    )
    if args.detect and not detector.enabled:
        LOGGER.info("Continuing without detection. Install dependencies per README to enable it.")
        if args.save_on:
            LOGGER.warning("--save-on is set but detection is disabled; images will NOT be filtered.")

    # Initialize camera (auto-detects Picamera2 vs OpenCV webcam)
    with CamExt() as camera:
        # Apply requested exposure settings (Picamera2 only)
        try:
            apply_exposure_controls(camera, args)
        except Exception as e:
            LOGGER.warning(f"Failed to apply exposure controls: {e}")

        LOGGER.info(f"Saving one image every {interval}s to {save_dir} (Ctrl+C to stop)")
        if not args.ae:
            LOGGER.info(
                "Manual exposure enabled → Shutter: %s us%s",
                args.shutter_us,
                f", ISO≈{args.iso}" if args.iso is not None else "",
            )
        if detector.enabled:
            if args.detector == 'ultralytics':
                LOGGER.info("Detection: ON → backend=ultralytics, model=%s, conf=%s, imgsz=%s", args.model, args.conf, args.imgsz)
            elif args.detector == 'opencv-motion':
                LOGGER.info("Detection: ON → backend=opencv-motion, min_area=%s", args.min_area)
            elif args.detector == 'opencv-dnn-ssd':
                LOGGER.info("Detection: ON → backend=opencv-dnn-ssd, conf=%s", args.ssd_conf)
            elif args.detector == 'opencv-ball':
                hsv = None
                if args.ball_hsv_lower and args.ball_hsv_upper:
                    hsv = f"HSV={tuple(args.ball_hsv_lower)}..{tuple(args.ball_hsv_upper)}"
                LOGGER.info(
                    "Detection: ON → backend=opencv-ball, dp=%s, min_dist=%s, canny=%s, accum=%s, minR=%s, maxR=%s, resize=%s%s",
                    args.ball_dp, args.ball_min_dist, args.ball_canny, args.ball_accum,
                    args.ball_min_radius, args.ball_max_radius, args.ball_resize,
                    f", {hsv}" if hsv else "",
                )
            if args.save_annotated:
                LOGGER.info("Annotated images will be saved with suffix _det.jpg")
            if args.save_on:
                LOGGER.info("Save filter active: images will only be kept if one of %s is detected", args.save_on)
        else:
            LOGGER.info("Object detection: OFF")
            if args.save_on:
                LOGGER.warning("--save-on specified but detection is OFF; images will NOT be filtered.")

        while running:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            base = save_dir / f"image_{ts}"

            # Fast path: Ultralytics + save-on → do in-memory detection first; only save on match
            if detector.enabled and args.detector == 'ultralytics' and args.save_on:
                try:
                    frame = camera.capture_array()
                except Exception as e:
                    LOGGER.warning(f"Capture failed, skipping this cycle: {e}")
                    # Sleep after failure
                    for _ in range(interval):
                        if not running:
                            break
                        time.sleep(1)
                    continue
                result = detector.process_frame(frame)
                if result is None:
                    # Fallback to file-based flow if in-memory path isn't available
                    filename = base.with_suffix('.jpg')
                    camera.capture_file(str(filename))
                    LOGGER.info("Saved %s", filename)
                    matched = True
                    txt_path = None
                    det_res = detector.process_image(filename)
                    if det_res is not None:
                        txt_path, matched = det_res
                        LOGGER.info("Detections written to %s", txt_path.name)
                    if args.save_on and not matched:
                        try:
                            filename.unlink(missing_ok=True)
                            if txt_path:
                                txt_path.unlink(missing_ok=True)
                            det_img = filename.with_name(filename.stem + "_det.jpg")
                            # det_img.unlink(missing_ok=True)
                            LOGGER.info("Discarded capture (no required label detected)")
                        except Exception as e:
                            LOGGER.warning("Failed to remove unneeded files for %s: %s", filename.name, e)
                else:
                    matched, det_lines, annotated = result
                    if matched:

                        # We can do anything here

                        # Save image now
                        try:
                            import cv2  # type: ignore
                            filename = base.with_suffix('.jpg')
                            cv2.imwrite(str(filename), frame)
                            LOGGER.info("Saved %s", filename)
                            # Write sidecar .txt
                            txt_path = filename.with_suffix('.txt')
                            with txt_path.open('w') as f:
                                for line in det_lines:
                                    f.write(line + "\n")
                            LOGGER.info("Detections written to %s", txt_path.name)
                            # Optional annotated image
                            if annotated is not None:
                                det_img = filename.with_name(filename.stem + "_det.jpg")
                                try:
                                    cv2.imwrite(str(det_img), annotated)
                                except Exception as e:
                                    LOGGER.warning(f"Failed to save annotated image: {e}")
                        except Exception as e:
                            LOGGER.warning(f"Failed to save matched capture: {e}")
                    else:
                        LOGGER.info("No required label detected → not saving this capture")
            else:
                # Default path: capture file first, then run detection
                filename = base.with_suffix('.jpg')
                camera.capture_file(str(filename))
                LOGGER.info("Saved %s", filename)

                # Run detection and write sidecar .txt (and optional _det.jpg)
                matched = True
                txt_path = None
                if detector.enabled:
                    result = detector.process_image(filename)
                    if result is not None:
                        txt_path, matched = result
                        LOGGER.info("Detections written to %s", txt_path.name)
                # If save filter is active and nothing matched, delete files
                if detector.enabled and args.save_on and not matched:
                    try:
                        filename.unlink(missing_ok=True)
                        if txt_path:
                            txt_path.unlink(missing_ok=True)
                        det_img = filename.with_name(filename.stem + "_det.jpg")
                        #det_img.unlink(missing_ok=True)
                        LOGGER.info("Discarded capture (no required label detected)")
                    except Exception as e:
                        LOGGER.warning("Failed to remove unneeded files for %s: %s", filename.name, e)

            # Sleep after capture so the first image is immediate
            for _ in range(interval):
                if not running:
                    break
                time.sleep(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    try:
        main()
    except KeyboardInterrupt:
        pass
