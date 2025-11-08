#!/usr/bin/env python3
from datetime import datetime
from pathlib import Path
import time
import signal
import argparse
from typing import Optional, List, Tuple, Iterable
import logging
import subprocess
import shlex

from camera import Camera as CamExt  # local module
from detector import Detector as DetectorExt  # local module

# Configure basic logging (keeps print statements intact for CLI simplicity)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
LOGGER = logging.getLogger("timelapse")


def _extract_labels_from_lines(lines: list[str]) -> list[str]:
    labels: list[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("no_detections"):
            continue
        parts = line.split()
        if parts:
            labels.append(parts[0])
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for l in labels:
        if l not in seen:
            uniq.append(l)
            seen.add(l)
    return uniq


def _run_on_match(cmd_template: str, *, image: Path, txt: Optional[Path], annotated: Optional[Path], labels: list[str], timestamp: str, save_dir: Path, sync: bool = False, timeout: Optional[int] = None, shell: bool = False) -> None:
    """Run an external command when a match occurs.

    Placeholders in cmd_template:
      {image} {txt} {annotated} {labels} {timestamp} {save_dir}
    PatInit commiths are substituted as absolute paths. {labels} is a comma-separated list.
    """
    try:
        ctx = {
            "image": str(image.resolve()),
            "txt": str(txt.resolve()) if txt else "",
            "annotated": str(annotated.resolve()) if annotated else "",
            "labels": ",".join(labels),
            "timestamp": timestamp,
            "save_dir": str(save_dir.resolve()),
        }
        formatted = cmd_template.format(**ctx)
        if shell:
            cmd = formatted if isinstance(formatted, str) else str(formatted)
        else:
            cmd = shlex.split(formatted)
        if sync:
            LOGGER.info("Running on-match command (sync): %s", formatted)
            try:
                res = subprocess.run(cmd, timeout=timeout, shell=shell, capture_output=True, text=True)
                LOGGER.info("on-match exit code: %s", res.returncode)
                if res.stdout:
                    LOGGER.debug("on-match stdout: %s", res.stdout.strip())
                if res.stderr:
                    LOGGER.debug("on-match stderr: %s", res.stderr.strip())
            except subprocess.TimeoutExpired:
                LOGGER.warning("on-match command timed out after %ss", timeout)
            except Exception as e:
                LOGGER.warning("on-match command failed: %s", e)
        else:
            LOGGER.info("Starting on-match command (async): %s", formatted)
            try:
                subprocess.Popen(cmd, shell=shell)
            except Exception as e:
                LOGGER.warning("Failed to start on-match command: %s", e)
    except Exception as e:
        LOGGER.warning("Failed to prepare on-match command: %s", e)


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
    p = argparse.ArgumentParser(description="Raspberry Pi timelapse with optional manual exposure and Ultralytics detection.")
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

    # Detection-related (Ultralytics only)
    p.add_argument("--detect", action="store_true", default=DETECT_ENABLE, help="Enable object detection for each image (default: on)")
    p.add_argument("--no-detect", dest="detect", action="store_false", help="Disable object detection")
    p.add_argument("--model", type=str, default=YOLO_MODEL, help="Ultralytics model path or name (default: yolov8n.pt)")
    p.add_argument("--conf", type=float, default=YOLO_CONF, help="Detection confidence threshold (ultralytics) (default: 0.25)")
    p.add_argument("--imgsz", type=int, default=640, help="Ultralytics inference image size (short side). Lower is faster but less accurate (e.g., 416)")
    p.add_argument("--yolo-classes", nargs="+", help="Restrict Ultralytics to these classes (names or IDs). Example: --yolo-classes 'sports ball' 32")

    p.add_argument("--save-annotated", action="store_true", default=SAVE_ANNOTATED, help="Also save annotated image with boxes (adds _det.jpg)")

    # Conditional saving: keep captures only if at least one of these labels is detected
    p.add_argument("--save-on", nargs="+", help="Only keep the image if one of these labels is detected (case-insensitive). Examples: --save-on orange person 'sports ball'")

    # Callback on match: run external command or script
    p.add_argument("--on-match-cmd", type=str, help="Command to run when a required label is detected and the image is saved. Supports placeholders: {image} {txt} {annotated} {labels} {timestamp} {save_dir}")
    p.add_argument("--on-match-sync", action="store_true", help="Run the on-match command synchronously and wait for it to finish (default: run asynchronously)")
    p.add_argument("--on-match-timeout", type=int, default=None, help="Timeout in seconds for synchronous on-match command")
    p.add_argument("--on-match-shell", action="store_true", help="Run the on-match command via shell=True (use carefully)")

    return p.parse_args()


def main():
    args = parse_args()

    # Basic validation for callback args
    if getattr(args, "on_match_timeout", None) and not getattr(args, "on_match_sync", False):
        LOGGER.warning("--on-match-timeout has no effect without --on-match-sync")

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
        yolo_classes=args.yolo_classes,
        save_on=args.save_on,
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
            LOGGER.info("Detection: ON → Ultralytics, model=%s, conf=%s, imgsz=%s", args.model, args.conf, args.imgsz)
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
            if detector.enabled and args.save_on:
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
                        # On-match callback (fallback path) when kept
                        if args.on_match_cmd:
                            # Try to get labels from .txt we just wrote
                            labels = []
                            try:
                                with (txt_path if txt_path else filename.with_suffix('.txt')).open('r') as f:
                                    for line in f:
                                        line = line.strip()
                                        if not line or line.startswith('no_detections'):
                                            continue
                                        parts = line.split()
                                        if parts:
                                            labels.append(parts[0])
                            except Exception:
                                pass
                            det_img_path = filename.with_name(filename.stem + "_det.jpg")
                            annotated_path = det_img_path if det_img_path.exists() else None
                            _run_on_match(
                                args.on_match_cmd,
                                image=filename,
                                txt=txt_path,
                                annotated=annotated_path,
                                labels=list(dict.fromkeys(labels)),
                                timestamp=ts,
                                save_dir=save_dir,
                                sync=getattr(args, 'on_match_sync', False),
                                timeout=getattr(args, 'on_match_timeout', None),
                                shell=getattr(args, 'on_match_shell', False),
                            )
                else:
                    matched, det_lines, annotated, objects = result
                    LOGGER.info(f"Found: {objects}")
                    if matched:
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
                            annotated_path = None
                            if annotated is not None:
                                det_img = filename.with_name(filename.stem + "_det.jpg")
                                try:
                                    cv2.imwrite(str(det_img), annotated)
                                    annotated_path = det_img
                                except Exception as e:
                                    LOGGER.warning(f"Failed to save annotated image: {e}")
                            # On-match callback (in-memory path)
                            if args.on_match_cmd:
                                labels = _extract_labels_from_lines(det_lines)
                                _run_on_match(
                                    args.on_match_cmd,
                                    image=filename,
                                    txt=txt_path,
                                    annotated=annotated_path,
                                    labels=labels,
                                    timestamp=ts,
                                    save_dir=save_dir,
                                    sync=getattr(args, 'on_match_sync', False),
                                    timeout=getattr(args, 'on_match_timeout', None),
                                    shell=getattr(args, 'on_match_shell', False),
                                )
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
                else:
                    # On-match callback (default/file-based path) when kept and detection enabled
                    if detector.enabled and args.on_match_cmd and (not args.save_on or matched):
                        labels = []
                        try:
                            with (txt_path if txt_path else filename.with_suffix('.txt')).open('r') as f:
                                for line in f:
                                    line = line.strip()
                                    if not line or line.startswith('no_detections'):
                                        continue
                                    parts = line.split()
                                    if parts:
                                        labels.append(parts[0])
                        except Exception:
                            pass
                        det_img_path = filename.with_name(filename.stem + "_det.jpg")
                        annotated_path = det_img_path if det_img_path.exists() else None
                        _run_on_match(
                            args.on_match_cmd,
                            image=filename,
                            txt=txt_path,
                            annotated=annotated_path,
                            labels=list(dict.fromkeys(labels)),
                            timestamp=ts,
                            save_dir=save_dir,
                            sync=getattr(args, 'on_match_sync', False),
                            timeout=getattr(args, 'on_match_timeout', None),
                            shell=getattr(args, 'on_match_shell', False),
                        )

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
