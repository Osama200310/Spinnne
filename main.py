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


def _canon(s: str) -> str:
    """Canonicalize label for matching: lowercase, alnum only."""
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _extract_best_score_from_lines(lines: list[str], save_on: Optional[list[str]]) -> Optional[float]:
    """Return the highest detection confidence.

    If save_on is provided, consider only detections whose label matches any of save_on
    (case-insensitive, canonicalized). Otherwise consider all detections.
    Returns None if no candidates found.
    """
    targets = None
    if save_on:
        targets = { _canon(x) for x in save_on }
        if "ball" in targets:
            targets.add("sportsball")
    best: Optional[float] = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("no_detections"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        label = parts[0]
        try:
            score = float(parts[1])
        except ValueError:
            continue
        if targets is not None and _canon(label) not in targets:
            continue
        if best is None or score > best:
            best = score
    return best


def _run_on_match(cmd_template: str, *, image: Path, txt: Optional[Path], annotated: Optional[Path], labels: list[str], timestamp: str, save_dir: Path, sync: bool = False, timeout: Optional[int] = None, shell: bool = False) -> None:
    """Run an external command when a match occurs.

    Placeholders in cmd_template:
      {image} {txt} {annotated} {labels} {timestamp} {save_dir}
    Paths are substituted as absolute paths. {labels} is a comma-separated list.
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


def _post_on_match(*, url: str, image: Path, txt: Optional[Path], annotated: Optional[Path], labels: list[str], timestamp: str, save_dir: Path, timeout: Optional[int] = None, headers: Optional[dict] = None, include_annotated: bool = False, image_field_name: str = "image", duplicate_file_field: bool = False, only_file: bool = False, score: Optional[float] = None, score_field_name: str = "score", objects: Optional[list[str]] = None, objects_field_name: str = "objects") -> None:
    """POST the captured image (and optional artifacts) to a web server.

    Sends multipart/form-data with fields:
      - labels: comma-separated labels (omitted if only_file=True)
      - timestamp: ISO-like timestamp string used in filename (omitted if only_file=True)
      - save_dir: absolute path of the save directory (informational) (omitted if only_file=True)
      - score: best detection score (if provided). Included even when only_file=True.
      - objects: semicolon-separated detected class names (if provided). Included even when only_file=True.
    Files:
      - <image_field_name>: the captured JPEG (default key: image)
      - file: optional duplicate of image for FastAPI-style endpoints (if duplicate_file_field=True)
      - txt: YOLO-like detections text (if available and only_file=False)
      - annotated: annotated JPEG with boxes (if available and include_annotated=True and only_file=False)
    """
    try:
        try:
            import requests  # type: ignore
        except Exception:
            LOGGER.warning("'requests' is not installed; cannot POST to %s", url)
            return

        data = {}
        if not only_file:
            data = {
                "labels": ",".join(labels),
                "timestamp": timestamp,
                "save_dir": str(save_dir.resolve()),
            }
        # Include score field if provided (even in only_file mode)
        if score is not None:
            try:
                data[score_field_name] = f"{float(score):.3f}"
            except Exception:
                # fallback to raw conversion
                data[score_field_name] = str(score)
        # Include objects field if provided (even in only_file mode)
        LOGGER.warning("labels: %s", labels)
        if labels:
            try:
                # de-duplicate while preserving order
                seen = set()
                uniq = []
                for o in labels:
                    s = str(o).strip()
                    if not s:
                        continue
                    if s not in seen:
                        uniq.append(s)
                        seen.add(s)
                if uniq:
                    data[objects_field_name] = ";".join(uniq)
            except Exception:
                try:
                    data[objects_field_name] = ";".join(map(str, labels))
                except Exception:
                    pass
        files = {}
        # Primary image field
        try:
            files[image_field_name] = (annotated.name, annotated.open("rb"), "image/jpeg")
        except Exception as e:
            LOGGER.warning("Failed to open image for POST: %s", e)
            return
        # Optional duplicate under 'file' for compatibility (requires a separate handle)
        if not only_file and duplicate_file_field and image_field_name != "file":
            try:
                files["file"] = (annotated.name, annotated.open("rb"), "image/jpeg")
            except Exception as e:
                LOGGER.debug("Failed to attach duplicate 'file' field: %s", e)
        if not only_file and txt and txt.exists():
            try:
                files["txt"] = (txt.name, txt.open("rb"), "text/plain")
            except Exception as e:
                LOGGER.debug("Failed to attach txt for POST: %s", e)
        if not only_file and include_annotated and annotated and Path(annotated).exists():
            try:
                files["annotated"] = (Path(annotated).name, Path(annotated).open("rb"), "image/jpeg")
            except Exception as e:
                LOGGER.debug("Failed to attach annotated for POST: %s", e)

        LOGGER.info("POSTing image to %s", url)
        try:
            resp = requests.post(url, data=data if data else None, files=files, headers=headers or {}, timeout=timeout or 15)
            LOGGER.info("POST status: %s", resp.status_code)
            if resp.status_code >= 400:
                # Print a helpful hint for common 422 "missing 'file'" cases
                msg = resp.text[:1000] if hasattr(resp, "text") else str(resp)
                if resp.status_code == 422 and ("\"file\"" in msg or "'file'" in msg) and "Field required" in msg:
                    LOGGER.warning("Server reports missing 'file' field. Try --post-image-field file or --post-compat-file-field.")
                LOGGER.warning("POST failed: %s", msg)
        except requests.exceptions.RequestException as e:
            LOGGER.warning("POST request failed: %s", e)
        finally:
            # Close any file handles we opened in 'files'
            for k, v in list(files.items()):
                try:
                    v[1].close()
                except Exception:
                    pass
    except Exception as e:
        LOGGER.warning("Unexpected error in POST helper: %s", e)


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

    # HTTP POST on match
    p.add_argument("--post-url", type=str, help="If set, POST the saved image to this URL when a required label is detected")
    p.add_argument("--post-timeout", type=int, default=15, help="Timeout in seconds for POST request (default: 15)")
    p.add_argument("--post-include-annotated", action="store_true", help="Include annotated _det.jpg in POST if available")
    p.add_argument("--post-header", action="append", default=[], help="Extra HTTP header in the form Key:Value. May be repeated")
    p.add_argument("--post-image-field", type=str, default="image", help="Multipart field name to use for the image (default: image). Example: file for FastAPI UploadFile")
    p.add_argument("--post-compat-file-field", action="store_true", help="Also send the image under field 'file' for compatibility with servers expecting that name")
    p.add_argument("--post-only-file", action="store_true", help="Send only the image as a single multipart part (no extra form fields or attachments)")
    p.add_argument("--post-score-field", type=str, default="score", help="Form field name to use for the confidence score (default: score)")
    p.add_argument("--post-objects-field", type=str, default="objects", help="Form field name to use for the detected objects list (semicolon-separated). Default: objects")

    return p.parse_args()


def main():
    args = parse_args()

    # Basic validation for callback args
    if getattr(args, "on_match_timeout", None) and not getattr(args, "on_match_sync", False):
        LOGGER.warning("--on-match-timeout has no effect without --on-match-sync")

    # Prepare POST headers dict once
    post_headers: dict[str, str] = {}
    if getattr(args, "post_header", None):
        for h in args.post_header:
            if not h:
                continue
            if ":" not in h:
                LOGGER.warning("Ignoring malformed --post-header (expected Key:Value): %s", h)
                continue
            key, val = h.split(":", 1)
            post_headers[key.strip()] = val.strip()

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
                            if getattr(args, 'post_url', None):
                                # Compute score from txt if available
                                score_lines: list[str] = []
                                try:
                                    with (txt_path if txt_path else filename.with_suffix('.txt')).open('r') as f:
                                        score_lines = [ln.strip() for ln in f if ln.strip()]
                                except Exception:
                                    score_lines = []
                                best_score = _extract_best_score_from_lines(score_lines, args.save_on)
                                _post_on_match(
                                    url=args.post_url,
                                    image=filename,
                                    txt=txt_path,
                                    annotated=annotated_path,
                                    labels=list(dict.fromkeys(labels)),
                                    timestamp=ts,
                                    save_dir=save_dir,
                                    timeout=getattr(args, 'post_timeout', 15),
                                    headers=post_headers,
                                    include_annotated=getattr(args, 'post_include_annotated', False),
                                    image_field_name=getattr(args, 'post_image_field', 'image'),
                                    duplicate_file_field=getattr(args, 'post_compat_file_field', False),
                                    only_file=getattr(args, 'post_only_file', False),
                                    score=best_score,
                                    score_field_name=getattr(args, 'post_score_field', 'score'),
                                    score_field_objects=getattr(args, 'post_score_field', 'score'),
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
                            # Prepare labels for callbacks/POST
                            labels = _extract_labels_from_lines(det_lines)
                            # On-match callback (in-memory path)
                            if args.on_match_cmd:
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
                            # Optional HTTP POST
                            if getattr(args, 'post_url', None):
                                best_score = _extract_best_score_from_lines(det_lines, args.save_on)
                                _post_on_match(
                                    url=args.post_url,
                                    image=filename,
                                    txt=txt_path,
                                    annotated=annotated_path,
                                    labels=labels,
                                    timestamp=ts,
                                    save_dir=save_dir,
                                    timeout=getattr(args, 'post_timeout', 15),
                                    headers=post_headers,
                                    include_annotated=getattr(args, 'post_include_annotated', False),
                                    image_field_name=getattr(args, 'post_image_field', 'image'),
                                    duplicate_file_field=getattr(args, 'post_compat_file_field', False),
                                    only_file=getattr(args, 'post_only_file', False),
                                    score=best_score,
                                    score_field_name=getattr(args, 'post_score_field', 'score'),
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
                        if getattr(args, 'post_url', None):
                            # Compute score from txt if available
                            score_lines: list[str] = []
                            try:
                                with (txt_path if txt_path else filename.with_suffix('.txt')).open('r') as f:
                                    score_lines = [ln.strip() for ln in f if ln.strip()]
                            except Exception:
                                score_lines = []
                            best_score = _extract_best_score_from_lines(score_lines, args.save_on)
                            _post_on_match(
                                url=args.post_url,
                                image=filename,
                                txt=txt_path,
                                annotated=annotated_path,
                                labels=list(dict.fromkeys(labels)),
                                timestamp=ts,
                                save_dir=save_dir,
                                timeout=getattr(args, 'post_timeout', 15),
                                headers=post_headers,
                                include_annotated=getattr(args, 'post_include_annotated', False),
                                image_field_name=getattr(args, 'post_image_field', 'image'),
                                duplicate_file_field=getattr(args, 'post_compat_file_field', False),
                                only_file=getattr(args, 'post_only_file', False),
                                score=best_score,
                                score_field_name=getattr(args, 'post_score_field', 'score'),
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
