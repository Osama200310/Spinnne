#!/usr/bin/env python3
from picamera2 import Picamera2
from datetime import datetime
from pathlib import Path
import time
import signal
import argparse
from typing import Optional

# Lazy imports for detection (ultralytics / cv2) are inside Detector

# Configure where to save and the interval
SAVE_DIR = Path("/home/admin/timelapse")  # change this path if you like
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
    global running
    print("\nStopping timelapse…")
    running = False


# we should extrude the provided classes and clean up the code

class Detector:
    """Detection backends: 'opencv-motion', 'opencv-dnn-ssd', 'opencv-ball', 'ultralytics'.
    Provides per-image inference and optional annotation saving. Safe to disable if deps missing.
    """
    def __init__(
        self,
        enabled: bool,
        model_path: str,
        conf: float,
        save_annotated: bool,
        imgsz: int = 640,
        detector: str = "opencv-motion",
        min_area: int = 500,
        ssd_prototxt: Optional[Path] = None,
        ssd_weights: Optional[Path] = None,
        ssd_conf: float = 0.4,
        yolo_classes: Optional[list] = None,
        # OpenCV-ball parameters
        ball_dp: float = 1.2,
        ball_min_dist: int = 20,
        ball_canny: int = 100,
        ball_accum: int = 30,
        ball_min_radius: int = 5,
        ball_max_radius: int = 80,
        ball_resize: float = 1.0,
        ball_hsv_lower: Optional[tuple] = None,
        ball_hsv_upper: Optional[tuple] = None,
    ):
        self.enabled = enabled
        self.detector = detector
        self.model_path = model_path
        self.conf = conf
        self.save_annotated = save_annotated
        self.imgsz = int(imgsz)
        self.min_area = int(min_area)
        self.ssd_prototxt = str(ssd_prototxt) if ssd_prototxt else None
        self.ssd_weights = str(ssd_weights) if ssd_weights else None
        self.ssd_conf = float(ssd_conf)
        self.yolo_classes = yolo_classes
        # Ball params
        self.ball_dp = float(ball_dp)
        self.ball_min_dist = int(ball_min_dist)
        self.ball_canny = int(ball_canny)
        self.ball_accum = int(ball_accum)
        self.ball_min_radius = max(0, int(ball_min_radius))
        self.ball_max_radius = max(0, int(ball_max_radius))
        self.ball_resize = max(0.1, float(ball_resize))
        self.ball_hsv_lower = tuple(ball_hsv_lower) if ball_hsv_lower else None
        self.ball_hsv_upper = tuple(ball_hsv_upper) if ball_hsv_upper else None

        self._model = None
        self._names = None
        self._cv2 = None
        self._bg = None  # background subtractor for motion
        self._dnn = None  # OpenCV DNN net
        self._dnn_classes = None

        if not self.enabled:
            return

        try:
            if self.detector == "ultralytics":
                from ultralytics import YOLO  # type: ignore
                self._model = YOLO(self.model_path)
                self._names = self._model.names
                # Build allowed class id set if filtering is requested
                self._allowed_cls = None
                if self.yolo_classes:
                    allowed = set()
                    # names mapping: id -> name
                    name_map = self._names if isinstance(self._names, dict) else {}
                    def norm(s: str) -> str:
                        return s.strip().lower().replace("_", " ")
                    # Add common alias mapping for sports ball
                    alias_map = {"ball": "sports ball"}
                    for item in self.yolo_classes:
                        if isinstance(item, int):
                            allowed.add(int(item))
                            continue
                        s = str(item)
                        if s.isdigit():
                            allowed.add(int(s))
                            continue
                        s_norm = norm(s)
                        if s_norm in alias_map:
                            s_norm = alias_map[s_norm]
                        # find id by name
                        for cid, cname in name_map.items():
                            if norm(str(cname)) == s_norm:
                                allowed.add(int(cid))
                                break
                    if allowed:
                        self._allowed_cls = allowed
                if self.save_annotated:
                    import cv2  # type: ignore
                    self._cv2 = cv2
            elif self.detector == "opencv-motion":
                import cv2  # type: ignore
                self._cv2 = cv2
                # history/varThreshold are tunable; detectShadows True by default
                self._bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
            elif self.detector == "opencv-dnn-ssd":
                import cv2  # type: ignore
                self._cv2 = cv2
                if not (self.ssd_prototxt and self.ssd_weights):
                    raise RuntimeError("SSD detector requires --ssd-prototxt and --ssd-weights paths")
                net = cv2.dnn.readNetFromCaffe(self.ssd_prototxt, self.ssd_weights)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self._dnn = net
                # PASCAL VOC 20 classes
                self._dnn_classes = [
                    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
                ]
            elif self.detector == "opencv-ball":
                import cv2  # type: ignore
                self._cv2 = cv2
            else:
                raise RuntimeError(f"Unknown detector '{self.detector}'")
        except Exception as e:
            print(f"[WARN] Object detection disabled (failed to init {self.detector}): {e}")
            self.enabled = False

    def process_image(self, image_path: Path) -> Optional[Path]:
        """Run detection on image_path. Write a .txt next to the image.
        Returns path to the written txt, or None if disabled.
        """
        if not self.enabled:
            return None
        try:
            txt_path = image_path.with_suffix('.txt')
            if self.detector == "ultralytics":
                if self._model is None:
                    return None
                results = self._model(str(image_path), conf=self.conf, imgsz=self.imgsz, verbose=False)
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                with txt_path.open('w') as f:
                    if boxes is None or len(boxes) == 0:
                        f.write("no_detections\n")
                    else:
                        import numpy as np  # ensure numpy present
                        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                        wrote_any = False
                        for i in range(len(xyxy)):
                            cid = int(cls[i])
                            if getattr(self, "_allowed_cls", None) is not None and cid not in self._allowed_cls:
                                continue
                            x1, y1, x2, y2 = map(float, xyxy[i])
                            score = float(conf[i])
                            cname = self._names.get(cid, str(cid)) if isinstance(self._names, dict) else str(cid)
                            f.write(f"{cname} {score:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
                            wrote_any = True
                        if not wrote_any:
                            f.write("no_detections\n")
                if self.save_annotated and self._cv2 is not None:
                    try:
                        plotted = r.plot()  # returns a BGR numpy array
                        out_path = image_path.with_name(image_path.stem + "_det.jpg")
                        self._cv2.imwrite(str(out_path), plotted)
                    except Exception as e:
                        print(f"[WARN] Failed to save annotated image: {e}")
                return txt_path

            elif self.detector == "opencv-motion":
                cv2 = self._cv2
                if cv2 is None or self._bg is None:
                    return None
                img = cv2.imread(str(image_path))
                if img is None:
                    return None
                mask = self._bg.apply(img)
                # Threshold and morphology to clean
                _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                detections = []
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h < self.min_area:
                        continue
                    detections.append((x, y, x+w, y+h))
                with txt_path.open('w') as f:
                    if not detections:
                        f.write("no_detections\n")
                    else:
                        for (x1, y1, x2, y2) in detections:
                            f.write(f"motion 0.000 {float(x1):.1f} {float(y1):.1f} {float(x2):.1f} {float(y2):.1f}\n")
                if self.save_annotated and cv2 is not None:
                    try:
                        out_img = img.copy()
                        for (x1, y1, x2, y2) in detections:
                            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        out_path = image_path.with_name(image_path.stem + "_det.jpg")
                        cv2.imwrite(str(out_path), out_img)
                    except Exception as e:
                        print(f"[WARN] Failed to save annotated image: {e}")
                return txt_path

            elif self.detector == "opencv-dnn-ssd":
                cv2 = self._cv2
                net = self._dnn
                if cv2 is None or net is None:
                    return None
                img = cv2.imread(str(image_path))
                if img is None:
                    return None
                (h, w) = img.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()
                results = []
                import numpy as np
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence < self.ssd_conf:
                        continue
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    cname = self._dnn_classes[idx] if self._dnn_classes and 0 <= idx < len(self._dnn_classes) else str(idx)
                    results.append((cname, confidence, int(x1), int(y1), int(x2), int(y2)))
                with txt_path.open('w') as f:
                    if not results:
                        f.write("no_detections\n")
                    else:
                        for (cname, score, x1, y1, x2, y2) in results:
                            f.write(f"{cname} {score:.3f} {float(x1):.1f} {float(y1):.1f} {float(x2):.1f} {float(y2):.1f}\n")
                if self.save_annotated and cv2 is not None:
                    try:
                        out_img = img.copy()
                        for (cname, score, x1, y1, x2, y2) in results:
                            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{cname}:{score:.2f}"
                            cv2.putText(out_img, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                        out_path = image_path.with_name(image_path.stem + "_det.jpg")
                        cv2.imwrite(str(out_path), out_img)
                    except Exception as e:
                        print(f"[WARN] Failed to save annotated image: {e}")
                return txt_path

            elif self.detector == "opencv-ball":
                cv2 = self._cv2
                if cv2 is None:
                    return None
                img = cv2.imread(str(image_path))
                if img is None:
                    return None
                # Optionally resize for speed
                scale = 1.0
                proc = img
                if self.ball_resize and abs(self.ball_resize - 1.0) > 1e-3:
                    scale = float(self.ball_resize)
                    proc = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                # Prepare grayscale
                gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                gray = cv2.medianBlur(gray, 5)
                # Optional HSV mask to isolate ball color
                if self.ball_hsv_lower is not None and self.ball_hsv_upper is not None:
                    hsv = cv2.cvtColor(proc, cv2.COLOR_BGR2HSV)
                    lower = self.ball_hsv_lower
                    upper = self.ball_hsv_upper
                    mask = cv2.inRange(hsv, lower, upper)
                    gray = cv2.bitwise_and(gray, gray, mask=mask)
                # Hough Circle detection
                minDist = max(1, self.ball_min_dist)
                minR = max(0, int(self.ball_min_radius * scale))
                maxR = 0 if self.ball_max_radius <= 0 else int(self.ball_max_radius * scale)
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=self.ball_dp,
                    minDist=minDist,
                    param1=self.ball_canny,
                    param2=self.ball_accum,
                    minRadius=minR,
                    maxRadius=maxR
                )
                detections = []
                if circles is not None:
                    circles = circles[0, :]
                    inv = 1.0/scale if scale != 0 else 1.0
                    for (x, y, r) in circles:
                        cx = float(x) * inv
                        cy = float(y) * inv
                        rr = float(r) * inv
                        x1 = max(0.0, cx - rr)
                        y1 = max(0.0, cy - rr)
                        x2 = cx + rr
                        y2 = cy + rr
                        detections.append((x1, y1, x2, y2))
                # Write results
                with txt_path.open('w') as f:
                    if not detections:
                        f.write("no_detections\n")
                    else:
                        for (x1, y1, x2, y2) in detections:
                            f.write(f"ball 1.000 {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
                if self.save_annotated and cv2 is not None:
                    try:
                        out_img = img.copy()
                        if circles is not None:
                            inv = 1.0/scale if scale != 0 else 1.0
                            for (x, y, r) in circles:
                                cx = int(round(float(x) * inv))
                                cy = int(round(float(y) * inv))
                                rr = int(round(float(r) * inv))
                                cv2.circle(out_img, (cx, cy), rr, (0, 255, 255), 2)
                                cv2.rectangle(out_img, (max(0, cx-rr), max(0, cy-rr)), (cx+rr, cy+rr), (0, 255, 255), 1)
                        out_path = image_path.with_name(image_path.stem + "_det.jpg")
                        cv2.imwrite(str(out_path), out_img)
                    except Exception as e:
                        print(f"[WARN] Failed to save annotated image: {e}")
                return txt_path

            else:
                return None
        except Exception as e:
            print(f"[WARN] Detection failed on {image_path.name}: {e}")
            return None


def apply_exposure_controls(picam2, args):
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

    picam2.set_controls(controls)


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

    return p.parse_args()


def main():
    args = parse_args()

    save_dir = args.save_dir
    interval = args.interval

    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare detector (may disable itself if deps are missing)
    detector = Detector(
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
        print("[INFO] Continuing without detection. Install dependencies per README to enable it.")

    picam2 = Picamera2()
    # Simple still configuration; adjust resolution if needed
    config = picam2.create_still_configuration()

    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)

   # picam2.configure(config)
    picam2.start()

    # Apply requested exposure settings
    apply_exposure_controls(picam2, args)

    print(f"Saving one image every {interval}s to {save_dir} (Ctrl+C to stop)")
    if not args.ae:
        print("Manual exposure enabled → Shutter: {} us{}".format(
            args.shutter_us,
            f", ISO≈{args.iso}" if args.iso is not None else ""
        ))
    if detector.enabled:
        if args.detector == 'ultralytics':
            print(f"Detection: ON → backend=ultralytics, model={args.model}, conf={args.conf}, imgsz={args.imgsz}")
        elif args.detector == 'opencv-motion':
            print(f"Detection: ON → backend=opencv-motion, min_area={args.min_area}")
        elif args.detector == 'opencv-dnn-ssd':
            print(f"Detection: ON → backend=opencv-dnn-ssd, conf={args.ssd_conf}")
        elif args.detector == 'opencv-ball':
            hsv = None
            if args.ball_hsv_lower and args.ball_hsv_upper:
                hsv = f"HSV={tuple(args.ball_hsv_lower)}..{tuple(args.ball_hsv_upper)}"
            print("Detection: ON → backend=opencv-ball, "
                  f"dp={args.ball_dp}, min_dist={args.ball_min_dist}, canny={args.ball_canny}, accum={args.ball_accum}, "
                  f"minR={args.ball_min_radius}, maxR={args.ball_max_radius}, resize={args.ball_resize}"
                  + (f", {hsv}" if hsv else ""))
        if args.save_annotated:
            print("Annotated images will be saved with suffix _det.jpg")
    else:
        print("Object detection: OFF")

    while running:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = save_dir / f"image_{ts}.jpg"
        picam2.capture_file(str(filename))
        print(f"Saved {filename}")

        # Run detection and write sidecar .txt (and optional _det.jpg)
        if detector.enabled:
            txt = detector.process_image(filename)
            if txt is not None:
                print(f"Detections written to {txt.name}")

        # Sleep after capture so the first image is immediate
        for _ in range(interval):
            if not running:
                break
            time.sleep(1)

    picam2.stop()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    try:
        main()
    except KeyboardInterrupt:
        pass
