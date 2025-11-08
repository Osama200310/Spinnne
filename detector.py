# Detector class extracted from main.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("timelapse.detector")


class Detector:
    """Detection backends: 'opencv-motion', 'opencv-dnn-ssd', 'opencv-ball', 'ultralytics'.
    Provides per-image inference and optional annotation saving. Safe to disable if deps missing.

    New behavior: if `save_on` is provided, detection will also compute whether any detection
    matches one of the requested labels (case-insensitive). Use this in the caller to decide
    whether to keep or discard the captured image.

    Additionally, for the Ultralytics backend, `process_frame(frame)` allows running detection
    directly on an in-memory image (numpy array) so the caller can decide whether to save the
    capture at all. This avoids writing and then deleting files when using `--save-on`.
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
        save_on: Optional[list] = None,
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
        # Target-saving filter
        self.save_on = [str(x).strip() for x in save_on] if save_on else None
        self._save_on_norm = {str(x).strip().lower() for x in save_on} if save_on else None
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
                    "background",
                    "aeroplane",
                    "bicycle",
                    "bird",
                    "boat",
                    "bottle",
                    "bus",
                    "car",
                    "cat",
                    "chair",
                    "cow",
                    "diningtable",
                    "dog",
                    "horse",
                    "motorbike",
                    "person",
                    "pottedplant",
                    "sheep",
                    "sofa",
                    "train",
                    "tvmonitor",
                ]
            elif self.detector == "opencv-ball":
                import cv2  # type: ignore

                self._cv2 = cv2
            else:
                raise RuntimeError(f"Unknown detector '{self.detector}'")
        except Exception as e:
            LOGGER.warning(f"Object detection disabled (failed to init {self.detector}): {e}")
            self.enabled = False

    def process_frame(self, frame) -> Optional[tuple[bool, list[str], Optional[object]]]:
        """Ultralytics-only: run detection on an in-memory image (numpy array, BGR or RGB).
        Returns (matched_target, det_lines, annotated_bgr_or_None) or None if disabled or backend unsupported.
        - det_lines: list of strings formatted like process_image() writes to the .txt.
        - annotated_bgr_or_None: an annotated image (BGR ndarray) if save_annotated is True.
        """
        if not self.enabled or self.detector != "ultralytics":
            return None
        try:
            def norm(s: str) -> str:
                return s.strip().lower().replace("_", " ")

            results = self._model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False) if self._model else None
            if results is None:
                return None
            r = results[0]
            boxes = getattr(r, 'boxes', None)
            save_set = self._save_on_norm
            default_match = True if save_set is None else False
            matched = default_match
            lines: list[str] = []
            if boxes is None or len(boxes) == 0:
                lines.append("no_detections")
            else:
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
                    lines.append(f"{cname} {score:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
                    wrote_any = True
                    if save_set is not None:
                        #LOGGER.warning(cname)
                        if norm(str(cname)) in save_set or (str(cid) in save_set):
                            matched = True
                if not wrote_any:
                    lines.append("no_detections")
            annotated = None
            if self.save_annotated and getattr(self, "_cv2", None) is not None:
                try:
                    annotated = r.plot()
                except Exception as e:
                    LOGGER.warning(f"Failed to prepare annotated image: {e}")
            return (matched, lines, annotated)
        except Exception as e:
            LOGGER.warning(f"In-memory detection failed: {e}")
            return None

    def process_image(self, image_path: Path) -> Optional[tuple[Path, bool]]:
        """Run detection on image_path. Write a .txt next to the image.
        Returns a tuple (txt_path, matched_target) or None if disabled.
        If `save_on` was provided at init, `matched_target` is True when any detection's label
        matches one of the requested labels (case-insensitive). When `save_on` is None, the
        second tuple element is always True (meaning "do not filter").
        """
        if not self.enabled:
            return None
        try:
            def norm(s: str) -> str:
                return s.strip().lower().replace("_", " ")

            txt_path = image_path.with_suffix('.txt')
            save_set = self._save_on_norm
            # If no filtering requested, act as always and return matched=True
            default_match = True if save_set is None else False
            if self.detector == "ultralytics":
                if self._model is None:
                    return None
                results = self._model(str(image_path), conf=self.conf, imgsz=self.imgsz, verbose=False)
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                matched = default_match
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
                            if save_set is not None:
                                if norm(str(cname)) in save_set or (str(cid) in save_set):
                                    matched = True
                        if not wrote_any:
                            f.write("no_detections\n")
                if self.save_annotated and self._cv2 is not None:
                    try:
                        plotted = r.plot()  # returns a BGR numpy array
                        out_path = image_path.with_name(image_path.stem + "_det.jpg")
                        self._cv2.imwrite(str(out_path), plotted)
                    except Exception as e:
                        LOGGER.warning(f"Failed to save annotated image: {e}")
                return (txt_path, matched)

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
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
                )
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                detections = []
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h < self.min_area:
                        continue
                    detections.append((x, y, x + w, y + h))
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
                        LOGGER.warning(f"Failed to save annotated image: {e}")
                # Matching logic: label is 'motion' when any detection exists
                matched = default_match
                if save_set is not None:
                    if 'motion' in save_set and len(detections) > 0:
                        matched = True
                return (txt_path, matched)

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
                import numpy as np  # type: ignore

                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence < self.ssd_conf:
                        continue
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype("int")
                    cname = (
                        self._dnn_classes[idx]
                        if self._dnn_classes and 0 <= idx < len(self._dnn_classes)
                        else str(idx)
                    )
                    results.append((cname, confidence, int(x1), int(y1), int(x2), int(y2)))
                matched = default_match
                with txt_path.open('w') as f:
                    if not results:
                        f.write("no_detections\n")
                    else:
                        for (cname, score, x1, y1, x2, y2) in results:
                            f.write(
                                f"{cname} {score:.3f} {float(x1):.1f} {float(y1):.1f} {float(x2):.1f} {float(y2):.1f}\n"
                            )
                            if save_set is not None:
                                if norm(str(cname)) in save_set:
                                    matched = True
                if self.save_annotated and cv2 is not None:
                    try:
                        out_img = img.copy()
                        for (cname, score, x1, y1, x2, y2) in results:
                            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{cname}:{score:.2f}"
                            cv2.putText(
                                out_img,
                                label,
                                (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                        out_path = image_path.with_name(image_path.stem + "_det.jpg")
                        cv2.imwrite(str(out_path), out_img)
                    except Exception as e:
                        LOGGER.warning(f"Failed to save annotated image: {e}")
                return (txt_path, matched)

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
                    maxRadius=maxR,
                )
                detections = []
                if circles is not None:
                    circles = circles[0, :]
                    inv = 1.0 / scale if scale != 0 else 1.0
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
                            inv = 1.0 / scale if scale != 0 else 1.0
                            for (x, y, r) in circles:
                                cx = int(round(float(x) * inv))
                                cy = int(round(float(y) * inv))
                                rr = int(round(float(r) * inv))
                                cv2.circle(out_img, (cx, cy), rr, (0, 255, 255), 2)
                                cv2.rectangle(
                                    out_img, (max(0, cx - rr), max(0, cy - rr)), (cx + rr, cy + rr), (0, 255, 255), 1
                                )
                        out_path = image_path.with_name(image_path.stem + "_det.jpg")
                        cv2.imwrite(str(out_path), out_img)
                    except Exception as e:
                        LOGGER.warning(f"Failed to save annotated image: {e}")
                matched = default_match
                if save_set is not None:
                    if 'ball' in save_set and len(detections) > 0:
                        matched = True
                return (txt_path, matched)

            else:
                return None
        except Exception as e:
            LOGGER.warning(f"Detection failed on {image_path.name}: {e}")
            return None
