# Detector class extracted from main.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("timelapse.detector")


class Detector:
    """Ultralytics-only detector.
    Provides per-image inference and optional annotation saving. Safe to disable if deps missing.

    Behavior with `save_on`:
    - If `save_on` is provided, detection also computes whether any detection matches one of the
      requested labels (case-insensitive). Use this in the caller to decide whether to keep or
      discard the captured image.

    Additionally, `process_frame(frame)` allows running detection directly on an in-memory image
    (numpy array) so the caller can decide whether to save the capture at all.
    """

    def __init__(
        self,
        enabled: bool,
        model_path: str,
        conf: float,
        save_annotated: bool,
        imgsz: int = 640,
        yolo_classes: Optional[list] = None,
        save_on: Optional[list] = None,
    ):
        self.enabled = enabled
        self.model_path = model_path
        self.conf = conf
        self.save_annotated = save_annotated
        self.imgsz = int(imgsz)
        self.yolo_classes = yolo_classes
        # Target-saving filter
        self.save_on = [str(x).strip() for x in save_on] if save_on else None
        # Canonicalize save_on terms to handle spaces, hyphens, underscores, etc.
        def _canon(s: str) -> str:
            return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())
        self._canon = _canon
        self._save_on_norm = {self._canon(x) for x in save_on} if save_on else None
        # alias support: treat plain "ball" as YOLO's "sports ball"
        if self._save_on_norm is not None and "ball" in self._save_on_norm:
            self._save_on_norm.add("sportsball")

        self._model = None
        self._names = None
        self._cv2 = None
        self._allowed_cls = None

        if not self.enabled:
            return

        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_path)
            self._names = self._model.names
            # Build allowed class id set if filtering is requested
            if self.yolo_classes:
                allowed = set()
                # names mapping: id -> name
                name_map = self._names if isinstance(self._names, dict) else {}
                # Build canonical name -> id map (alnum-only lowercase), e.g. "sports ball"/"sports-ball" → "sportsball"
                canon_name_map = {self._canon(cname): int(cid) for cid, cname in name_map.items()}
                for item in self.yolo_classes:
                    if isinstance(item, int):
                        allowed.add(int(item))
                        continue
                    s = str(item).strip()
                    # allow numeric strings
                    if s.isdigit():
                        allowed.add(int(s))
                        continue
                    s_can = self._canon(s)
                    # alias: plain "ball" → YOLO's "sports ball" if present
                    if s_can == "ball" and "sportsball" in canon_name_map:
                        allowed.add(canon_name_map["sportsball"])
                        continue
                    # match by canonicalized class name
                    if s_can in canon_name_map:
                        allowed.add(canon_name_map[s_can])
                        continue
                if allowed:
                    self._allowed_cls = allowed
            if self.save_annotated:
                import cv2  # type: ignore

                self._cv2 = cv2
        except Exception as e:
            LOGGER.warning(f"Object detection disabled (failed to init ultralytics): {e}")
            self.enabled = False

    def process_frame(self, frame) -> Optional[tuple[bool, list[str], Optional[object]]]:
        """Run detection on an in-memory image (numpy array, BGR or RGB).
        Returns (matched_target, det_lines, annotated_bgr_or_None) or None if disabled.
        - det_lines: list of strings formatted like process_image() writes to the .txt.
        - annotated_bgr_or_None: an annotated image (BGR ndarray) if save_annotated is True.
        """
        if not self.enabled:
            return None
        try:
            # use canonical normalization (lowercase alnum only)
            def norm(s: str) -> str:
                return self._canon(s)

            results = self._model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False) if self._model else None
            if results is None:
                return None
            r = results[0]
            boxes = getattr(r, 'boxes', None)
            save_set = self._save_on_norm
            default_match = True if save_set is None else False
            matched = default_match
            lines: list[str] = []
            objects: list[str] = []
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
                    objects.append(cname)
                    lines.append(f"{cname} {score:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}")
                    wrote_any = True
                    if save_set is not None:
                        if norm(str(cname)) in save_set or (str(cid) in save_set):
                            matched = True
                if not wrote_any:
                    lines.append("no_detections")
            annotated = None
            if self.save_annotated and getattr(self, "_cv2", None) is not None:
                try:
                    # If a save-on filter is provided, draw only matching detections.
                    if self._save_on_norm:
                        cv2 = self._cv2
                        try:
                            img = frame.copy()
                        except Exception:
                            # as a fallback, let Ultralytics draw everything
                            annotated = r.plot()
                            return (matched, lines, annotated, objects)
                        total = 0
                        drawn = 0
                        if boxes is not None and len(getattr(boxes, 'xyxy', [])):
                            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                            for i in range(len(xyxy)):
                                cid = int(cls[i])
                                if getattr(self, "_allowed_cls", None) is not None and cid not in self._allowed_cls:
                                    continue
                                total += 1
                                x1, y1, x2, y2 = map(int, xyxy[i])
                                score = float(conf[i])
                                cname = self._names.get(cid, str(cid)) if isinstance(self._names, dict) else str(cid)
                                if self._canon(str(cname)) in self._save_on_norm or (str(cid) in self._save_on_norm):
                                    drawn += 1
                                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label = f"{cname} {score:.2f}"
                                    cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        annotated = img
                        if total:
                            LOGGER.debug("Annotated (filtered): drew %d/%d boxes matching --save-on", drawn, total)
                    else:
                        # No filtering requested → draw all detections using Ultralytics helper
                        annotated = r.plot()
                except Exception as e:
                    LOGGER.warning(f"Failed to prepare annotated image: {e}")
            return (matched, lines, annotated, objects)
        except Exception as e:
            LOGGER.warning(f"In-memory detection failed: {e}")
            return None

    def process_image(self, image_path: Path) -> Optional[tuple[Path, bool]]:
        """Run Ultralytics detection on image_path. Write a .txt next to the image.
        Returns a tuple (txt_path, matched_target) or None if disabled.
        If `save_on` was provided at init, `matched_target` is True when any detection's label
        matches one of the requested labels (case-insensitive). When `save_on` is None, the
        second tuple element is always True (meaning "do not filter").
        """
        if not self.enabled:
            return None
        try:
            # use canonical normalization (lowercase alnum only)
            def norm(s: str) -> str:
                return self._canon(s)

            txt_path = image_path.with_suffix('.txt')
            save_set = self._save_on_norm
            # If no filtering requested, act as always and return matched=True
            default_match = True if save_set is None else False
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
                    cv2 = self._cv2
                    # If a save-on filter is provided, draw only matching detections; otherwise use Ultralytics plot.
                    if self._save_on_norm:
                        img = cv2.imread(str(image_path))
                        if img is None:
                            # Fallback to Ultralytics plot if image can't be read here
                            plotted = r.plot()
                            out_path = image_path.with_name(image_path.stem + "_det.jpg")
                            cv2.imwrite(str(out_path), plotted)
                        else:
                            total = 0
                            drawn = 0
                            if boxes is not None and len(getattr(boxes, 'xyxy', [])):
                                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else boxes.xyxy
                                conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                                cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else boxes.cls
                                for i in range(len(xyxy)):
                                    cid = int(cls[i])
                                    if getattr(self, "_allowed_cls", None) is not None and cid not in self._allowed_cls:
                                        continue
                                    total += 1
                                    x1, y1, x2, y2 = map(int, xyxy[i])
                                    score = float(conf[i])
                                    cname = self._names.get(cid, str(cid)) if isinstance(self._names, dict) else str(cid)
                                    if self._canon(str(cname)) in self._save_on_norm or (str(cid) in self._save_on_norm):
                                        drawn += 1
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        label = f"{cname} {score:.2f}"
                                        cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            out_path = image_path.with_name(image_path.stem + "_det.jpg")
                            cv2.imwrite(str(out_path), img)
                            if total:
                                LOGGER.debug("Annotated (filtered file path): drew %d/%d boxes matching --save-on", drawn, total)
                    else:
                        plotted = r.plot()  # returns a BGR numpy array
                        out_path = image_path.with_name(image_path.stem + "_det.jpg")
                        cv2.imwrite(str(out_path), plotted)
                except Exception as e:
                    LOGGER.warning(f"Failed to save annotated image: {e}")
            return (txt_path, matched)
        except Exception as e:
            LOGGER.warning(f"Detection failed on {image_path.name}: {e}")
            return None
