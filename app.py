# advanced_flying_object_detector.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import time
from datetime import datetime
import os
from collections import deque, defaultdict
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')

# Advanced computer vision imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available. Install: pip install ultralytics")

try:
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available. Install: pip install scipy")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

@dataclass
class TrackedObject:
    """Data structure for tracked flying objects"""
    id: int
    class_name: str
    confidence_history: List[float] = field(default_factory=list)
    positions_history: List[Tuple[int, int]] = field(default_factory=list)
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    total_frames: int = 0
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    estimated_next_position: Optional[Tuple[int, int]] = None
    motion_pattern: str = "unknown"  # linear, circular, erratic, hovering
    speed: float = 0.0
    altitude_estimate: float = 0.0  # Relative altitude estimate
    size_history: List[int] = field(default_factory=list)  # Bounding box area
    classification_confidence: Dict[str, float] = field(default_factory=dict)
    spectral_features: Optional[np.ndarray] = None

class AdvancedFlyingDetector:
    def __init__(self):
        # Multiple detection models
        self.yolo_model = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=True
        )
        self.optical_flow = None
        self.prev_gray = None

        # Detection parameters
        self.confidence_threshold = 0.4
        self.nms_threshold = 0.45
        self.frame_skip = 1
        self.frame_count = 0

        # Advanced tracking
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 0
        self.trail_length = 60
        self.object_trails = {}

        # Motion detection
        self.motion_regions = []
        self.motion_history = deque(maxlen=30)

        # Kalman filters for each tracked object
        self.kalman_filters = {}

        # Object classification with multiple features
        self.flying_categories = {
            'airplane': {'min_speed': 50, 'max_speed': 250, 'typical_size': (100, 300)},
            'drone': {'min_speed': 10, 'max_speed': 70, 'typical_size': (30, 100)},
            'bird': {'min_speed': 5, 'max_speed': 40, 'typical_size': (15, 50)},
            'helicopter': {'min_speed': 0, 'max_speed': 80, 'typical_size': (80, 200)},
            'kite': {'min_speed': 5, 'max_speed': 30, 'typical_size': (40, 120)},
            'balloon': {'min_speed': 0, 'max_speed': 20, 'typical_size': (20, 80)},
            'insect': {'min_speed': 2, 'max_speed': 15, 'typical_size': (5, 20)},
            'satellite': {'min_speed': 200, 'max_speed': 800, 'typical_size': (5, 15)},
            'ufo_unknown': {'min_speed': 0, 'max_speed': 1000, 'typical_size': (10, 500)}
        }

        # Enhanced object classes for YOLO
        self.flying_classes = {
            'airplane', 'bird', 'kite', 'drone', 'helicopter',
            'balloon', 'insect', 'satellite', 'butterfly', 'dragonfly',
            'bat', 'eagle', 'hawk', 'seagull', 'pigeon', 'crow'
        }

        # Detection history for pattern analysis
        self.detection_history = []
        self.anomaly_detector = None

        # Performance metrics
        self.detection_stats = {
            'total_detections': 0,
            'unique_objects': 0,
            'false_positives': 0,
            'tracking_accuracy': 0.0
        }

        # Initialize YOLO
        self.load_models()

        # Initialize anomaly detection
        if SKLEARN_AVAILABLE:
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

    def load_models(self):
        """Load multiple detection models"""
        if YOLO_AVAILABLE:
            try:
                print("Loading YOLOv8 model for flying object detection...")
                self.yolo_model = YOLO('yolov8x.pt')  # Using larger model for better accuracy
                print("YOLO model loaded successfully!")
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                self.yolo_model = None

    def detect_motion(self, frame):
        """Detect motion using background subtraction and optical flow"""
        height, width = frame.shape[:2]

        # Background subtraction
        fgmask = self.background_subtractor.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)

        # Find contours of moving objects
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, x + w, y + h))

        # Optical flow for fine-grained motion detection
        if self.prev_gray is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Analyze flow vectors to detect small flying objects
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Find regions with significant motion but small area
            mask = mag > 2
            if np.any(mask):
                flow_contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in flow_contours:
                    area = cv2.contourArea(contour)
                    if 20 < area < 500:  # Small moving objects
                        x, y, w, h = cv2.boundingRect(contour)
                        motion_regions.append((x, y, x + w, y + h))

            self.prev_gray = gray
        else:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Merge overlapping regions
        motion_regions = self.merge_overlapping_regions(motion_regions)

        return motion_regions

    def detect_with_yolo(self, frame):
        """Detect objects using YOLO"""
        detections = []

        if self.yolo_model is None:
            return detections

        try:
            results = self.yolo_model(frame, conf=self.confidence_threshold,
                                     iou=self.nms_threshold, verbose=False)

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = self.get_class_name(class_id)
                        confidence = float(box.conf[0])

                        # Check if it's a flying object or potential flying object
                        if self.is_flying_object(class_name):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'center': (int((x1 + x2)/2), int((y1 + y2)/2)),
                                'class': class_name,
                                'confidence': confidence,
                                'source': 'yolo',
                                'timestamp': datetime.now()
                            })
        except Exception as e:
            print(f"YOLO detection error: {e}")

        return detections

    def detect_edges_and_shapes(self, frame):
        """Detect objects based on edge detection and shape analysis"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Reasonable size for flying objects
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)

                # Analyze shape
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Classify based on shape
                if circularity > 0.7:
                    object_type = "round_flying_object"
                elif circularity > 0.4:
                    object_type = "irregular_flying_object"
                else:
                    object_type = "elongated_flying_object"

                detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'center': (x + w//2, y + h//2),
                    'class': object_type,
                    'confidence': 0.6,
                    'source': 'edge_detection',
                    'timestamp': datetime.now(),
                    'shape_features': {'circularity': circularity, 'area': area}
                })

        return detections

    def detect_with_tracking(self, frame, previous_detections):
        """Use previous tracking information to predict and detect"""
        detections = []

        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.estimated_next_position:
                # Create search region around estimated position
                x, y = tracked_obj.estimated_next_position
                search_radius = 50
                x1 = max(0, x - search_radius)
                y1 = max(0, y - search_radius)
                x2 = min(frame.shape[1], x + search_radius)
                y2 = min(frame.shape[0], y + search_radius)

                # Extract region and enhance for detection
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    # Use template matching or feature matching
                    # For simplicity, we'll use color histogram comparison
                    if len(tracked_obj.positions_history) > 5:
                        # Calculate expected size
                        avg_size = np.mean(tracked_obj.size_history[-10:]) if tracked_obj.size_history else 100
                        size = int(np.sqrt(avg_size))

                        detections.append({
                            'bbox': (x - size//2, y - size//2, x + size//2, y + size//2),
                            'center': (x, y),
                            'class': tracked_obj.class_name,
                            'confidence': tracked_obj.classification_confidence.get(tracked_obj.class_name, 0.7),
                            'source': 'tracking_prediction',
                            'timestamp': datetime.now(),
                            'object_id': obj_id
                        })

        return detections

    def merge_detections(self, all_detections):
        """Merge detections from multiple sources and remove duplicates"""
        if not all_detections:
            return []

        # Sort by confidence
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)

        merged = []
        used = [False] * len(all_detections)

        for i, det in enumerate(all_detections):
            if used[i]:
                continue

            # Get the highest confidence detection
            current_best = det.copy()
            bbox1 = det['bbox']
            center1 = det['center']

            # Look for overlapping detections
            overlapping = []
            for j, other_det in enumerate(all_detections[i+1:], i+1):
                if used[j]:
                    continue

                bbox2 = other_det['bbox']
                # Calculate IoU
                intersection = self.calculate_iou(bbox1, bbox2)

                if intersection > 0.3:  # Significant overlap
                    overlapping.append(other_det)
                    used[j] = True

                    # Update confidence
                    current_best['confidence'] = max(current_best['confidence'], other_det['confidence'])

                    # Merge classifications
                    if other_det['confidence'] > current_best['confidence'] * 0.8:
                        current_best['class'] = other_det['class']

            merged.append(current_best)

        return merged

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def merge_overlapping_regions(self, regions):
        """Merge overlapping bounding boxes"""
        if not regions:
            return []

        merged = []
        regions = sorted(regions, key=lambda x: (x[0], x[1]))

        current = list(regions[0])
        for region in regions[1:]:
            if (region[0] <= current[2] and region[1] <= current[3]):
                # Overlap, merge
                current[0] = min(current[0], region[0])
                current[1] = min(current[1], region[1])
                current[2] = max(current[2], region[2])
                current[3] = max(current[3], region[3])
            else:
                merged.append(tuple(current))
                current = list(region)

        merged.append(tuple(current))
        return merged

    def advanced_tracking(self, frame, detections):
        """Advanced multi-object tracking with Kalman filters and feature matching"""
        if not detections:
            # Update existing tracks without new detections
            for obj_id in list(self.tracked_objects.keys()):
                self.update_track_without_detection(obj_id, frame)
            return []

        # Prepare detection matrix
        detection_centers = np.array([d['center'] for d in detections])

        if len(self.tracked_objects) > 0:
            # Get predicted positions from Kalman filters
            predicted_positions = []
            for obj_id, tracked_obj in self.tracked_objects.items():
                if obj_id in self.kalman_filters:
                    prediction = self.kalman_filters[obj_id].predict()
                    predicted_positions.append((prediction[0], prediction[1]))
                else:
                    predicted_positions.append(tracked_obj.estimated_next_position or tracked_obj.positions_history[-1])

            predicted_positions = np.array(predicted_positions)

            # Calculate cost matrix
            cost_matrix = cdist(detection_centers, predicted_positions)

            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Update matched tracks
            matched_pairs = []
            for detection_idx, track_idx in zip(row_ind, col_ind):
                if cost_matrix[detection_idx, track_idx] < 100:  # Distance threshold
                    matched_pairs.append((detection_idx, track_idx))

            # Update matched objects
            for detection_idx, track_idx in matched_pairs:
                obj_id = list(self.tracked_objects.keys())[track_idx]
                self.update_tracked_object(obj_id, detections[detection_idx], frame)

            # Create new tracks for unmatched detections
            matched_detection_indices = set([p[0] for p in matched_pairs])
            for i, detection in enumerate(detections):
                if i not in matched_detection_indices:
                    self.create_new_track(detection, frame)

            # Update unmatched tracks
            matched_track_indices = set([p[1] for p in matched_pairs])
            for i, obj_id in enumerate(list(self.tracked_objects.keys())):
                if i not in matched_track_indices:
                    self.update_track_without_detection(obj_id, frame)

        else:
            # Create new tracks for all detections
            for detection in detections:
                self.create_new_track(detection, frame)

        # Remove old tracks
        self.cleanup_old_tracks()

        # Analyze motion patterns
        for obj_id, tracked_obj in self.tracked_objects.items():
            self.analyze_motion_pattern(obj_id, tracked_obj)
            self.classify_flying_object(obj_id, tracked_obj)

        # Convert tracked objects to detection format for display
        current_detections = []
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.positions_history:
                current_detections.append({
                    'id': obj_id,
                    'center': tracked_obj.positions_history[-1],
                    'class': tracked_obj.class_name,
                    'confidence': np.mean(tracked_obj.confidence_history[-5:]) if tracked_obj.confidence_history else 0.5,
                    'speed': tracked_obj.speed,
                    'motion_pattern': tracked_obj.motion_pattern,
                    'bbox': tracked_obj.bbox_history[-1] if tracked_obj.bbox_history else None
                })

        return current_detections

    def create_kalman_filter(self):
        """Create Kalman filter for tracking"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                           [0, 1, 0, 1],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32) * 0.03
        return kalman

    def create_new_track(self, detection, frame):
        """Create a new tracked object"""
        obj_id = self.next_object_id
        self.next_object_id += 1

        tracked_obj = TrackedObject(
            id=obj_id,
            class_name=detection['class'],
            confidence_history=[detection['confidence']],
            positions_history=[detection['center']],
            bbox_history=[detection['bbox']],
            last_seen=datetime.now()
        )

        # Calculate size
        bbox = detection['bbox']
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        tracked_obj.size_history.append(size)

        # Create Kalman filter
        self.kalman_filters[obj_id] = self.create_kalman_filter()
        kalman = self.kalman_filters[obj_id]
        kalman.statePre = np.array([[detection['center'][0]],
                                   [detection['center'][1]],
                                   [0], [0]], np.float32)

        self.tracked_objects[obj_id] = tracked_obj
        self.detection_stats['unique_objects'] += 1

        # Log new object
        print(f"New object tracked: ID={obj_id}, Class={detection['class']}, "
              f"Confidence={detection['confidence']:.2f}")

    def update_tracked_object(self, obj_id, detection, frame):
        """Update existing tracked object with new detection"""
        if obj_id not in self.tracked_objects:
            return

        tracked_obj = self.tracked_objects[obj_id]

        # Update history
        tracked_obj.confidence_history.append(detection['confidence'])
        tracked_obj.positions_history.append(detection['center'])
        tracked_obj.bbox_history.append(detection['bbox'])
        tracked_obj.last_seen = datetime.now()
        tracked_obj.total_frames += 1

        # Keep history limited
        max_history = 100
        if len(tracked_obj.confidence_history) > max_history:
            tracked_obj.confidence_history = tracked_obj.confidence_history[-max_history:]
        if len(tracked_obj.positions_history) > max_history:
            tracked_obj.positions_history = tracked_obj.positions_history[-max_history:]
        if len(tracked_obj.bbox_history) > max_history:
            tracked_obj.bbox_history = tracked_obj.bbox_history[-max_history:]

        # Calculate size
        bbox = detection['bbox']
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        tracked_obj.size_history.append(size)
        if len(tracked_obj.size_history) > 50:
            tracked_obj.size_history = tracked_obj.size_history[-50:]

        # Calculate velocity and acceleration
        if len(tracked_obj.positions_history) >= 2:
            prev_pos = tracked_obj.positions_history[-2]
            curr_pos = tracked_obj.positions_history[-1]
            distance = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))
            tracked_obj.velocities.append(distance)
            tracked_obj.speed = np.mean(tracked_obj.velocities[-10:])

            if len(tracked_obj.velocities) >= 2:
                acc = tracked_obj.velocities[-1] - tracked_obj.velocities[-2]
                tracked_obj.accelerations.append(acc)

        # Update Kalman filter
        if obj_id in self.kalman_filters:
            kalman = self.kalman_filters[obj_id]
            kalman.correct(np.array([[detection['center'][0]],
                                    [detection['center'][1]]], np.float32))
            prediction = kalman.predict()
            tracked_obj.estimated_next_position = (int(prediction[0]), int(prediction[1]))

    def update_track_without_detection(self, obj_id, frame):
        """Update track when no detection is available"""
        if obj_id not in self.tracked_objects:
            return

        tracked_obj = self.tracked_objects[obj_id]

        # Use Kalman filter prediction
        if obj_id in self.kalman_filters:
            kalman = self.kalman_filters[obj_id]
            prediction = kalman.predict()
            predicted_pos = (int(prediction[0]), int(prediction[1]))

            tracked_obj.estimated_next_position = predicted_pos
            tracked_obj.positions_history.append(predicted_pos)

            # Decrease confidence
            if tracked_obj.confidence_history:
                new_confidence = tracked_obj.confidence_history[-1] * 0.9
                tracked_obj.confidence_history.append(new_confidence)

    def cleanup_old_tracks(self):
        """Remove tracks that haven't been seen recently"""
        current_time = datetime.now()
        to_remove = []

        for obj_id, tracked_obj in self.tracked_objects.items():
            time_since_seen = (current_time - tracked_obj.last_seen).total_seconds()
            if time_since_seen > 3.0:  # Remove after 3 seconds
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            if obj_id in self.kalman_filters:
                del self.kalman_filters[obj_id]

    def analyze_motion_pattern(self, obj_id, tracked_obj):
        """Analyze the motion pattern of tracked object"""
        if len(tracked_obj.positions_history) < 20:
            return

        positions = np.array(tracked_obj.positions_history)

        # Calculate trajectory characteristics
        diffs = np.diff(positions, axis=0)
        directions = np.arctan2(diffs[:, 1], diffs[:, 0])
        direction_changes = np.abs(np.diff(directions))

        # Calculate curvature
        if len(positions) >= 3:
            vectors = positions[1:] - positions[:-1]
            angles = []
            for i in range(len(vectors) - 1):
                angle = np.arccos(np.clip(
                    np.dot(vectors[i], vectors[i+1]) /
                    (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i+1]) + 1e-6),
                    -1, 1
                ))
                angles.append(angle)

            mean_curvature = np.mean(angles) if angles else 0
        else:
            mean_curvature = 0

        # Classify motion pattern
        speed_std = np.std(tracked_obj.velocities) if len(tracked_obj.velocities) > 1 else 0
        avg_speed = tracked_obj.speed

        if avg_speed < 2:
            tracked_obj.motion_pattern = "hovering"
        elif direction_changes.mean() < 0.5 and speed_std < 2:
            tracked_obj.motion_pattern = "linear"
        elif mean_curvature > 1:
            tracked_obj.motion_pattern = "circular"
        elif speed_std > avg_speed * 0.5:
            tracked_obj.motion_pattern = "erratic"
        else:
            tracked_obj.motion_pattern = "smooth_curved"

    def classify_flying_object(self, obj_id, tracked_obj):
        """Advanced classification using multiple features"""
        avg_speed = tracked_obj.speed
        avg_size = np.mean(tracked_obj.size_history) if tracked_obj.size_history else 0
        motion_pattern = tracked_obj.motion_pattern

        # Calculate confidence scores for each type
        scores = {}

        for obj_type, features in self.flying_categories.items():
            speed_score = 1.0
            if features['min_speed'] <= avg_speed <= features['max_speed']:
                speed_score = 1.0
            elif avg_speed < features['min_speed']:
                speed_score = max(0, avg_speed / features['min_speed'])
            else:
                speed_score = max(0, features['max_speed'] / avg_speed)

            size_score = 1.0
            if features['typical_size'][0] <= avg_size <= features['typical_size'][1]:
                size_score = 1.0
            elif avg_size < features['typical_size'][0]:
                size_score = avg_size / features['typical_size'][0]
            else:
                size_score = features['typical_size'][1] / avg_size

            # Motion pattern matching
            motion_score = 1.0
            if obj_type == 'drone':
                motion_score = 1.0 if motion_pattern in ['hovering', 'erratic'] else 0.5
            elif obj_type == 'airplane':
                motion_score = 1.0 if motion_pattern == 'linear' else 0.3
            elif obj_type == 'bird':
                motion_score = 1.0 if motion_pattern in ['erratic', 'smooth_curved'] else 0.6

            total_score = (speed_score * 0.4 + size_score * 0.3 + motion_score * 0.3)
            scores[obj_type] = total_score

        # Get best match
        if scores:
            best_match = max(scores, key=scores.get)
            if scores[best_match] > 0.6:
                tracked_obj.class_name = best_match
                tracked_obj.classification_confidence = scores

    def is_flying_object(self, class_name):
        """Determine if a class name corresponds to a flying object"""
        class_lower = class_name.lower()
        return any(flying_class in class_lower for flying_class in self.flying_classes)

    def get_class_name(self, class_id):
        """Get class name from COCO class ID"""
        coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        return coco_classes.get(class_id, 'unknown')

    def detect_flying_objects(self, frame):
        """Main detection method combining all techniques"""
        # Multi-source detection
        motion_regions = self.detect_motion(frame)
        yolo_detections = self.detect_with_yolo(frame)
        edge_detections = self.detect_edges_and_shapes(frame)
        tracking_detections = self.detect_with_tracking(frame, [])

        # Combine all detections
        all_detections = yolo_detections + edge_detections + tracking_detections

        # Convert motion regions to detection format
        for region in motion_regions:
            x1, y1, x2, y2 = region
            center = ((x1 + x2)//2, (y1 + y2)//2)
            all_detections.append({
                'bbox': (x1, y1, x2, y2),
                'center': center,
                'class': 'moving_object',
                'confidence': 0.7,
                'source': 'motion',
                'timestamp': datetime.now()
            })

        # Merge detections
        merged_detections = self.merge_detections(all_detections)

        # Advanced tracking
        tracked_objects = self.advanced_tracking(frame, merged_detections)

        # Update statistics
        self.detection_stats['total_detections'] += len(tracked_objects)

        return tracked_objects

class EnhancedDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Flying Object Detection & Tracking System")
        self.root.geometry("1600x900")

        # Variables
        self.camera = None
        self.is_detecting = False
        self.detection_thread = None
        self.video_source = 0
        self.current_frame = None
        self.detector = AdvancedFlyingDetector()

        # Performance monitoring
        self.fps = 0
        self.last_fps_update = time.time()
        self.frame_count_fps = 0

        # Detection display options
        self.show_trails = tk.BooleanVar(value=True)
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_motion_paths = tk.BooleanVar(value=True)
        self.show_heatmap = tk.BooleanVar(value=False)
        self.show_speed_vectors = tk.BooleanVar(value=True)

        # Setup UI
        self.setup_ui()

        # Initialize camera
        self.init_camera()

    def setup_ui(self):
        """Setup the enhanced user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Title
        title_label = ttk.Label(left_panel, text="Advanced Flying Object Tracker",
                                font=('Arial', 18, 'bold'))
        title_label.pack(pady=10)

        # Stats frame
        stats_frame = ttk.LabelFrame(left_panel, text="Real-time Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=10)

        self.stats_text = tk.Text(stats_frame, height=10, width=35, font=('Courier', 10))
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Controls frame
        controls_frame = ttk.LabelFrame(left_panel, text="Detection Controls", padding=10)
        controls_frame.pack(fill=tk.X, pady=10)

        # Source selection
        ttk.Label(controls_frame, text="Video Source:").pack(anchor=tk.W)
        self.source_var = tk.StringVar(value="webcam")
        source_combo = ttk.Combobox(controls_frame, textvariable=self.source_var,
                                   values=["webcam", "file", "ip_camera"])
        source_combo.pack(fill=tk.X, pady=5)
        source_combo.bind('<<ComboboxSelected>>', self.on_source_change)

        # File path (hidden initially)
        self.file_frame = ttk.Frame(controls_frame)
        ttk.Label(self.file_frame, text="Video File:").pack(anchor=tk.W)
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(self.file_frame, textvariable=self.file_path_var)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        browse_btn = ttk.Button(self.file_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # IP Camera frame
        self.ip_frame = ttk.Frame(controls_frame)
        ttk.Label(self.ip_frame, text="RTSP URL:").pack(anchor=tk.W)
        self.ip_url_var = tk.StringVar()
        ip_entry = ttk.Entry(self.ip_frame, textvariable=self.ip_url_var)
        ip_entry.pack(fill=tk.X, pady=5)

        # Detection sensitivity
        ttk.Label(controls_frame, text="Detection Sensitivity:").pack(anchor=tk.W, pady=(10, 0))
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_scale = ttk.Scale(controls_frame, from_=0.1, to=0.9,
                                     variable=self.sensitivity_var,
                                     command=self.on_sensitivity_change)
        sensitivity_scale.pack(fill=tk.X)

        # Display options
        display_frame = ttk.LabelFrame(left_panel, text="Display Options", padding=10)
        display_frame.pack(fill=tk.X, pady=10)

        ttk.Checkbutton(display_frame, text="Show Trajectory Trails",
                       variable=self.show_trails).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Bounding Boxes",
                       variable=self.show_boxes).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Labels & IDs",
                       variable=self.show_labels).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Motion Paths",
                       variable=self.show_motion_paths).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Speed Vectors",
                       variable=self.show_speed_vectors).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Show Detection Heatmap",
                       variable=self.show_heatmap).pack(anchor=tk.W)

        # Control buttons
        btn_frame = ttk.Frame(controls_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(btn_frame, text="Start Detection",
                                    command=self.start_detection)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="Stop Detection",
                                   command=self.stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="Export Log", command=self.export_log).pack(side=tk.LEFT, padx=5)

        # Detection log
        log_frame = ttk.LabelFrame(left_panel, text="Detection Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.log_text = ScrolledText(log_frame, height=12, width=40, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        clear_btn = ttk.Button(log_frame, text="Clear Log", command=self.clear_log)
        clear_btn.pack(pady=5)

        # Right panel - Video display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Video canvas
        self.video_canvas = tk.Canvas(right_panel, bg='black')
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_bar = ttk.Label(self.root, text="System Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def init_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.log_message("Error: Could not open webcam", "ERROR")
                self.status_bar.config(text="Error: Could not open webcam")
            else:
                self.log_message("Webcam initialized successfully")
                self.status_bar.config(text="Webcam ready - Click Start Detection")
        except Exception as e:
            self.log_message(f"Error initializing camera: {e}", "ERROR")

    def browse_file(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
                      ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            self.log_message(f"Selected video: {os.path.basename(filename)}")

    def on_source_change(self, event=None):
        """Handle video source change"""
        source = self.source_var.get()

        # Hide all source-specific frames first
        self.file_frame.pack_forget()
        self.ip_frame.pack_forget()

        if source == "file":
            self.file_frame.pack(fill=tk.X, pady=(5, 0))
        elif source == "ip_camera":
            self.ip_frame.pack(fill=tk.X, pady=(5, 0))

    def on_sensitivity_change(self, value):
        """Update detection sensitivity"""
        sensitivity = float(value)
        self.detector.confidence_threshold = 1.0 - sensitivity
        self.log_message(f"Detection sensitivity set to {sensitivity:.2f}")

    def start_detection(self):
        """Start object detection"""
        if self.is_detecting:
            return

        # Set video source
        source = self.source_var.get()

        # Release existing camera
        if self.camera:
            self.camera.release()

        try:
            if source == "webcam":
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    raise Exception("Could not open webcam")
            elif source == "file":
                file_path = self.file_path_var.get()
                if not file_path or not os.path.exists(file_path):
                    self.log_message("Please select a valid video file", "ERROR")
                    return
                self.camera = cv2.VideoCapture(file_path)
                if not self.camera.isOpened():
                    raise Exception("Could not open video file")
            elif source == "ip_camera":
                ip_url = self.ip_url_var.get()
                if not ip_url:
                    self.log_message("Please enter RTSP URL", "ERROR")
                    return
                self.camera = cv2.VideoCapture(ip_url)
                if not self.camera.isOpened():
                    raise Exception("Could not connect to IP camera")

            self.is_detecting = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.log_message(f"Detection started - Source: {source}")
            self.status_bar.config(text="Detection running...")

            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()

        except Exception as e:
            self.log_message(f"Error starting detection: {e}", "ERROR")
            self.status_bar.config(text=f"Error: {str(e)}")

    def stop_detection(self):
        """Stop object detection"""
        self.is_detecting = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log_message("Detection stopped")
        self.status_bar.config(text="Detection stopped")

    def detection_loop(self):
        """Main detection loop"""
        self.frame_count_fps = 0
        self.last_fps_update = time.time()

        while self.is_detecting:
            if self.camera is None or not self.camera.isOpened():
                self.log_message("Camera not available", "ERROR")
                break

            ret, frame = self.camera.read()
            if not ret:
                if self.source_var.get() == "file":
                    # Loop video
                    self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # Calculate FPS
            self.frame_count_fps += 1
            current_time = time.time()
            if current_time - self.last_fps_update >= 1.0:
                self.fps = self.frame_count_fps
                self.frame_count_fps = 0
                self.last_fps_update = current_time

            # Detect flying objects
            detected_objects = self.detector.detect_flying_objects(frame)

            # Draw enhanced visualizations
            display_frame = self.draw_enhanced_detections(frame, detected_objects)

            # Update display
            self.update_video_display(display_frame)

            # Update statistics
            self.update_statistics(detected_objects)

            # Log new objects
            for obj in detected_objects:
                self.log_object_detection(obj)

            # Small delay to control frame rate
            time.sleep(0.01)

    def draw_enhanced_detections(self, frame, objects):
        """Draw advanced visualizations on frame"""
        display_frame = frame.copy()

        # Draw detection heatmap if enabled
        if self.show_heatmap.get() and len(self.detector.detection_history) > 0:
            heatmap = np.zeros_like(frame[:,:,0], dtype=np.float32)
            for detection in self.detector.detection_history[-100:]:
                if 'center' in detection:
                    x, y = detection['center']
                    if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
                        heatmap[y, x] += 1
            heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
            display_frame = cv2.addWeighted(display_frame, 0.7, heatmap_colored, 0.3, 0)

        # Draw motion paths
        if self.show_motion_paths.get():
            for obj_id, trail in self.detector.object_trails.items():
                if len(trail) > 5:
                    points = [tuple(p) for p in trail]
                    for i in range(1, len(points)):
                        alpha = i / len(points)
                        color = (0, int(200 * alpha), int(255 * (1 - alpha)))
                        cv2.line(display_frame, points[i-1], points[i], color, 2)

        # Draw each tracked object
        for obj in objects:
            obj_id = obj.get('id', -1)
            center = obj['center']
            class_name = obj['class']
            confidence = obj['confidence']
            speed = obj.get('speed', 0)
            motion_pattern = obj.get('motion_pattern', 'unknown')

            # Color based on object type
            color_map = {
                'airplane': (0, 255, 0),
                'drone': (0, 165, 255),
                'bird': (255, 100, 0),
                'helicopter': (255, 255, 0),
                'kite': (255, 0, 255),
                'balloon': (0, 255, 255),
                'insect': (100, 255, 100),
                'default': (255, 255, 255)
            }
            color = color_map.get(class_name, color_map['default'])

            # Draw bounding box if available
            if self.show_boxes.get() and 'bbox' in obj and obj['bbox']:
                bbox = obj['bbox']
                cv2.rectangle(display_frame, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]), color, 2)

            # Draw speed vector
            if self.show_speed_vectors.get() and len(self.detector.tracked_objects.get(obj_id, {}).positions_history) > 3:
                tracked_obj = self.detector.tracked_objects.get(obj_id)
                if tracked_obj and len(tracked_obj.positions_history) >= 2:
                    prev_pos = tracked_obj.positions_history[-2]
                    if prev_pos:
                        vector = (center[0] - prev_pos[0], center[1] - prev_pos[1])
                        vector_norm = np.linalg.norm(vector)
                        if vector_norm > 0:
                            vector = (int(vector[0] * 5 / vector_norm),
                                     int(vector[1] * 5 / vector_norm))
                            end_point = (center[0] + vector[0], center[1] + vector[1])
                            cv2.arrowedLine(display_frame, center, end_point, color, 2)

            # Draw trajectory trail
            if self.show_trails.get():
                if obj_id in self.detector.object_trails:
                    trail = self.detector.object_trails[obj_id]
                    for i in range(1, len(trail)):
                        alpha = i / len(trail)
                        trail_color = (int(color[0] * alpha),
                                      int(color[1] * alpha),
                                      int(color[2] * alpha))
                        cv2.line(display_frame, trail[i-1], trail[i], trail_color, 1)

            # Draw label
            if self.show_labels.get():
                label = f"ID:{obj_id} {class_name.upper()} {confidence:.2f}"
                if speed > 0:
                    label += f" {speed:.1f}px/f"
                label += f" [{motion_pattern}]"

                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                label_y = max(center[1] - 10, label_size[1] + 5)
                label_x = center[0] - label_size[0] // 2

                # Draw background rectangle
                cv2.rectangle(display_frame,
                            (label_x - 2, label_y - label_size[1] - 2),
                            (label_x + label_size[0] + 2, label_y + 2),
                            color, -1)
                cv2.putText(display_frame, label,
                          (label_x, label_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                # Draw center point
                cv2.circle(display_frame, center, 3, color, -1)

        # Draw FPS and stats
        info_y = 30
        cv2.putText(display_frame, f"FPS: {self.fps}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Objects: {len(objects)}", (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Total Tracked: {len(self.detector.tracked_objects)}",
                   (10, info_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return display_frame

    def update_video_display(self, frame):
        """Update the video display canvas"""
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            frame = cv2.resize(frame, (canvas_width, canvas_height))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_canvas.image = imgtk

    def update_statistics(self, objects):
        """Update statistics display"""
        stats = f"{'='*35}\n"
        stats += f"DETECTION STATISTICS\n"
        stats += f"{'='*35}\n\n"
        stats += f"Active Objects: {len(objects)}\n"
        stats += f"Total Tracked: {len(self.detector.tracked_objects)}\n"
        stats += f"Total Detections: {self.detector.detection_stats['total_detections']}\n"
        stats += f"Unique Objects: {self.detector.detection_stats['unique_objects']}\n"
        stats += f"FPS: {self.fps}\n\n"

        if objects:
            stats += f"{'='*35}\n"
            stats += f"CURRENT OBJECTS\n"
            stats += f"{'='*35}\n\n"
            for obj in objects[:5]:  # Show top 5
                stats += f"ID {obj['id']}: {obj['class']}\n"
                stats += f"  Conf: {obj['confidence']:.2f}\n"
                if 'speed' in obj:
                    stats += f"  Speed: {obj['speed']:.1f} px/f\n"
                if 'motion_pattern' in obj:
                    stats += f"  Motion: {obj['motion_pattern']}\n"
                stats += "\n"

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)

    def log_object_detection(self, obj):
        """Log object detection to text widget"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] ID:{obj['id']} | {obj['class'].upper()} | "
        log_entry += f"Conf:{obj['confidence']:.2f} | Motion:{obj.get('motion_pattern', 'N/A')}"
        if 'speed' in obj:
            log_entry += f" | Speed:{obj['speed']:.1f}"

        # Only log new detections or significant changes
        if not hasattr(self, '_last_logged_obj'):
            self._last_logged_obj = {}

        obj_key = f"{obj['id']}"
        current_time = time.time()

        if obj_key not in self._last_logged_obj or \
           current_time - self._last_logged_obj[obj_key] > 2:  # Log every 2 seconds
            self.log_message(log_entry)
            self._last_logged_obj[obj_key] = current_time

    def log_message(self, message, level="INFO"):
        """Add message to log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] [{level}] {message}\n"

        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)

        if level == "ERROR":
            self.status_bar.config(text=f"Error: {message}")

    def clear_log(self):
        """Clear the log text widget"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Log cleared")

    def export_log(self):
        """Export detection log to file"""
        filename = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(self.log_text.get(1.0, tk.END))
        self.log_message(f"Log exported to {filename}")

    def on_closing(self):
        """Handle application closing"""
        self.stop_detection()
        if self.camera:
            self.camera.release()
        self.root.destroy()

def main():
    """Main entry point"""
    root = tk.Tk()

    # Check for required packages
    if not YOLO_AVAILABLE:
        messagebox.showwarning("Missing Dependency",
                             "YOLO not installed.\nInstall with: pip install ultralytics\n\n"
                             "The app will still work with motion detection.")

    if not SCIPY_AVAILABLE:
        messagebox.showwarning("Missing Dependency",
                             "SciPy not installed.\nInstall with: pip install scipy\n\n"
                             "Tracking may be less accurate.")

    app = EnhancedDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
