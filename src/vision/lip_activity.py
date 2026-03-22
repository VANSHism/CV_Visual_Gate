import math
import os
import urllib.request
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python_tasks
from mediapipe.tasks.python import vision as mp_vision_tasks

from src.config import VisionConfig


UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263
DEFAULT_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class LipActivityResult:
    is_speaking: bool
    normalized_lip_gap: float
    smoothed_lip_gap: float


class LipActivityDetector:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._mode = "solutions" if hasattr(mp, "solutions") else "tasks"
        self._mesh = None
        self._landmarker = None
        if self._mode == "solutions":
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mesh = self._mp_face_mesh.FaceMesh(
                max_num_faces=config.max_faces,
                refine_landmarks=False,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
        else:
            model_path = self._ensure_face_landmarker_model()
            base_options = mp_python_tasks.BaseOptions(model_asset_path=model_path)
            options = mp_vision_tasks.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision_tasks.RunningMode.IMAGE,
                num_faces=config.max_faces,
                min_face_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
            )
            self._landmarker = mp_vision_tasks.FaceLandmarker.create_from_options(options)
        self._smoothed = 0.0
        self._is_speaking = False

    @staticmethod
    def _ensure_face_landmarker_model() -> str:
        model_dir = os.path.join(os.getcwd(), ".models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "face_landmarker.task")
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(DEFAULT_FACE_LANDMARKER_MODEL_URL, model_path)
        return model_path

    @staticmethod
    def _distance(p1, p2) -> float:
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def process_frame(self, bgr_frame) -> LipActivityResult | None:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if self._mode == "solutions":
            result = self._mesh.process(rgb)
            if not result.multi_face_landmarks:
                self._is_speaking = False
                self._smoothed *= 0.95
                return LipActivityResult(
                    is_speaking=False,
                    normalized_lip_gap=0.0,
                    smoothed_lip_gap=self._smoothed,
                )
            landmarks = result.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)
            if not result.face_landmarks:
                self._is_speaking = False
                self._smoothed *= 0.95
                return LipActivityResult(
                    is_speaking=False,
                    normalized_lip_gap=0.0,
                    smoothed_lip_gap=self._smoothed,
                )
            landmarks = result.face_landmarks[0]
        upper = landmarks[UPPER_LIP_IDX]
        lower = landmarks[LOWER_LIP_IDX]
        left_eye = landmarks[LEFT_EYE_OUTER_IDX]
        right_eye = landmarks[RIGHT_EYE_OUTER_IDX]

        lip_gap = self._distance(upper, lower)
        eye_dist = max(self._distance(left_eye, right_eye), 1e-6)
        normalized = lip_gap / eye_dist

        alpha = self.config.smoothing_alpha
        self._smoothed = (1.0 - alpha) * self._smoothed + alpha * normalized

        if self._is_speaking:
            self._is_speaking = self._smoothed > self.config.speech_close_threshold
        else:
            self._is_speaking = self._smoothed > self.config.speech_open_threshold

        return LipActivityResult(
            is_speaking=self._is_speaking,
            normalized_lip_gap=normalized,
            smoothed_lip_gap=self._smoothed,
        )

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()
