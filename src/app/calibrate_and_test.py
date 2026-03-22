import statistics
import time

import cv2

from src.config import AppConfig
from src.vision.lip_activity import LipActivityDetector


def run_closed_mouth_calibration(sample_seconds: float = 4.0) -> None:
    config = AppConfig()
    detector = LipActivityDetector(config.vision)
    cap = cv2.VideoCapture(config.vision.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera.")

    print("Calibration started. Keep your mouth closed and face camera.")
    values: list[float] = []
    start = time.time()
    try:
        while time.time() - start < sample_seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            result = detector.process_frame(frame)
            if result is not None:
                values.append(result.smoothed_lip_gap)
                cv2.putText(
                    frame,
                    f"Closed-mouth sample: {result.smoothed_lip_gap:.4f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

    if not values:
        print("No face samples captured.")
        return

    baseline = statistics.median(values)
    open_threshold = baseline * 1.35
    close_threshold = baseline * 1.18
    print(f"Closed-mouth baseline: {baseline:.5f}")
    print(f"Suggested speech_open_threshold: {open_threshold:.5f}")
    print(f"Suggested speech_close_threshold: {close_threshold:.5f}")
    print("Now test speaking and silence behavior in main.py with these values.")
    print("")
    print("Manual validation checklist:")
    print("1) Stay silent with mouth closed -> gate gain should remain low.")
    print("2) Speak clearly -> gate gain should rise smoothly, no hard clipping artifacts.")
    print("3) Background chatter with your mouth closed -> gate should avoid opening.")
    print("4) Pause between phrases -> noise profile update should become allowed.")
    print("5) Move closer/farther from camera -> normalized lip gap should remain stable.")


if __name__ == "__main__":
    run_closed_mouth_calibration()
