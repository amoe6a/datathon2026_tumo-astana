"""Local video decoding; timestamps come from the decoder, not model guesses."""
import base64
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Frame:
    time: float
    bgr: np.ndarray

def sample_video(path, fps=2.0, max_duration=180, width=640):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Sampling FPS must be positive")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot decode video. Use a valid MOV or MP4 file.")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite(source_fps) or source_fps <= 0 or count <= 0:
            raise ValueError("Invalid video timing metadata")
        duration = count / source_fps
        if not np.isfinite(duration) or duration > max_duration:
            raise ValueError(f"Video exceeds {max_duration} seconds")
        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 3840 * 2160:
            raise ValueError("Video resolution exceeds the 4K processing limit")
        frames, next_at, idx, previous_time = [], 0.0, 0, -1.0
        last_frame = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t = decoded if np.isfinite(decoded) and decoded > previous_time else idx / source_fps
            if t <= previous_time:
                raise ValueError("Nonmonotonic video timestamps")
            previous_time = t
            last_frame = frame
            if t > max_duration:
                raise ValueError("Video exceeds duration limit")
            if t + 1e-6 >= next_at:
                h, w = frame.shape[:2]
                if w > width:
                    frame = cv2.resize(frame, (width, round(h * width / w)))
                frames.append(Frame(round(t, 3), frame))
                next_at = t + 1 / fps
            idx += 1
        if not frames:
            raise ValueError("Video contains no decodable frames")
        if idx < int(count):
            raise ValueError("Video appears truncated; analysis was not completed")
        # Preserve the final real frame even when it falls between sample ticks.
        if round(previous_time, 3) > frames[-1].time:
            h, w = last_frame.shape[:2]
            if w > width:
                last_frame = cv2.resize(last_frame, (width, round(h * width / w)))
            frames.append(Frame(round(previous_time, 3), last_frame))
        return frames, max(duration, previous_time)
    finally:
        cap.release()

def jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame.bgr, [cv2.IMWRITE_JPEG_QUALITY, 84])
    if not ok:
        raise ValueError("Evidence encoding failed")
    return {"timestamp_sec": frame.time, "image_base64": base64.b64encode(buf).decode("ascii")}

def features(frames):
    """Small appearance + motion baseline. Not viewpoint invariant or semantic."""
    result, previous = [], None
    for f in frames:
        small = cv2.resize(f.bgr, (32, 24)).astype(np.float32) / 255
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        motion = np.zeros_like(gray) if previous is None else np.abs(gray - previous)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        result.append(np.concatenate([small.ravel(), motion.ravel(), dx.ravel(), dy.ravel()]))
        previous = gray
    return np.asarray(result, dtype=np.float32)
