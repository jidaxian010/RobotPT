"""
Single-frame ArUco detection tuning tool.

Extracts frame N from a video, applies preprocessing (sharpening, contrast,
adaptive threshold, etc.), and shows detection results side-by-side so you
can tune parameters for better marker detection.

Usage:
    python posEstimate/scripts/single_frame.py
    - Adjust FRAME_N and the PREPROCESS list below, then re-run.
    - Press any key to close windows.
"""

import cv2
import numpy as np
from pathlib import Path

# ==========================================
# --- CONFIGURATION ---
# ==========================================

DATA_NAME = "P3-A2"
VIDEO_SIDE = "right"                       # "left" or "right"
FRAME_N = 393                             # which frame to grab

DATA_DIR = Path("posEstimate/data") / DATA_NAME
VIDEO_PATH = DATA_DIR / f"{DATA_NAME}_{VIDEO_SIDE}.mp4"

MARKER_SIZE_METERS = 0.0725

# ==========================================
# --- PREPROCESSING PIPELINES ---
# ==========================================
# Each entry is (label, function(bgr) -> bgr).
# All are run on the same frame so you can compare.


def preprocess_none(img):
    """Raw frame, no processing."""
    return img


def preprocess_clahe(img):
    """CLAHE on the L channel (LAB color space) for local contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_sharpen(img):
    """Unsharp-mask sharpening."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def preprocess_sharpen_strong(img):
    """Stronger unsharp mask."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=5)
    return cv2.addWeighted(img, 2.0, blur, -1.0, 0)


def preprocess_clahe_sharpen(img):
    """CLAHE then sharpen."""
    return preprocess_sharpen(preprocess_clahe(img))


def preprocess_gray_thresh(img):
    """Adaptive threshold on grayscale (returns 3-ch for display)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10
    )
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def preprocess_bilateral(img):
    """Bilateral filter (edge-preserving denoise) then sharpen."""
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return preprocess_sharpen(filtered)


def preprocess_contrast_stretch(img):
    """Linear contrast stretch — push darkest to 0, brightest to 255."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_min, l_max = l.min(), l.max()
    if l_max > l_min:
        l = ((l.astype(np.float32) - l_min) / (l_max - l_min) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_contrast_sharp(img):
    """Contrast stretch then strong sharpen."""
    return preprocess_sharpen_strong(preprocess_contrast_stretch(img))


def preprocess_gamma_bright(img):
    """Gamma correction (< 1 = brighter, makes whites pop)."""
    gamma = 0.6
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def preprocess_gamma_sharp(img):
    """Gamma brighten then strong sharpen."""
    return preprocess_sharpen_strong(preprocess_gamma_bright(img))


# Which pipelines to run — comment/uncomment to experiment
PREPROCESS = [
    ("raw",              preprocess_none),
    ("sharpen_strong",   preprocess_sharpen_strong),
    ("contrast_stretch", preprocess_contrast_stretch),
    ("contrast+sharp",   preprocess_contrast_sharp),
    ("gamma_bright",     preprocess_gamma_bright),
    ("gamma+sharp",      preprocess_gamma_sharp),
    # ("CLAHE",            preprocess_clahe),
    # ("sharpen",          preprocess_sharpen),
    # ("CLAHE+sharpen",    preprocess_clahe_sharpen),
    # ("adaptive_thresh",  preprocess_gray_thresh),
    # ("bilateral+sharp",  preprocess_bilateral),
]

# ==========================================
# --- ARUCO DETECTOR CONFIGS ---
# ==========================================
# You can also experiment with different DetectorParameters here.


def make_detector_default():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def make_detector_tuned():
    """More aggressive detection: lower thresholds, more corner refinement."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    # Adaptive thresholding
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7
    # Be more lenient with marker border
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    # Corner refinement
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01
    return cv2.aruco.ArucoDetector(dictionary, params)


# Which detector(s) to test — set to one or both
DETECTORS = [
    ("default", make_detector_default()),
    ("tuned",   make_detector_tuned()),
]

# ==========================================
# --- MAIN ---
# ==========================================


def grab_frame(video_path, frame_n):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_n >= total:
        cap.release()
        raise ValueError(f"Frame {frame_n} out of range (video has {total} frames)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_n}")
    return frame


def detect_and_draw(img, detector, label):
    """Run detection, draw results, return annotated image and detection info."""
    vis = img.copy()
    corners, ids, rejected = detector.detectMarkers(vis)

    n_detected = 0 if ids is None else len(ids)
    n_rejected = len(rejected)

    # Draw detected markers
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    # Draw rejected candidates in red
    for rej in rejected:
        pts = rej[0].astype(int)
        for j in range(4):
            cv2.line(vis, tuple(pts[j]), tuple(pts[(j + 1) % 4]), (0, 0, 255), 1)

    # Info text
    txt = f"{label}: {n_detected} det, {n_rejected} rej"
    cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if ids is not None:
        id_str = "IDs: " + ", ".join(str(int(i)) for i in ids.flatten())
        cv2.putText(vis, id_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return vis, n_detected, ids


def main():
    print(f"Video : {VIDEO_PATH}")
    print(f"Frame : {FRAME_N}")

    frame = grab_frame(VIDEO_PATH, FRAME_N)
    h, w = frame.shape[:2]
    print(f"Size  : {w}x{h}")

    # Run all combinations
    results = []
    for pp_label, pp_func in PREPROCESS:
        processed = pp_func(frame)
        for det_label, detector in DETECTORS:
            vis, n_det, ids = detect_and_draw(processed, detector, f"{pp_label} | {det_label}")
            results.append((f"{pp_label} | {det_label}", vis, n_det, ids))
            id_list = list(ids.flatten()) if ids is not None else []
            print(f"  {pp_label:20s} + {det_label:10s} → {n_det} markers {id_list}")

    # Display results in windows
    for label, vis, _, _ in results:
        # Scale down for display if too large
        scale = min(1.0, 1200.0 / w)
        if scale < 1.0:
            vis = cv2.resize(vis, None, fx=scale, fy=scale)
        cv2.imshow(label, vis)

    print("\nPress any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
