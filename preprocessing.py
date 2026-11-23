"""
Multi-Modal Forensic Preprocessing Pipeline
Handles:
  1. Face ROI localization & extraction (256x256 RGB float32 in [0,1])
  2. Ocular Region (Eyes) ROI extraction (96x96 RGB float32 in [0,1])
  3. 2D-DCT Frequency Spectrum Extraction (128x128 float32 4x4 block DCT map)
  4. Laplacian Sharpness Metric & Blur Diagnostic Guard
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOCAL_CASCADES_DIR = BASE_DIR / "haarcascades"


def _find_cascade(filename: str) -> Optional[str]:
    # 1. Prioritize local repository bundled directory
    local_p = LOCAL_CASCADES_DIR / filename
    if local_p.is_file():
        return str(local_p)
    # 2. Check cv2.data if available
    try:
        cv_data = getattr(cv2, "data", None)
        if cv_data is not None and hasattr(cv_data, "haarcascades"):
            cv_p = Path(cv_data.haarcascades) / filename
            if cv_p.is_file():
                return str(cv_p)
    except Exception:
        pass
    # 3. Standard Linux system directories
    for sys_dir in ["/usr/share/opencv4/haarcascades", "/usr/share/opencv/haarcascades"]:
        sys_p = Path(sys_dir) / filename
        if sys_p.is_file():
            return str(sys_p)
    return None


def _load_cascade(filename: str) -> Optional[cv2.CascadeClassifier]:
    p = _find_cascade(filename)
    if p:
        try:
            clf = cv2.CascadeClassifier(p)
            if not clf.empty():
                return clf
        except Exception:
            pass
    return None


face_cascade = _load_cascade("haarcascade_frontalface_default.xml")
eye_cascade = _load_cascade("haarcascade_eye.xml")


def detect_and_crop_face(
    image_bgr: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    margin_ratio: float = 0.15
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Detects the primary frontal face in an image and crops with padding.
    Falls back to center crop if no face is detected or classifier is unavailable.
    Returns:
        cropped_face_bgr (H, W, 3) uint8
        bbox (x, y, w, h) in original image coordinates or None
    """
    h_orig, w_orig = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = ()
    if face_cascade is not None and not face_cascade.empty():
        try:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        except Exception:
            faces = ()

    if len(faces) > 0:
        # Choose the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # Apply proportional margin
        margin_x = int(w * margin_ratio)
        margin_y = int(h * margin_ratio)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_orig, x + w + margin_x)
        y2 = min(h_orig, y + h + margin_y)

        cropped = image_bgr[y1:y2, x1:x2]
        bbox = (x, y, w, h)
    else:
        # Fallback to square center crop
        min_dim = min(h_orig, w_orig)
        start_x = (w_orig - min_dim) // 2
        start_y = (h_orig - min_dim) // 2
        cropped = image_bgr[start_y:start_y + min_dim, start_x:start_x + min_dim]
        bbox = (start_x, start_y, min_dim, min_dim)

    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return resized, bbox


def extract_eyes_region(
    face_bgr_256: np.ndarray,
    target_size: Tuple[int, int] = (96, 96)
) -> np.ndarray:
    """
    Extracts the ocular (both eyes) region from a 256x256 face crop.
    Uses Haar eye cascade if eyes detected, else uses anatomical band heuristic
    (vertical 20%-52%, horizontal 10%-90%).
    Returns:
        eyes_bgr (96, 96, 3) uint8
    """
    h, w = face_bgr_256.shape[:2]
    gray = cv2.cvtColor(face_bgr_256, cv2.COLOR_BGR2GRAY)
    
    # Restrict eye search to upper 60% of face
    upper_gray = gray[:int(h * 0.6), :]
    detected_eyes = ()
    if eye_cascade is not None and not eye_cascade.empty():
        try:
            detected_eyes = eye_cascade.detectMultiScale(
                upper_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )
        except Exception:
            detected_eyes = ()

    if len(detected_eyes) >= 2:
        # Sort left-to-right
        detected_eyes = sorted(detected_eyes, key=lambda e: e[0])
        e1, e2 = detected_eyes[0], detected_eyes[-1]
        
        # Bounding box spanning both eyes with margin
        x_min = max(0, min(e1[0], e2[0]) - 10)
        x_max = min(w, max(e1[0] + e1[2], e2[0] + e2[2]) + 10)
        y_min = max(0, min(e1[1], e2[1]) - 10)
        y_max = min(h, max(e1[1] + e1[3], e2[1] + e2[3]) + 10)
        
        eyes_crop = face_bgr_256[y_min:y_max, x_min:x_max]
    else:
        # Robust anatomical ocular band: Y: 20% to 52%, X: 10% to 90%
        y1 = int(h * 0.20)
        y2 = int(h * 0.52)
        x1 = int(w * 0.10)
        x2 = int(w * 0.90)
        eyes_crop = face_bgr_256[y1:y2, x1:x2]

    if eyes_crop.size == 0:
        eyes_crop = face_bgr_256[:int(h * 0.5), :]

    eyes_resized = cv2.resize(eyes_crop, target_size, interpolation=cv2.INTER_AREA)
    return eyes_resized


def compute_2d_dct_map(
    face_bgr_256: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
    block_size: int = 4
) -> np.ndarray:
    """
    Computes 2D-DCT spectrum map (128x128 float32) across 4x4 blocks.
    Matches the frequency-domain feature representation utilized by the DCTBranch.
    """
    gray = cv2.cvtColor(face_bgr_256, cv2.COLOR_BGR2GRAY)
    g128 = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    
    h, w = target_size
    dct_map = np.zeros((h, w), dtype=np.float32)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = g128[i:i + block_size, j:j + block_size]
            dct_map[i:i + block_size, j:j + block_size] = cv2.dct(block)
            
    return dct_map


def compute_laplacian_sharpness(
    image_bgr: np.ndarray,
    blur_threshold: float = 65.0,
    mild_threshold: float = 110.0
) -> Dict[str, Any]:
    """
    Computes Laplacian variance sharpness metric for blur diagnosis.
    Returns:
        variance: float
        edge_map: np.ndarray (for visualization)
        status: "SHARP" | "MILD_BLUR" | "SEVERE_BLUR"
        alert_triggered: bool
        alert_message: str
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    variance = float(laplacian.var())
    
    # Absolute normalized edge map for display
    edge_map = np.uint8(np.clip(np.abs(laplacian) / (np.max(np.abs(laplacian)) + 1e-6) * 255.0, 0, 255))
    
    if variance < blur_threshold:
        status = "SEVERE_BLUR"
        alert = True
        msg = (
            f"⚠️ SEVERE BLUR DETECTED (Laplacian Var: {variance:.1f} < {blur_threshold}): "
            "High-frequency attenuation is severe. The DCT frequency branch attention weight "
            "may be distorted by optical or compression blur mimicking deepfake frequency anomalies."
        )
    elif variance < mild_threshold:
        status = "MILD_BLUR"
        alert = True
        msg = (
            f"ℹ️ MILD BLUR DETECTED (Laplacian Var: {variance:.1f}): "
            "Moderate sharpness loss detected. Spatial face & eye branches should be prioritized over DCT."
        )
    else:
        status = "SHARP"
        alert = False
        msg = f"✅ OPTIMAL SHARPNESS (Laplacian Var: {variance:.1f} >= {mild_threshold}): High-frequency detail is intact."
        
    return {
        "variance": variance,
        "edge_map": edge_map,
        "status": status,
        "alert_triggered": alert,
        "alert_message": msg,
    }


def preprocess_frame(
    frame_bgr: np.ndarray,
    blur_threshold: float = 65.0
) -> Dict[str, Any]:
    """
    Complete end-to-end preprocessing pipeline for a single raw image or video frame.
    Outputs model input PyTorch tensors and display matrices.
    """
    # 1. Face Detection & Crop
    face_bgr_256, bbox = detect_and_crop_face(frame_bgr, target_size=(256, 256))
    face_rgb_256 = cv2.cvtColor(face_bgr_256, cv2.COLOR_BGR2RGB)
    
    # 2. Eye Region Crop
    eyes_bgr_96 = extract_eyes_region(face_bgr_256, target_size=(96, 96))
    eyes_rgb_96 = cv2.cvtColor(eyes_bgr_96, cv2.COLOR_BGR2RGB)
    
    # 3. 2D-DCT Frequency Map
    dct_map_128 = compute_2d_dct_map(face_bgr_256, target_size=(128, 128), block_size=4)
    
    # 4. Laplacian Sharpness Diagnostic
    blur_info = compute_laplacian_sharpness(face_bgr_256, blur_threshold=blur_threshold)
    
    # 5. Format PyTorch Tensors
    # Face: (1, 3, 256, 256) float32 in [0, 1]
    face_tensor = torch.from_numpy(face_rgb_256.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    
    # Eyes: (1, 3, 96, 96) float32 in [0, 1]
    eyes_tensor = torch.from_numpy(eyes_rgb_96.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    
    # DCT: (1, 1, 128, 128) float32
    dct_tensor = torch.from_numpy(dct_map_128.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    return {
        "face_tensor": face_tensor,
        "eyes_tensor": eyes_tensor,
        "dct_tensor": dct_tensor,
        "face_rgb": face_rgb_256,
        "eyes_rgb": eyes_rgb_96,
        "dct_map": dct_map_128,
        "bbox": bbox,
        "blur_info": blur_info,
    }
