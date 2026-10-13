"""Shared face helpers. register_face.py, recognize.py and main.py all use these,
so the faces saved for training and the faces seen at recognition time are
processed in exactly the same way (this matters a lot for LBPH accuracy)."""
import cv2, os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CASCADE = os.path.join(BASE, "models", "haarcascade_frontalface_default.xml")
FACE_SIZE = (200, 200)

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_detector():
    detector = cv2.CascadeClassifier(CASCADE)
    if detector.empty():
        raise SystemExit(f"Cascade failed to load: {CASCADE}")
    return detector


# Detection filters (tuned for higher strictness to reject false positives)
MIN_CONF = 2.5       # Increased confidence requirement for Haar Cascade
MIN_STD = 45         # Increased pixel contrast threshold (rejects flat walls/curtains)
RELAXED_CONF = 1.5   # Looser confidence used by register_face.py

# (rect, confidence, contrast, accepted) for every candidate in the latest call
last_candidates = []


EYE_CASCADE = os.path.join(BASE, "models", "haarcascade_eye.xml")
_eye_detector = None
_eye_checked = False


def _get_eye_detector():
    global _eye_detector, _eye_checked
    if not _eye_checked:
        _eye_checked = True
        if os.path.exists(EYE_CASCADE):
            det = cv2.CascadeClassifier(EYE_CASCADE)
            if not det.empty():
                _eye_detector = det
        if _eye_detector is None:
            print("Eye cascade not found (models/haarcascade_eye.xml): "
                  "faces will NOT be double-checked for eyes.")
    return _eye_detector


def _count_eyes(eq, rect):
    det = _get_eye_detector()
    if det is None:
        return -1
    x, y, w, h = rect
    top = eq[y:y + int(h * 0.6), x:x + w]
    size = max(10, w // 10)
    return len(det.detectMultiScale(top, scaleFactor=1.1, minNeighbors=3,
                                    minSize=(size, size)))


def detect_faces(detector, gray, relaxed=False):
    global last_candidates
    neighbors = 3 if relaxed else 6
    min_conf = RELAXED_CONF if relaxed else MIN_CONF
    eq = cv2.equalizeHist(gray)
    try:
        rects, _, weights = detector.detectMultiScale3(
            eq, scaleFactor=1.1, minNeighbors=neighbors, minSize=(100, 100),
            outputRejectLevels=True
        )
        weights = [float(w) for w in np.ravel(weights)]
    except (cv2.error, TypeError):
        rects = detector.detectMultiScale(
            eq, scaleFactor=1.1, minNeighbors=neighbors, minSize=(100, 100)
        )
        weights = [min_conf] * len(rects)

    candidates, accepted = [], []
    for rect, conf in zip(rects, weights):
        x, y, w, h = (int(v) for v in rect)
        contrast = float(cv2.meanStdDev(gray[y:y + h, x:x + w])[1][0][0])
        ok = conf >= min_conf and contrast >= MIN_STD
        eyes = -1
        if ok and not relaxed:
            eyes = _count_eyes(eq, (x, y, w, h))
            ok = eyes != 0
        candidates.append(((x, y, w, h), conf, contrast, eyes, ok))
        if ok:
            accepted.append((x, y, w, h))
    last_candidates = candidates
    return _clean(accepted)


def _clean(faces):
    if len(faces) == 0:
        return []
    faces = sorted((tuple(int(v) for v in f) for f in faces),
                   key=lambda f: f[2] * f[3], reverse=True)
    biggest = faces[0][2] * faces[0][3]
    kept = []
    for (x, y, w, h) in faces:
        if w * h < 0.4 * biggest:
            continue
        cx, cy = x + w // 2, y + h // 2
        if any(kx <= cx <= kx + kw and ky <= cy <= ky + kh
               for kx, ky, kw, kh in kept):
            continue
        kept.append((x, y, w, h))
    return kept


def preprocess_face(gray, rect):
    x, y, w, h = rect
    dx, dy = int(0.10 * w), int(0.05 * h)
    crop = gray[y + dy:y + h - dy, x + dx:x + w - dx]
    crop = cv2.resize(crop, FACE_SIZE)
    return _clahe.apply(crop)


LBF_MODEL = os.path.join(BASE, "models", "lbfmodel.yaml")


def load_landmarker():
    if not os.path.exists(LBF_MODEL) or not hasattr(cv2, "face"):
        return None
    try:
        facemark = cv2.face.createFacemarkLBF()
        facemark.loadModel(LBF_MODEL)
        return facemark
    except cv2.error:
        return None


def get_landmarks(facemark, gray, rect):
    x, y, w, h = rect
    faces = np.array([[x, y, w, h]], dtype=np.int32)
    ok, landmarks = facemark.fit(gray, faces)
    if not ok or len(landmarks) == 0:
        return None
    return np.asarray(landmarks[0]).reshape(-1, 2)


def _eye_aspect_ratio(p):
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (
        2.0 * np.linalg.norm(p[0] - p[3]))


def face_features(pts, rect=None):
    if pts is None or len(pts) != 68:
        return None
    right_eye = pts[36:42].mean(axis=0)
    left_eye = pts[42:48].mean(axis=0)
    nose_tip = pts[30]
    mouth = pts[48:60].mean(axis=0)
    eye_mid = (right_eye + left_eye) / 2
    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    if eye_dist < 1 or not (eye_mid[1] < nose_tip[1] < mouth[1]):
        return None
    if rect is not None and not (0.25 * rect[2] <= eye_dist <= 0.75 * rect[2]):
        return None
    return {
        "roll": float(np.degrees(np.arctan2(left_eye[1] - right_eye[1],
                                            left_eye[0] - right_eye[0]))),
        "yaw": float((nose_tip[0] - eye_mid[0]) / eye_dist),
        "pitch": float((nose_tip[1] - eye_mid[1]) / (mouth[1] - eye_mid[1])),
        "ear": float(min(_eye_aspect_ratio(pts[36:42]), _eye_aspect_ratio(pts[42:48]))),
        "mouth_open": float((pts[66][1] - pts[62][1]) / eye_dist),
        "mouth_w": float(np.linalg.norm(pts[54] - pts[48]) / eye_dist),
    }


def draw_features(frame, pts):
    p = pts.astype(np.int32)
    groups = [
        (range(36, 42), True, (255, 255, 0)), (range(42, 48), True, (255, 255, 0)),
        (range(27, 31), False, (0, 255, 255)), (range(31, 36), False, (0, 255, 255)),
        (range(48, 60), True, (255, 0, 255)), (range(60, 68), True, (255, 0, 255)),
    ]
    for idx, closed, color in groups:
        cv2.polylines(frame, [p[list(idx)].reshape(-1, 1, 2)], closed, color, 1)