import cv2, json, os, time, threading
from collections import deque
from datetime import datetime

import numpy as np

import facelib
from facelib import detect_faces, preprocess_face

try:
    import winsound          # built into Windows, no install needed
except ImportError:
    winsound = None

# ---------------- Settings (tuned for improved stability) ----------------
SOURCE = 0               # 0 = webcam; or a video path like "samples/test.mp4"
THRESHOLD = 50           # LBPH distance: lower = better match
MIN_MOTION_AREA = 2500   # increased to ignore small noise blobs
ACTIVE_HOLD = 5          # seconds to keep scanning faces after motion stops
INTRUDER_FRAMES = 5      # consecutive frames a stranger must be seen before alarm
ALARM_COOLDOWN = 10      # seconds between alarms
BANNER_TIME = 3          # seconds the red banner stays up

# Tracking settings
MAX_MISSED = 25          # increased frames a track survives without a matching blob
MAX_MATCH_DIST = 120     # max pixels a blob can move between frames and keep its ID
TRAIL_LEN = 40           # number of past positions kept for the motion trail

# Face gating: a face only counts if it sits on something that is moving
FACE_MOTION_MARGIN = 0.25  # grow each motion box by this fraction when checking
CONFIRM_HOLD = 4           # seconds a validated face keeps counting after it stops moving
CONFIRM_DIST = 80          # px: how close a face must stay to its earlier spot
REQUIRE_MOTION_FOR_FACE = True   # Enabled: only accept faces on moving objects
# -------------------------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(BASE, "snapshots")
LOG_FILE = os.path.join(BASE, "events.log")
os.makedirs(SNAP_DIR, exist_ok=True)

# Load face detector
cascade_path = os.path.join(BASE, "models", "haarcascade_frontalface_default.xml")
detector = cv2.CascadeClassifier(cascade_path)
if detector.empty():
    print("Cascade failed to load:", cascade_path)
    exit()

# Load trained recognizer
model_path = os.path.join(BASE, "model.yml")
names_path = os.path.join(BASE, "names.json")
if not (os.path.exists(model_path) and os.path.exists(names_path)):
    print("model.yml / names.json not found. Run register_face.py then train.py first.")
    exit()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
with open(names_path) as fp:
    names = json.load(fp)


def log_event(message):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as fp:
        fp.write(f"[{stamp}] {message}\n")
    print(f"[{stamp}] {message}")


def play_alarm():
    """Beep in a separate thread so the video doesn't freeze."""
    def _beep():
        if winsound:
            for _ in range(3):
                winsound.Beep(1000, 250)
                winsound.Beep(1500, 250)
        else:
            print("\a")
    threading.Thread(target=_beep, daemon=True).start()


class CentroidTracker:
    """Gives each moving blob a persistent ID by matching centroids frame to frame."""

    def __init__(self, max_missed, max_dist, trail_len):
        self.max_missed = max_missed
        self.max_dist = max_dist
        self.trail_len = trail_len
        self.next_id = 0
        self.tracks = {}   # id -> dict(centroid, rect, missed, trail, status)

    def _register(self, rect, centroid):
        self.tracks[self.next_id] = {
            "centroid": centroid,
            "rect": rect,
            "missed": 0,
            "trail": deque([centroid], maxlen=self.trail_len),
            "status": None,    # None = unknown, "intruder", or an authorized name
        }
        log_event(f"TRACK: object #{self.next_id} entered the scene")
        self.next_id += 1

    def update(self, rects):
        if len(rects) == 0:
            for tid in list(self.tracks):
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    log_event(f"TRACK: object #{tid} left the scene")
                    del self.tracks[tid]
            return

        centroids = np.array([(x + w // 2, y + h // 2) for x, y, w, h in rects])

        if not self.tracks:
            for r, c in zip(rects, centroids):
                self._register(r, (int(c[0]), int(c[1])))
            return

        ids = list(self.tracks)
        old = np.array([self.tracks[i]["centroid"] for i in ids])
        D = np.linalg.norm(old[:, None, :] - centroids[None, :, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols or D[r, c] > self.max_dist:
                continue
            t = self.tracks[ids[r]]
            t["centroid"] = (int(centroids[c][0]), int(centroids[c][1]))
            t["rect"] = rects[c]
            t["missed"] = 0
            t["trail"].append(t["centroid"])
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(ids))) - used_rows:
            tid = ids[r]
            self.tracks[tid]["missed"] += 1
            if self.tracks[tid]["missed"] > self.max_missed:
                log_event(f"TRACK: object #{tid} left the scene")
                del self.tracks[tid]

        for c in set(range(len(rects))) - used_cols:
            self._register(rects[c], (int(centroids[c][0]), int(centroids[c][1])))

    def assign_face(self, face_center, status):
        fx, fy = face_center
        for t in self.tracks.values():
            if t["missed"] > 0:
                continue
            x, y, w, h = t["rect"]
            if x <= fx <= x + w and y <= fy <= y + h:
                t["status"] = status


def draw_tracks(frame, tracker, show_trails):
    for tid, t in tracker.tracks.items():
        if t["missed"] > 0:
            continue
        status = t["status"]
        if status is None:
            color, label = (0, 255, 255), f"ID {tid}"
        elif status == "intruder":
            color, label = (0, 0, 255), f"ID {tid} INTRUDER"
        else:
            color, label = (0, 255, 0), f"ID {tid} {status}"

        x, y, w, h = t["rect"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, t["centroid"], 4, color, -1)

        if show_trails:
            pts = list(t["trail"])
            for i in range(1, len(pts)):
                thickness = max(1, int(3 * i / len(pts)))
                cv2.line(frame, pts[i - 1], pts[i], color, thickness)


def face_is_valid(rect, now):
    x, y, w, h = rect
    cx, cy = x + w // 2, y + h // 2
    for t in tracker.tracks.values():
        if t["missed"] > 0:
            continue
        tx, ty, tw, th = t["rect"]
        mx, my = int(tw * FACE_MOTION_MARGIN), int(th * FACE_MOTION_MARGIN)
        if tx - mx <= cx <= tx + tw + mx and ty - my <= cy <= ty + th + my:
            return True
    return any(now - seen < CONFIRM_HOLD and abs(cx - px) < CONFIRM_DIST
               and abs(cy - py) < CONFIRM_DIST for px, py, seen in confirmed_faces)


def update_confirmed(valid_faces, now):
    new = [(x + w // 2, y + h // 2, now) for (x, y, w, h) in valid_faces]
    keep = [f for f in confirmed_faces if now - f[2] < CONFIRM_HOLD
            and not any(abs(f[0] - n[0]) < CONFIRM_DIST and abs(f[1] - n[1]) < CONFIRM_DIST
                        for n in new)]
    confirmed_faces[:] = new + keep


cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
tracker = CentroidTracker(MAX_MISSED, MAX_MATCH_DIST, TRAIL_LEN)

last_motion = 0
intruder_streak = 0
last_alarm = 0
banner_until = 0
last_logged = {}
show_mask = False
show_trails = True
show_debug = False
confirmed_faces = []

log_event("System started")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    now = time.time()
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    mask = subtractor.apply(blur)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_rects = []
    for c in contours:
        if cv2.contourArea(c) >= MIN_MOTION_AREA:
            last_motion = now
            motion_rects.append(cv2.boundingRect(c))

    tracker.update(motion_rects)
    active = (now - last_motion) < ACTIVE_HOLD

    intruder_now = False
    face_results = []
    if active:
        faces = detect_faces(detector, gray)
        if REQUIRE_MOTION_FOR_FACE:
            faces = [f for f in faces if face_is_valid(f, now)]
        update_confirmed(faces, now)
        for (x, y, w, h) in faces:
            crop = preprocess_face(gray, (x, y, w, h))
            label, dist = recognizer.predict(crop)
            center = (x + w // 2, y + h // 2)

            if dist < THRESHOLD:
                name = names.get(str(label), "Unknown")
                text, color = f"{name} ({dist:.0f})", (0, 255, 0)
                tracker.assign_face(center, name)
                if now - last_logged.get(name, 0) > 30:
                    log_event(f"AUTHORIZED: {name} (distance {dist:.0f})")
                    last_logged[name] = now
            else:
                text, color = f"INTRUDER ({dist:.0f})", (0, 0, 255)
                tracker.assign_face(center, "intruder")
                intruder_now = True

            face_results.append(((x, y, w, h), text, color))

    draw_tracks(frame, tracker, show_trails)
    for (x, y, w, h), text, color in face_results:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if show_debug and active:
        for (x, y, w, h), conf, contrast, eyes, ok in facelib.last_candidates:
            dbg_color = (255, 255, 0) if ok else (160, 160, 160)
            cv2.rectangle(frame, (x, y), (x + w, y + h), dbg_color, 1)
            cv2.putText(frame, f"c{conf:.1f} s{contrast:.0f} e{eyes if eyes >= 0 else '-'}", (x, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, dbg_color, 1)

    intruder_streak = intruder_streak + 1 if intruder_now else 0
    if intruder_streak >= INTRUDER_FRAMES and now - last_alarm > ALARM_COOLDOWN:
        last_alarm = now
        banner_until = now + BANNER_TIME
        play_alarm()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap = os.path.join(SNAP_DIR, f"intruder_{stamp}.jpg")
        cv2.imwrite(snap, frame)
        log_event(f"INTRUDER ALERT - snapshot saved: {os.path.basename(snap)}")

    visible = sum(1 for t in tracker.tracks.values() if t["missed"] == 0)
    status = "SCANNING FACES" if active else "ARMED - monitoring"
    cv2.putText(frame, f"{status} | tracking: {visible}", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if now < banner_until and int(now * 4) % 2 == 0:
        cv2.rectangle(frame, (0, 0), (640, 50), (0, 0, 255), -1)
        cv2.putText(frame, "INTRUDER ALERT!", (150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Home Protection System (q quit, m mask, t trails, d debug)", frame)
    if show_mask:
        cv2.imshow("Motion Mask", mask)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            cv2.destroyWindow("Motion Mask")
    elif key == ord('t'):
        show_trails = not show_trails
    elif key == ord('d'):
        show_debug = not show_debug

cap.release()
cv2.destroyAllWindows()
log_event("System stopped")