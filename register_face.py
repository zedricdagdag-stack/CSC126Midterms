import cv2, os, time
import numpy as np
from facelib import (load_detector, detect_faces, preprocess_face, load_landmarker,
                     get_landmarks, face_features, draw_features, FACE_SIZE, BASE)

# Guided capture: (instruction, number of images, pose kind). Varied poses make the
# recognizer far more reliable than near-identical frontal shots.
STAGES = [
    ("Look straight at the camera", 15, "straight"),
    ("Turn your head a little (about 15 deg) toward the LEFT of the screen", 8, "screen_left"),
    ("Turn your head a little (about 15 deg) toward the RIGHT of the screen", 8, "screen_right"),
    ("Tilt your head UP just a little", 6, "up"),
    ("Tilt your head DOWN just a little (keep eyes on the camera)", 6, "down"),
    ("Smile big or open your mouth a little", 8, "expression"),
]

# ---------------- Quality settings (tune these) ----------------
MIN_GAP = 0.25            # seconds between saved images (avoids duplicates)
MIN_FACE_WIDTH = 120      # face must be at least this wide (px): move closer if not
MIN_SHARPNESS = 40        # Laplacian variance; below this the image is too blurry
BRIGHTNESS_RANGE = (45, 215)   # mean gray level of the face region

# Feature checks (only used when models/lbfmodel.yaml is present)
ROLL_MAX = 20             # degrees of sideways head lean allowed
MIN_EAR = 0.17            # eye openness; below this = blinking / eyes closed
YAW_STRAIGHT = 0.12       # max nose offset for the "straight" step
YAW_TURN = 0.10           # how far (vs. your straight pose) to turn left/right
PITCH_TURN = 0.05         # how far (vs. your straight pose) to tilt up/down
MOUTH_OPEN_MIN = 0.08     # mouth opening for the expression step
SMILE_WIDEN = 1.12        # or mouth this much wider than your neutral mouth
# ---------------------------------------------------------------


def pose_check(kind, f, base):
    """Returns (ok, message). `base` holds this person's own straight-on values."""
    if abs(f["roll"]) > ROLL_MAX:
        return False, "Keep your head level"
    if f["ear"] < MIN_EAR:
        return False, "Eyes open please"
    if kind == "straight":
        if abs(f["yaw"]) > YAW_STRAIGHT:
            return False, "Face the camera straight on"
        return True, "Good"
    if base is None:
        return True, "Good"
    dyaw = f["yaw"] - base["yaw"]
    dpitch = f["pitch"] - base["pitch"]
    if kind == "screen_left" and dyaw > -YAW_TURN:
        return False, "Turn more toward the LEFT of the screen"
    if kind == "screen_right" and dyaw < YAW_TURN:
        return False, "Turn more toward the RIGHT of the screen"
    if kind == "up" and dpitch > -PITCH_TURN:
        return False, "Tilt your head up a bit more"
    if kind == "down" and dpitch < PITCH_TURN:
        return False, "Tilt your head down a bit more"
    if kind == "expression" and not (
            f["mouth_open"] >= MOUTH_OPEN_MIN or f["mouth_w"] >= base["mouth_w"] * SMILE_WIDEN):
        return False, "Smile bigger or open your mouth slightly"
    return True, "Good"


name = input("Enter person's name: ").strip()
if not name:
    raise SystemExit("A name is required.")

folder = os.path.join(BASE, "faces", name)
os.makedirs(folder, exist_ok=True)

existing = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
if existing:
    choice = input(f"{len(existing)} images already exist for '{name}'. "
                   "[a]ppend or [r]eplace? ").strip().lower()
    if choice == "r":
        for f in existing:
            os.remove(os.path.join(folder, f))
        existing = []

next_idx = max((int(os.path.splitext(f)[0]) for f in existing
                if os.path.splitext(f)[0].isdigit()), default=-1) + 1

detector = load_detector()
landmarker = load_landmarker()
if landmarker is None:
    print("Landmark model not found (models/lbfmodel.yaml): capturing WITHOUT "
          "eye/nose/mouth checks.")
else:
    print("Landmark model loaded: eyes, nose and mouth will be checked.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam.")

saved = 0
total = sum(n for _, n, _ in STAGES)
quit_all = False
base = None            # this person's own straight-on measurements
base_samples = []

for stage_no, (instruction, target, kind) in enumerate(STAGES, 1):
    captured = 0
    capturing = False      # waits for SPACE so you can get into position
    last_save = 0

    while captured < target and not quit_all:
        ret, frame = cap.read()
        if not ret:
            quit_all = True
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(detector, gray, relaxed=True)

        face_ok = False
        feats = None
        msg, color = "", (0, 0, 255)
        sharp = bright = 0

        if len(faces) == 0:
            msg = "No face found" if kind == "straight" else "No face found - turn back a little"
        elif len(faces) > 1:
            msg = "Only one person in frame, please"
        else:
            x, y, w, h = faces[0]
            region = gray[y:y + h, x:x + w]
            sharp = cv2.Laplacian(cv2.resize(region, FACE_SIZE), cv2.CV_64F).var()
            bright = region.mean()

            if w < MIN_FACE_WIDTH:
                msg = "Move closer"
            elif bright < BRIGHTNESS_RANGE[0]:
                msg = "Too dark - add light"
            elif bright > BRIGHTNESS_RANGE[1]:
                msg = "Too bright / glare"
            elif sharp < MIN_SHARPNESS:
                msg = "Hold still (blurry)"
            elif landmarker is None:
                face_ok = True
                msg, color = "Good", (0, 255, 0)
            else:
                pts = get_landmarks(landmarker, gray, faces[0])
                feats = face_features(pts, faces[0])
                if feats is None:
                    msg, color = "Can't see eyes/nose/mouth clearly", (0, 165, 255)
                else:
                    draw_features(frame, pts)
                    face_ok, msg = pose_check(kind, feats, base)
                    color = (0, 255, 0) if face_ok else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        now = time.time()
        if capturing and face_ok and now - last_save >= MIN_GAP:
            face_img = preprocess_face(gray, faces[0])
            cv2.imwrite(os.path.join(folder, f"{next_idx}.jpg"), face_img)
            next_idx += 1
            captured += 1
            saved += 1
            last_save = now
            if kind == "straight" and feats is not None:
                base_samples.append(feats)

        # --- On-screen guidance ---
        cv2.putText(frame, f"Step {stage_no}/{len(STAGES)}: {instruction}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if capturing:
            status = f"Capturing {captured}/{target}  (total {saved}/{total})"
        else:
            status = "Get into position, then press SPACE"
        cv2.putText(frame, status, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, msg, (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if feats is not None:
            info = (f"yaw {feats['yaw']:+.2f}  pitch {feats['pitch']:.2f}  "
                    f"eyes {feats['ear']:.2f}  mouth {feats['mouth_open']:.2f}")
        else:
            info = f"sharp {sharp:.0f}  bright {bright:.0f}"
        cv2.putText(frame, info, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Register face (SPACE start step, q quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            quit_all = True
        elif key == ord(' '):
            capturing = True

    if quit_all:
        break

    # After the straight-on step, remember this person's own neutral pose so the
    # turn / tilt / smile steps are judged relative to it.
    if kind == "straight" and base_samples:
        base = {k: float(np.median([s[k] for s in base_samples]))
                for k in ("yaw", "pitch", "mouth_w")}

cap.release()
cv2.destroyAllWindows()
print(f"Saved {saved} new images for '{name}' in {folder}")
print("Next: run train.py to rebuild the model.")