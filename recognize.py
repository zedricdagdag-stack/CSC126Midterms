import cv2, json, os
from facelib import load_detector, detect_faces, preprocess_face, BASE

model_path = os.path.join(BASE, "model.yml")
names_path = os.path.join(BASE, "names.json")
if not os.path.exists(model_path):
    raise SystemExit("model.yml not found. Run train.py first.")

detector = load_detector()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
with open(names_path) as fp:
    names = json.load(fp)

THRESHOLD = 70  # lower distance = better match; tune this

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(detector, gray)

    for (x, y, w, h) in faces:
        crop = preprocess_face(gray, (x, y, w, h))
        label, dist = recognizer.predict(crop)

        if dist < THRESHOLD:
            text, color = f"{names[str(label)]} ({dist:.0f})", (0, 255, 0)
        else:
            text, color = f"INTRUDER ({dist:.0f})", (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Home Protection (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()