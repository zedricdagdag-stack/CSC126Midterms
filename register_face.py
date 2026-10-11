import cv2, os

name = input("Enter person's name: ").strip()
folder = os.path.join("faces", name)
os.makedirs(folder, exist_ok=True)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(0)
count, TARGET = 0, 40

while count < TARGET:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    for (x, y, w, h) in faces[:1]:  # one face at a time
        crop = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
        cv2.imwrite(os.path.join(folder, f"{count}.jpg"), crop)
        count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Captured {count}/{TARGET}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Register", frame)
    if cv2.waitKey(150) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()