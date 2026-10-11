import cv2, os, json
import numpy as np

if not os.path.isdir("faces"):
    print("No faces/ folder found. Run register_face.py first.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
images, labels, names = [], [], {}

# Each person's folder in faces/ becomes one label
for label, person in enumerate(sorted(os.listdir("faces"))):
    folder = os.path.join("faces", person)
    if not os.path.isdir(folder):
        continue
    names[label] = person
    for f in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            labels.append(label)

if not images:
    print("No training images found.")
    exit()

recognizer.train(images, np.array(labels))
recognizer.write("model.yml")

with open("names.json", "w") as fp:
    json.dump(names, fp)

print(f"Trained on {len(images)} images of {len(names)} people: {list(names.values())}")