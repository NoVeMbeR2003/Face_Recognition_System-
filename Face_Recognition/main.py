import cv2
import os
import numpy as np

# =========================
# STEP 1: COLLECT DATASET
# =========================

choice = input("Collect new dataset? (y/n): ")

dataset_path = "D:/python_myProjects/Face_Recognition/dataset"

if choice.lower() == 'y':
    name = input("Enter the name: ")
    person_path = os.path.join(dataset_path, name)
    os.makedirs(person_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            file_path = f"{person_path}/img_{count}.jpg"
            cv2.imwrite(file_path, face)

            count += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Collecting Dataset", frame)

        if count >= 50 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Dataset Collected ✅")


# =========================
# STEP 2: TRAIN MODEL
# =========================

faces = []
labels = []
label_map = {}
current_label = 0

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_map[current_label] = person_name

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

labels = np.array(labels)

print("Training on faces:", len(faces))

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

model.save("face_model.yml")
np.save("labels.npy", label_map)

print("Model Training Completed ✅")


# =========================
# STEP 3: REAL-TIME RECOGNITION
# =========================

model.read("face_model.yml")
label_map = np.load("labels.npy", allow_pickle=True).item()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Starting Face Recognition... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = model.predict(face)

        # Confidence check (lower = better)
        if confidence < 70:
            name = label_map[label]
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()