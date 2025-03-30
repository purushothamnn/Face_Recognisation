import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize MTCNN and InceptionResnetV1
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to detect and encode faces
def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces
    return []

# Function to encode specific images with predefined names
def encode_known_faces(known_faces):
    known_face_encodings = []
    known_face_names = []

    for name, image_path in known_faces.items():
        known_image = cv2.imread(image_path)
        if known_image is not None:
            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(known_image_rgb)
            if encodings:
                known_face_encodings.append(encodings[0])  # Assuming one face per image
                known_face_names.append(name)

    return known_face_encodings, known_face_names

# Define known faces with explicit names
known_faces = {
    "elonmusk" : "Images/Elon_Musk_Royal_Society.webp"
}

# Encode known faces
known_face_encodings, known_face_names = encode_known_faces(known_faces)

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        if distances[min_distance_idx] < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

# Start video capture
cap = cv2.VideoCapture(0)
threshold = 0.6

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test_face_encodings = detect_and_encode(frame_rgb)

    if test_face_encodings and known_face_encodings:
        names = recognize_faces(np.array(known_face_encodings), known_face_names, test_face_encodings, threshold)
        for name, box in zip(names, mtcnn.detect(frame_rgb)[0]):
            if box is not None:
                (x1, y1, x2, y2) = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
