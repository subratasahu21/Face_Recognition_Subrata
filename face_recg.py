import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

# Initialize MTCNN and InceptionResnetV1 with better parameters
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,
    post_process=False,
    min_face_size=60,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to detect and encode faces with improved processing
def detect_and_encode(image):
    try:
        # Convert to PIL Image for MTCNN
        image_pil = Image.fromarray(image)
        
        # Detect faces
        with torch.no_grad():
            boxes, probs, landmarks = mtcnn.detect(image_pil, landmarks=True)
            
        faces = []
        if boxes is not None:
            for box in boxes:
                # Extract face
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                    
                # Preprocess face
                face = cv2.resize(face, (160, 160))
                face = face.astype(np.float32) / 255.0
                face = (face - 0.5) / 0.5  # Normalize for facenet
                face = np.transpose(face, (2, 0, 1))
                face_tensor = torch.tensor(face).unsqueeze(0).to(device)
                
                # Get encoding
                encoding = resnet(face_tensor).cpu().detach().numpy().flatten()
                faces.append(encoding)
                
        return faces
    except Exception as e:
        print(f"Error in face detection: {e}")
        return []

# Function to encode known faces with error handling
def encode_known_faces(known_faces):
    known_face_encodings = []
    known_face_names = []

    for name, image_path in known_faces.items():
        try:
            # Load image
            known_image = cv2.imread(image_path)
            if known_image is None:
                print(f"Warning: Could not load image at {image_path}")
                continue
                
            # Convert to RGB
            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            
            # Detect and encode
            encodings = detect_and_encode(known_image_rgb)
            if encodings:
                for encoding in encodings:
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
            else:
                print(f"Warning: No faces detected in {image_path}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return known_face_encodings, known_face_names

# Expanded known faces dataset with more images
known_faces = {
    "Subrata Kumar Sahu": "images/Subrata.jpg",
    "Subrata Kumar Sahu": "images/Subrata_2.jpg",
    "Subrata Kumar Sahu": "images/Subrata4.jpg",
    "Subrata Kumar Sahu": "images/Subrata5.jpg",
    "Subrata Kumar Sahu": "images/2.jpg",
    "Subrata Kumar Sahu": "images/1.jpg",
    "Subrata Kumar Sahu": "images/3.jpg",
    "Subrata Kumar Sahu": "images/4.jpg",
    "Subrata Kumar Sahu": "images/5.jpg",
    "Subrata Kumar Sahu": "images/6.jpg",
    "Subrata Kumar Sahu": "images/7.jpg",
    "Subrata Kumar Sahu": "images/8.jpg",
    "Subrata Kumar Sahu": "images/9.jpg",
    "Subrata Kumar Sahu": "images/10.jpg",
    "Subrata Kumar Sahu": "images/11.jpg",
    "Subrata Kumar Sahu": "images/12.jpg",
    "Subrata Kumar Sahu": "images/13.jpg",
    "Subrata Kumar Sahu": "images/14.jpg"
}

# Encode known faces
print("Encoding known faces...")
known_face_encodings, known_face_names = encode_known_faces(known_faces)

if not known_face_encodings:
    print("Error: No face encodings were created. Check your images.")
    exit()

known_face_encodings = np.array(known_face_encodings)
print(f"Encoded {len(known_face_encodings)} faces")

# Improved face recognition function
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    for test_encoding in test_encodings:
        try:
            # Calculate distances using cosine similarity (more robust than L2 norm)
            distances = np.dot(known_encodings, test_encoding) / (
                np.linalg.norm(known_encodings, axis=1) * np.linalg.norm(test_encoding)
            )
            best_match_idx = np.argmax(distances)
            
            # Convert cosine similarity to distance-like metric
            similarity = distances[best_match_idx]
            if similarity > threshold:
                recognized_names.append(known_names[best_match_idx])
            else:
                recognized_names.append("Unknown")
        except Exception as e:
            print(f"Error in recognition: {e}")
            recognized_names.append("Error")
            
    return recognized_names

# Improved video capture loop
def run_face_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    threshold = 0.6  # Adjust this based on your needs
    frame_skip = 2  # Process every nth frame for better performance
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        test_face_encodings = detect_and_encode(frame_rgb)

        # Draw results
        boxes, _ = mtcnn.detect(frame_rgb)
        if boxes is not None and test_face_encodings:
            names = recognize_faces(
                known_face_encodings,
                known_face_names,
                test_face_encodings,
                threshold
            )
            
            for name, box in zip(names, boxes):
                (x1, y1, x2, y2) = map(int, box)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )

        # Display FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recognition()