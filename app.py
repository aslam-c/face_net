import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load FaceNet pre-trained model
facenet_model = load_model('facenet_keras.h5')

# Load image
image = cv2.imread('img.jpg')

# Detect faces using MTCNN
detector = MTCNN()
results = detector.detect_faces(image)

# Extract face regions
faces = []
for result in results:
    x, y, width, height = result['box']
    face = image[y:y+height, x:x+width]
    faces.append(face)

# Preprocess faces
def preprocess(face):
    face = cv2.resize(face, (160, 160))  # Resize to 160x160 for FaceNet
    face = face.astype('float32')
    face = preprocess_input(face)  # Normalize values
    return np.expand_dims(face, axis=0)

preprocessed_faces = [preprocess(face) for face in faces]

# Generate face embeddings (vectors)
def get_embedding(model, face):
    # Get embeddings (vector) for a single face
    return model.predict(face)[0]

face_vectors = [get_embedding(facenet_model, face) for face in preprocessed_faces]

# Print the resulting face vectors (embeddings)
for i, vector in enumerate(face_vectors):
    print(f"Face {i+1} vector: {vector}")
