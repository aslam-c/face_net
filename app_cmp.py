import numpy as np
import cv2
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FaceNet for face embeddings
embedder = FaceNet()

def detect_faces_and_get_embeddings(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Detect faces and get embeddings using FaceNet
    detections = embedder.extract(image, threshold=0.95)
    
    face_vectors = []
    for detection in detections:
        face_vector = detection['embedding']
        face_vectors.append(face_vector)
    
    return face_vectors

def compare_face_embeddings(embeddings1, embedding2, threshold=0.7):
    """
    Compare face embeddings using cosine similarity. A similarity score of closer to 1
    means the faces are similar, while scores closer to 0 mean they are different.
    """
    for idx, embedding in enumerate(embeddings1):
        similarity = cosine_similarity([embedding], [embedding2])
        print(f"Face {idx + 1} similarity score: {similarity[0][0]}")
        if similarity[0][0] > threshold:
            return True, idx + 1
    return False, -1

# Step 1: Get all face vectors from image1
image1_path = './img.jpg'  # image1 with multiple faces
image2_path = './obama.jpeg'  # image2 with a single face

# Detect faces and get embeddings for image1
face_embeddings_image1 = detect_faces_and_get_embeddings(image1_path)

# Step 2: Get the face vector from image2
face_embedding_image2 = detect_faces_and_get_embeddings(image2_path)

# Step 3: Compare the face vector of image2 with each face in image1
is_face_present, face_idx = compare_face_embeddings(face_embeddings_image1, face_embedding_image2[0])

if is_face_present:
    print(f"Face from image2 is found in image1, matching face index: {face_idx}")
else:
    print("Face from image2 is not found in image1")
