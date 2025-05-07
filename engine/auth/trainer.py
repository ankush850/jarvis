import cv2
import numpy as np
from PIL import Image
import os

# Path to training images
path = 'engine\\auth\\samples'

# Create the LBPH face recognizer
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    raise Exception("cv2.face module not found. Make sure you installed 'opencv-contrib-python'.")

# Load the Haar Cascade face detector
detector = cv2.CascadeClassifier("engine\\auth\\haarcascade_frontalface_default.xml")

def Images_And_Labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        gray_img = Image.open(imagePath).convert('L')  # convert to grayscale
        img_arr = np.array(gray_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])  # Extract ID from filename

        faces = detector.detectMultiScale(img_arr)
        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y:y+h, x:x+w])
            ids.append(id)
    
    return faceSamples, ids

print("Training faces. It will take a few seconds. Wait ...")

faces, ids = Images_And_Labels(path)
recognizer.train(faces, np.array(ids))

# Ensure the output directory exists
os.makedirs('engine\\auth\\trainer', exist_ok=True)
recognizer.write('engine\\auth\\trainer\\trainer.yml')

print("Model trained, Now we can recognize your face.")
