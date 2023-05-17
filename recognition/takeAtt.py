from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

import tensorflow as tf


def get_encoder(input_shape):
    """ Returns the image encoding model """

    pretrained_model = Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )
    
    for i in range(len(pretrained_model.layers)-27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model



def verify_images(img_path1, img_path2, model, threshold=1.3):
    # Read the images
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    # Preprocess the images
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (128, 128))
    img2 = cv2.resize(img2, (128, 128))
    img1 = preprocess_input(np.array([img1]))
    img2 = preprocess_input(np.array([img2]))
    
    # Get the encodings for the images
    tensor1 = model.predict(img1)
    tensor2 = model.predict(img2)

    # Compute the distance between the encodings
    distance = np.sum(np.square(tensor1-tensor2), axis=-1)
    print(distance)
    # Return whether the images belong to the same person or not
    return distance <= threshold


def get_model():
    encoder = get_encoder((128, 128, 3))
    """ encoder.load_weights("C:\\Users\\LENOVO\\Desktop\\AttendanceSystem\\Gp1\\encoder") """
    """ encoder.load_weights("./recognition/Gp1/encoder") """
    encoder.load_weights("..\\Gp1\\encoder")
    return encoder

def crop_face(image):
    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier('.\\recognition\\face_detection\\haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Crop the first face detected (assuming there's only one)
    (x, y, w, h) = faces[0]
    cropped_image = image[y:y+h, x:x+w]

    # Return the cropped image
    return cropped_image


""" model = get_model()

img_path1 = "C:\\Users\\LENOVO\\Desktop\\AttendanceSystem\\AttendanceSystem_django\\media\\students\\201900529\\ghandy.png"
img_path2 = "C:\\Users\\LENOVO\\Desktop\\AttendanceSystem\\Gp1\\IMG-20230508-WA0023.jpg"

result = verify_images(img_path1, img_path2, model)

if result:
    print("The images belong to the same person.")
else:
    print("The images belong to different persons.")  """