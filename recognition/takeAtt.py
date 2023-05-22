from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

import tensorflow as tf

import cv2
# import matplotlib.pyplot as plt
import os
from deepface import DeepFace
import numpy as np
import cv2


def detect_faces(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Define the path to the Caffe model and prototxt files
    model_path = ".\\recognition\\face_detection\\res10_300x300_ssd_iter_140000_fp16.caffemodel"
    prototxt_path = ".\\recognition\\face_detection\\deploy.prototxt"

    # Load the Caffe model and prototxt files
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Define the desired output size for the face detection
    output_size = (300, 300)

    # Preprocess the image for face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, output_size), 1.0, output_size, (104.0, 177.0, 123.0))

    # Pass the preprocessed image through the network to detect faces
    net.setInput(blob)
    detections = net.forward()

    # Define the amount of padding to include around each face
    padding = 0.5

    # Crop and save each face as a separate image
    cropped_faces = []
    for i in range(detections.shape[2]):
        # Extract the confidence score for theface detection
        confidence = detections[0, 0, i, 2]

        # Only process detections with a high confidence level
        if confidence > 0.2:
            # Extract the bounding box coordinates for the face detection
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Calculate the amount of padding to add
            padding_x = int((endX - startX) * padding)
            padding_y = int((endY - startY) * padding)

            # Apply the padding to the cropping coordinates
            startX -= padding_x
            startY -= padding_y
            endX += padding_x
            endY += padding_y

            # Crop the face and append to`cropped_faces` list
            cropped_face = image[startY:endY, startX:endX]
            cropped_faces.append(cropped_face)

    # Return the cropped faces
    return cropped_faces


# def plot_faces(cropped_faces):
#     # Determine the number of rows and columns to use in the plot
#     num_faces = len(cropped_faces)
#     num_rows = int(num_faces**0.5)
#     num_cols = int(num_faces/num_rows) + 1

#     # Create a figure to display the faces
#     fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10,10))

#     # Plot each face in a separate subplot
#     for i, ax in enumerate(axes.flatten()):
#         if i < num_faces:
#             # Convert BGR to RGB color space
#             face_rgb = cv2.cvtColor(cropped_faces[i], cv2.COLOR_BGR2RGB)
#             ax.imshow(face_rgb)
#             ax.axis('off')
#         else:
#             ax.axis('off')

#     # Display the plot
#     # plt.show()





def save_images(images, folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save each image in the folder with the .jpg extension
    for i, image in enumerate(images):
        if image is not None and len(image) > 0:
            file_path = os.path.join(folder_path, f"image_{i}.jpg")
            cv2.imwrite(file_path, image)
        else:
            print(f"Skipping invalid image {i}")

""" def take_multi_att(image_path):
    shots = "shots"
    students = "students"
    attendent_students = []
    cropped_faces = detect_faces(image_path)
    # plot_faces(cropped_faces)
    save_images(cropped_faces, shots)


    for shot in os.listdir(shots):
        shot_path = os.path.join(shots, shot)
        for student in os.listdir(students):
            student_path = os.path.join(students, student)
            try:
                for student_shot in os.listdir(student_path):
                    verify_image = os.path.join(student_path, student_shot)
                    result = DeepFace.verify(img1_path=shot_path, img2_path=verify_image)
                if(result['distance'] <= 0.25):
                    attendent_students.append(student)
                    continue
            except:
                continue
        os.remove(shot_path)

    return attendent_students """


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