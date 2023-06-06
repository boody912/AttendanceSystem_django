import json
from django.http import HttpResponse, JsonResponse
import cv2
from flask import Response
import numpy as np
from rest_framework.decorators import api_view, authentication_classes, permission_classes

import datetime, time
import os
import numpy as np
from threading import Thread



""" from channels.generic.websocket import AsyncWebsocketConsumer
import base64
from io import BytesIO
from PIL import Image """

import cv2
import base64
import numpy as np
import cv2
import datetime
import os
from django.http import HttpResponse
from rest_framework.decorators import api_view
from deepface import DeepFace
from .takeAtt import *
from attendance.models import Attendance


def loading_model():
    m = get_model()
    print("model loaded")
    return m


model = loading_model()

@api_view(['POST'])
def take_multi_attend(request):   
    # Get the uploaded file from the request
    image_data = request.POST.get('image')
    Class = request.data.get('class'),

    # preprocess the shot
    captured_image_bytes = base64.b64decode(image_data.split(',')[1])
    captured_image_np = np.frombuffer(captured_image_bytes, dtype=np.uint8)
    captured_image = cv2.imdecode(captured_image_np, cv2.IMREAD_COLOR)


    # Save the image using OpenCV
    cv2.imwrite(".\\media\\temp\\image.jpg", captured_image)
   
    shots = ".\\media\\shots"
    temp = ".\\media\\temp\\image.jpg"
    students = ".\\media\\students"
    attendent_students = []
    cropped_faces = detect_faces(temp) 
    save_images(cropped_faces, shots)


    for shot in os.listdir(shots):
        shot_path = os.path.join(shots, shot)
        for student in os.listdir(students):
            student_path = os.path.join(students, student)
            try:
                for student_shot in os.listdir(student_path):
                    verify_image = os.path.join(student_path, student_shot)
                    result = DeepFace.verify(img1_path=shot_path, img2_path=verify_image)
                if(result['distance'] <= 0.225):
                    attendent_students.append(student)
                    current_date_time = datetime.datetime.now()
                    attendance = Attendance.objects.create(
                            roll = student,
                            date = current_date_time.date(),
                            cl = Class[0],
                            present_status = "Present"
                            )
                    break
            except:
                continue
        os.remove(shot_path) 

    os.remove(temp) 
    print(attendent_students)
    return JsonResponse({
                            'success': True,
                            'data': attendent_students
                        })    
    

@api_view(['POST'])
def take_multi_att(request):   
    # Get the uploaded file from the request
    image_data = request.FILES.get('imagee')
    Class = request.data.get('class'),

    # Read the image using OpenCV
    image_array = np.frombuffer(image_data.read(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Save the image using OpenCV
    cv2.imwrite(".\\media\\temp\\image.jpg", image)
   
    shots = ".\\media\\shots"
    temp = ".\\media\\temp\\image.jpg"
    students = ".\\media\\students"
    attendent_students = []
    cropped_faces = detect_faces(temp) 
    save_images(cropped_faces, shots)


    for shot in os.listdir(shots):
        shot_path = os.path.join(shots, shot)
        for student in os.listdir(students):
            student_path = os.path.join(students, student)
            try:
                for student_shot in os.listdir(student_path):
                    verify_image = os.path.join(student_path, student_shot)
                    result = DeepFace.verify(img1_path=shot_path, img2_path=verify_image)                       
                if(result['distance'] <= 0.2):
                    attendent_students.append(student)
                    current_date_time = datetime.datetime.now()
                    attendance = Attendance.objects.create(
                            roll = student,
                            date = current_date_time.date(),
                            cl = Class[0],
                            present_status = "Present"
                            )
                    break
            except:
                continue
        os.remove(shot_path) 

    os.remove(temp) 
    print(attendent_students)
    return JsonResponse({
                            'success': True,
                            'data': attendent_students
                        })    
    


@api_view(['POST'])
def upload_image(request):
  
    image_data = request.POST.get('image') 
    roll = request.data.get('roll'),
    Class = request.data.get('class'),
    print(roll[0])
    print(type(image_data))

    # preprocess the shot
    captured_image_bytes = base64.b64decode(image_data.split(',')[1])
    captured_image_np = np.frombuffer(captured_image_bytes, dtype=np.uint8)
    captured_image = cv2.imdecode(captured_image_np, cv2.IMREAD_COLOR)

    # # Detect faces in the image 
    # captured_image = crop_face(captured_image)
    
    if captured_image is not None :
        # Image was captured successfully, process it
        now = datetime.datetime.now()
        p = os.path.sep.join(['.\\media\\shots', "shot_{}.png".format(str(now).replace(":",''))])
        cv2.imwrite(p, captured_image)

        """ test """ 

        rootdir = '.\\media\\students'
        for dir in os.listdir(rootdir):
            if(dir == roll[0]):
                d = os.path.join(rootdir, dir)
                for image in os.listdir(d):
                    anchor = os.path.join(d, image)
                    # anchor = preprocess(anchor)
                    # result = siamese_model.predict(list(np.expand_dims([img, anchor], axis=1)))
                    try:
                        result =DeepFace.verify(img1_path = p, img2_path = anchor)                         
                        """ result = verify_images(p, anchor, model) """
                        print(result)
                        if result['verified'] == True:
                            print("yes")
                            current_date_time = datetime.datetime.now()                         
                            attendance = Attendance.objects.create(
                            roll = roll[0],
                            date = current_date_time.date(),
                            cl = Class[0],
                            present_status = "Present"
                            )
                            os.remove(p)
                            return JsonResponse({
                                'success': True,
                                'recognized': True
                            })
                            
                        else:
                            print("not recognized")
                            os.remove(p)
                            return JsonResponse({
                                'success': True,
                                'recognized': False
                            })
                    except ValueError as e:
                        print('Error: ', e)
                        result = "no face detected" 
                        print(result) 
                        os.remove(p) 
                        return JsonResponse({
                                'success': False,
                                'recognized': False
                            })                                           
        os.remove(p)    
        # Save the image to the server or process it in some other way
        return HttpResponse(status=200)
    else:
        # Image was not captured, handle the error
        return HttpResponse(status=400)


    
  
