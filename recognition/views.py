import json
from django.http import HttpResponse
import cv2
from flask import Response
import numpy as np
from rest_framework.decorators import api_view, authentication_classes, permission_classes

import datetime, time
import os
import numpy as np
from threading import Thread



from channels.generic.websocket import AsyncWebsocketConsumer
import base64
from io import BytesIO
from PIL import Image

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




class ImageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        image_data = data['data']
        image = await self.decode_image(image_data)

        # Process the image data using your chosen image processing or face recognition library
        # ...

        # Send a response back to the Vue app with recognized faces or other data
        response = {
            'type': 'faces',
            """ 'data': faces, """
            'timestamp': data['timestamp'],
        }
        await self.send(json.dumps(response))

    async def decode_image(self, image_data):
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        return image
    


def loading_model():
    m = get_model()
    print("model loaded")
    return m


model = loading_model()


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

    # Detect faces in the image 
    captured_image = crop_face(captured_image)
    
    """  and captured_image.shape[0] > 0 """

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
                            """ print(current_date_time)   """                         
                            attendance = Attendance.objects.create(
                            roll = roll[0],
                            date = current_date_time.date(),
                            cl = Class[0],
                            present_status = "Present"
                            )

                        else:
                            print("no")
                    except ValueError as e:
                        result = False 
                        print(result)                                             
        os.remove(p)
    
        # Save the image to the server or process it in some other way
        return HttpResponse(status=200)
    else:
        # Image was not captured, handle the error
        return HttpResponse(status=400)


    
  



""" global thisframe

@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def stream_view(request):
    global thisframe
    dataURL = request.body
    # Convert the data URL to a OpenCV image
    img = cv2.imdecode(np.fromstring(dataURL, np.uint8), cv2.IMREAD_COLOR)
    # Do something with the image, such as process it or save it to disk
    # ...
    thisframe = img
    
    return HttpResponse(status=200)


@api_view(['POST'])
def tasks(request):
    global switch,camera
    if request.form.get('click') == 'Capture':
            global capture,this_frame,attended,first_Capture,identefier
            identefier = request.form['identefier']
            # this_frame = detect_face(this_frame)
            now = datetime.datetime.now()
            p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
            cv2.imwrite(p, this_frame)
            try:
                attended = take_attendance(identefier, p)
                os.remove(p)
            except ValueError as e:
                attended = False
                os.remove(p)
            # os.remove(p)
            first_Capture = True
                          
    if(first_Capture & attended):
        camera.release()
        cv2.destroyAllWindows()
        return HttpResponse("lecture.html",id = identefier)
    else:
        return HttpResponse('index.html',wrong = "wrong input try again")  """