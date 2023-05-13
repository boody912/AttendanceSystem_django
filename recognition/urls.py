from django.urls import path
from channels.routing import URLRouter ,ProtocolTypeRouter 
from channels.auth import AuthMiddlewareStack

from .views import upload_image , loading_model

""" urlpatterns = [
    path('video_feed/', views.video_feed),
       path('stream/', views.stream_view),
    path('tasks/', views.tasks),
] """


urlpatterns = [
 
    path('upload_image/', upload_image),
    
]



