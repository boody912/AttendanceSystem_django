from django.urls import path
from channels.routing import URLRouter ,ProtocolTypeRouter 
from channels.auth import AuthMiddlewareStack

from .views import upload_image , take_multi_att, take_multi_attend

""" urlpatterns = [
    path('video_feed/', views.video_feed),
       path('stream/', views.stream_view),
    path('tasks/', views.tasks),
] """


urlpatterns = [
 
    path('upload_image/', upload_image),
    path('take_multi_att/', take_multi_att),
    path('take_multi_attend/', take_multi_attend),
    
]


 
