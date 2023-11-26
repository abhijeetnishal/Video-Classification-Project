from django.urls import path
from .views import classify_video

urlpatterns = [
    path('api/v1/video', classify_video, name='classify_video'),
]