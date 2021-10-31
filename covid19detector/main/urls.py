from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/image', views.api_image, name='api_image'),
    path('api/ios/image', views.ios_api_image, name='ios_api_image'),
]