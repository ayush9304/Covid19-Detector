from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict', views.index_predict, name='index_predict'),
    path('api/image', views.api_image, name='api_image'),
    path('api/ios/image', views.ios_api_image, name='ios_api_image'),
    path('api/docs', views.api_docs, name='api_docs'),
    path('warmup', views.warmup, name='warmup'),
]