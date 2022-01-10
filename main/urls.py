from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict', views.index_predict, name='index_predict'),
    path('api/image', views.api_image, name='api_image'),
    path('api/ios/image', views.ios_api_image, name='ios_api_image'),
    path('api/docs', views.api_docs, name='api_docs'),
    path('warmup', views.warmup, name='warmup'),
    path('register', views.register_fn, name='register'),
    path('login', views.login_fn, name='login'),
    path("logout", views.logout_fn, name="logout"),
    path('scans', views.scans, name='scans'),
    path('scans/<int:scan_id>/delete', views.delete_scan, name='delete_scan'),
]