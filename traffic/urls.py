from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('', views.home, name='graph'),
    path('fault_view/', views.fault_view, name='graph'),
    path('home/',views.file,name='home')
]+ static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
