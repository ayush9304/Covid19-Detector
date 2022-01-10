from django.db import models
from django.contrib.auth.models import AbstractUser

from datetime import datetime

# Create your models here.

class User(AbstractUser):
    def __str__(self):
        return f"{self.id}: {self.first_name} {self.last_name} ({self.username})"

class Scan(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE,related_name="scans", blank=True, null=True)
    name = models.TextField(max_length=255, blank=True, null=True)
    xray = models.ImageField(upload_to='original/', blank=True)
    prediction_img = models.ImageField(upload_to='prediction/', blank=True)
    covid_percentage = models.FloatField(blank=True, null=True)
    pneumonia_percentage = models.FloatField(blank=True, null=True)
    normal_percentage = models.FloatField(blank=True, null=True)
    prediction = models.TextField(max_length=255, blank=True, null=True)
    scan_date = models.DateTimeField(default=datetime.now)

    def __str__(self):
        return f"Name: {self.name} | Prediction:{self.prediction}"
