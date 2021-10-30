from django.db import models

# Create your models here.

class Patient(models.Model):
    name = models.TextField(max_length=255, blank=True, null=True)
    xray = models.ImageField(upload_to='original/', blank=True)
    prediction_img = models.ImageField(upload_to='prediction/', blank=True)
    covid_percentage = models.FloatField(blank=True, null=True)
    pneumonia_percentage = models.FloatField(blank=True, null=True)
    normal_percentage = models.FloatField(blank=True, null=True),
    prediction = models.TextField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"Name: {self.name} | Prediction:{self.prediction}"
