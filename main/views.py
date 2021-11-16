from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import Patient
from django.conf import settings


import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.core.files.base import ContentFile
import random
import string
import base64
import os


def get_covid19_classifier():
	# model = load_model('models\\covid19_pneumonia_classifier.h5')
	model = load_model(os.path.join(settings.BASE_DIR, 'models/_covid19_pneumonia_classifier.h5'))
	# model.summary()
	return model

def get_xray_validator():
	model = load_model(os.path.join(settings.BASE_DIR, 'models/_xray_validator.h5'))
	# model.summary()
	return model

classifier = get_covid19_classifier()
validator = get_xray_validator()

predictions_class = {
    0: "COVID",
    1: "Normal",
    2: "Viral Pneumonia"
}

validation_class = {
    0: "not xray",
    1: "xray"
}


def validate(file, isurl=False) -> bool:
    if not isurl:
        img = Image.open(file.file)
    else:
        img = Image.open(os.path.join(settings.BASE_DIR, 'media/'+file))
    img = img.convert('RGB')
    img = img.resize((150, 150), Image.NEAREST)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)/255.0
    prediction = validator.predict(img)[0][0]
    return prediction>=0.5

def classify(file, isurl=False):
    if not isurl:
        img = Image.open(file.file)
    else:
        img = Image.open(os.path.join(settings.BASE_DIR, 'media/'+file))
    img = img.convert('RGB')
    img = img.resize((180, 180), Image.NEAREST)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)/255.0
    prediction = classifier.predict(img)[0]
    covid_p = prediction[0]
    normal_p = prediction[1]
    pneumonia_p = prediction[2]
    return predictions_class[np.argmax(prediction, axis=-1)], covid_p, normal_p, pneumonia_p


# Create your views here.

def index(request):
    return render(request, 'main/index.html')

def index_predict(request):
    if request.method != "POST":
        return render(request, 'main/index.html', {
            'result': [
                "success: False",
                "description: Request method must be POST"
            ]
        })
    if request.FILES.get('image'):
        name = request.POST.get('name')
        image = request.FILES['image']

        if not validate(image):
            return render(request, 'main/index.html', {
                'result': [
                    "success: False",
                    "description: Image is not a X-ray"
                ]
            })

        data = Patient.objects.create(name=name, xray=image)

        prediction, covid_percentage, normal_percentage, pneumonia_percentage = classify(image)
        data.prediction = prediction
        data.covid_percentage = covid_percentage
        data.normal_percentage = normal_percentage
        data.pneumonia_percentage = pneumonia_percentage
        data.save()

        return render(request, 'main/index.html', {
            "result": [
                f"success: True",
                f"method: {request.method}",
                f"name: {data.name}",
                f"description: Successfully uploaded",
                f"covid_percentage: {float(data.covid_percentage)}",
                f"normal_percentage: {float(data.normal_percentage)}",
                f"pneumonia_percentage: {float(data.pneumonia_percentage)}",
                f"prediction: {data.prediction}",
                f"image_url: https://covid19-detection-api.herokuapp.com/media/{str(data.xray)}"
            ]
        })
    else:
        return render(request, 'main/index.html', {
            "result": [
                "success: False",
                "description: No image found"
            ]
        })

@csrf_exempt
def api_image(request):
    if request.method != "POST":
        return JsonResponse({
            'success': False,
            'description': "Request method must be POST"
        })
    if request.FILES.get('image'):
        name = request.POST.get('name')
        image = request.FILES['image']

        if not validate(image):
            return JsonResponse({
                'success': False,
                'description': "Image is not a X-ray"
            })

        data = Patient.objects.create(name=name, xray=image)

        prediction, covid_percentage, normal_percentage, pneumonia_percentage = classify(image)
        data.prediction = prediction
        data.covid_percentage = covid_percentage
        data.normal_percentage = normal_percentage
        data.pneumonia_percentage = pneumonia_percentage
        data.save()

        return JsonResponse({
            'success': True,
            'method': request.method,
            'name': data.name,
            'description': "Successfully uploaded",
            'covid_percentage': float(data.covid_percentage),
            'normal_percentage': float(data.normal_percentage),
            'pneumonia_percentage': float(data.pneumonia_percentage),
            'prediction': data.prediction,
            'image_url': "https://covid19-detection-api.herokuapp.com/media/" + str(data.xray)
        })
    else:
        return JsonResponse({
            'success': False,
            'description': "No image found"
        })

@csrf_exempt
def ios_api_image(request):
    if request.method != "POST":
        return JsonResponse({
            'success': False,
            'description': "Request method must be POST",
            'covid_percentage': 1,
            'normal_percentage': 1,
            'pneumonia_percentage': 1,
            'prediction': "null",
            'image_url': "null"
        })

    if request.POST.get('image'):
        imgstr = request.POST.get('image')
        imgstr = imgstr.replace(" ", "+")

        filename = ''.join(random.choices(string.ascii_letters + string.digits, k=16)) + ".jpg"
        imgdata = ContentFile(base64.b64decode(imgstr))
        
        data = Patient.objects.create()
        data.xray.save(filename, imgdata, save=True)  # xray is Patient model's ImageField

        img_url = str(data.xray)
        
        if validate(img_url, isurl=True):
            prediction, covid_percentage, normal_percentage, pneumonia_percentage = classify(img_url, isurl=True)
            data.prediction = prediction
            data.covid_percentage = covid_percentage
            data.normal_percentage = normal_percentage
            data.pneumonia_percentage = pneumonia_percentage
            data.save()

            return JsonResponse({
                'success': True,
                'method': request.method,
                'name': "null",
                'description': "Successfully uploaded",
                'covid_percentage': data.covid_percentage+1,
                'normal_percentage': data.normal_percentage+1,
                'pneumonia_percentage': data.pneumonia_percentage+1,
                'prediction': data.prediction,
                'image_url': "https://covid19-detection-api.herokuapp.com/media/" + str(data.xray),
            })
        else:
            #data.delete()
            return JsonResponse({
                'success': False,
                'description': "Not an X-Ray image",
                'covid_percentage': 1,
                'normal_percentage': 1,
                'pneumonia_percentage': 1,
                'prediction': "null",
                'image_url': "https://covid19-detection-api.herokuapp.com/media/" + str(data.xray)
            })
    else:
        return JsonResponse({
            'success': False,
            'description': "No image found",
            'covid_percentage': 1,
            'normal_percentage': 1,
            'pneumonia_percentage': 1,
            'prediction': "null",
            'image_url': "null"
        })

def api_docs(request):
    return render(request, 'main/api_docs.html')

def warmup(request):
    return JsonResponse({
        'success': True,
        'description': "Warming up"
    })