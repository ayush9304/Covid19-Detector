from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import Patient


import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def get_covid19_classifier():
	model = load_model('models\\covid19_pneumonia_classifier.h5')
	model.summary()
	return model

def get_xray_validator():
	model = load_model('models\\xray_validator.h5')
	model.summary()
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


def validate(file) -> bool:
    img = Image.open(file.file)
    img = img.convert('RGB')
    img = img.resize((150, 150), Image.NEAREST)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)/255.0
    prediction = validator.predict(img)[0][0]
    return prediction>=0.5

def classify(file):
    img = Image.open(file.file)
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

@csrf_exempt
def api_image(request):
    if request.method != "POST":
        return JsonResponse({
            'success': False,
            'description': "Request method must be POST"
        })
    if request.FILES['image']:
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
            'image_url': "127.0.0.1:8000/media/" + str(data.xray)
        })

