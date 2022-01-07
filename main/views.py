from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import Patient
from django.conf import settings

from .utils import *

import tensorflow as tf
import numpy as np
from PIL import Image
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.core.files.base import ContentFile
import random
import string
import base64
import os
# import cv2

# def dice_coef(y_true, y_pred):
#     y_true_f = keras.flatten(y_true)
#     y_pred_f = keras.flatten(y_pred)
#     intersection = keras.sum(y_true_f * y_pred_f)
#     return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)

def get_covid19_classifier():
    # model = load_model(os.path.join(settings.BASE_DIR, 'models/v2/_densenet121.h5'))
    classifier = tf.lite.Interpreter(model_path="models/lite/densenet121.tflite")
    classifier.allocate_tensors()
    classifier_input = classifier.get_input_details()
    classifier_output = classifier.get_output_details()
    return classifier, classifier_input, classifier_output

def get_xray_validator():
	# model = load_model(os.path.join(settings.BASE_DIR, 'models/v2/_xray_validator_v2.h5'))
    validator = tf.lite.Interpreter(model_path="models/lite/xray_validator.tflite")
    validator.allocate_tensors()
    validator_input = validator.get_input_details()
    validator_output = validator.get_output_details()
    return validator, validator_input, validator_output

def get_lungs_segmentor():
    # model = load_model(os.path.join(settings.BASE_DIR, 'models/v2/_lungs_segmentation.h5'), custom_objects={'dice_coef':dice_coef, 'dice_coef_loss':dice_coef_loss})
    segmentor = tf.lite.Interpreter(model_path="models/lite/lungs_segmentation.tflite")
    segmentor.allocate_tensors()
    segmentor_input = segmentor.get_input_details()
    segmentor_output = segmentor.get_output_details()
    return segmentor, segmentor_input, segmentor_output

CLASSIFIER, CLASSIFIER_I, CLASSIFIER_O = get_covid19_classifier()
VALIDATOR, VALIDATOR_I, VALIDATOR_O = get_xray_validator()
SEGMENTOR, SEGMENTOR_I, SEGMENTOR_O = get_lungs_segmentor()

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
    img = img.resize((160, 160), Image.NEAREST)
    img = image.img_to_array(img)
    img = (np.expand_dims(img, axis=0)/255.0).astype(np.float32)
    VALIDATOR.set_tensor(VALIDATOR_I[0]['index'], img)
    VALIDATOR.invoke()
    output = VALIDATOR.get_tensor(VALIDATOR_O[0]['index'])
    # prediction = validator.predict(img)[0][0]
    # return output>=0.74
    return output[0][0] >= 0.74

# def classify(file, isurl=False):
#     if not isurl:
#         img = Image.open(file.file)
#     else:
#         img = Image.open(os.path.join(settings.BASE_DIR, 'media/'+file))
#     img = img.convert('RGB')
#     img = img.resize((180, 180), Image.NEAREST)
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)/255.0

#     ############
#     classifier.summary()
#     print("MODEL INPUT SHAPE:  ---------------- >  ",classifier.input_shape)
#     print("IMG SHAPE:  ---------------- >  ",img.shape)
#     ############
    
#     prediction = classifier.predict(img)[0]
#     covid_p = prediction[0]
#     normal_p = prediction[1]
#     pneumonia_p = prediction[2]
#     return predictions_class[np.argmax(prediction, axis=-1)], covid_p, normal_p, pneumonia_p

def predict(file, isurl=False):
    if not isurl:
        img = Image.open(file.file)
    else:
        img = Image.open(os.path.join(settings.BASE_DIR, 'media/'+file))
    img = img.convert('RGB')
    img = img.resize((512, 512), Image.NEAREST)
    x_g = image.img_to_array(img.convert('L'))
    X = x_g.reshape((1, 512, 512, 1))
    X_norm = ((X-127.0)/127.0).astype(np.float32)
    # segment = segmentor.predict(X_norm)

    SEGMENTOR.set_tensor(SEGMENTOR_I[0]['index'], X_norm)
    SEGMENTOR.invoke()
    output = SEGMENTOR.get_tensor(SEGMENTOR_O[0]['index'])

    segment = np.squeeze(output)
    segment[segment>=0.5] = 255
    segment[segment<0.5] = 0
    segment = segment.astype(np.uint8)
    ekernal = np.ones([19,19])
    dkernal = np.ones([25,25])
    pmask = erode(segment, ekernal)
    fmask = dilate(pmask, dkernal)
    segment = fmask

    e_lung = np.zeros((512,512))
    i=(np.squeeze(X_norm)*127)+127
    e_lung[segment>10] = i[segment>10]
    e_lung = e_lung.astype(np.uint8)
    e_lung = Image.fromarray(e_lung)
    e_lung = e_lung.convert('RGB')
    e_lung = e_lung.resize((224, 224), Image.NEAREST)
    e_lung = image.img_to_array(e_lung)

    X1_norm = e_lung/255.
    X1_norm = np.expand_dims(X1_norm, axis=0)
    # pred = classifier.predict(X1_norm)

    CLASSIFIER.set_tensor(CLASSIFIER_I[0]['index'], X1_norm)
    CLASSIFIER.invoke()
    output = CLASSIFIER.get_tensor(CLASSIFIER_O[0]['index'])

    pred = np.squeeze(output)
    covid_p = pred[0]
    normal_p = pred[1]
    pneumonia_p = pred[2]

    return predictions_class[np.argmax(pred, axis=-1)], covid_p, normal_p, pneumonia_p


# Create your views here.

def index(request):
    return render(request, 'main/index.html')

@csrf_exempt
def index_predict(request):
    if request.method != "POST":
        return JsonResponse({
            'success': False,
            'method': request.method,
            'description': "Request method must be POST"
        })
    if request.FILES.get('image'):
        name = request.POST.get('name')
        image = request.FILES['image']

        if not validate(image):
            return JsonResponse({
                'success': False,
                'method': request.method,
                'description': "Looks like this is not a X-ray image. Please upload a valid X-ray image."
            })

        data = Patient.objects.create(name=name, xray=image)

        prediction, covid_percentage, normal_percentage, pneumonia_percentage = predict(image)
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
            'method': request.method,
            'description': "No image found"
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
                'description': "Looks like this is not a X-ray image. Please upload a valid X-ray image."
            })

        data = Patient.objects.create(name=name, xray=image)

        prediction, covid_percentage, normal_percentage, pneumonia_percentage = predict(image)
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
            prediction, covid_percentage, normal_percentage, pneumonia_percentage = predict(img_url, isurl=True)
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
                'description': "Looks like this is not a X-ray image. Please upload a valid X-ray image.",
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

@csrf_exempt
def warmup(request):
    return JsonResponse({
        'success': True,
        'method': request.method,
        'description': "Warming up"
    })