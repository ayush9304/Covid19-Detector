from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse
from .models import *
from django.conf import settings
from django.contrib.sites.shortcuts import get_current_site

from .utils import *

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from django.core.files.base import ContentFile
import random
import string
import base64
import os

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
    del img
    return output[0][0] >= 0.74


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

    CLASSIFIER.set_tensor(CLASSIFIER_I[0]['index'], X1_norm)
    CLASSIFIER.invoke()
    output = CLASSIFIER.get_tensor(CLASSIFIER_O[0]['index'])

    pred = np.squeeze(output)
    covid_p = pred[0]
    normal_p = pred[1]
    pneumonia_p = pred[2]

    del img
    del X
    del X_norm
    del X1_norm
    del segment
    del e_lung
    del ekernal
    del dkernal
    del pmask
    del fmask
    del x_g
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

        if request.user.is_authenticated:
            data = Scan.objects.create(user=request.user, name=name, xray=image)
        else:
            data = Scan.objects.create(name=name, xray=image)

        prediction, covid_percentage, normal_percentage, pneumonia_percentage = predict(image)
        data.prediction = prediction
        data.covid_percentage = covid_percentage
        data.normal_percentage = normal_percentage
        data.pneumonia_percentage = pneumonia_percentage
        data.save()

        if "127.0.0.1" in get_current_site(request).domain:
            uri = "http://127.0.0.1:8000/media/"
        else:
            uri = "https://covid19-detection-api.herokuapp.com/media/"

        return JsonResponse({
            'success': True,
            'method': request.method,
            'name': data.name,
            'description': "Successfully uploaded",
            'covid_percentage': float(data.covid_percentage),
            'normal_percentage': float(data.normal_percentage),
            'pneumonia_percentage': float(data.pneumonia_percentage),
            'prediction': data.prediction,
            'image_url': uri + str(data.xray)
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

        if request.user.is_authenticated:
            data = Scan.objects.create(user=request.user, name=name, xray=image)
        else:
            data = Scan.objects.create(name=name, xray=image)

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
        
        data = Scan.objects.create()
        data.xray.save(filename, imgdata, save=True)  # xray is Scan model's ImageField

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

def register_fn(request):
    if request.method == "POST":
        fname = request.POST['firstname']
        lname = request.POST['lastname']
        username = request.POST["username"]
        email = request.POST["email"]

        # Ensuring password matches confirmation
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]
        if password != confirmation:
            return render(request, "main/register.html", {
                "message": "Passwords must match."
            })

        # Attempt to create new user
        try:
            user = User.objects.create_user(username, email, password)
            user.first_name = fname
            user.last_name = lname
            user.save()
        except:
            return render(request, "main/register.html", {
                "message": "Username already taken."
            })
        login(request, user)
        return HttpResponseRedirect(reverse("index"))
    else:
        return render(request, "main/register.html")

def login_fn(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse("index"))
            
        else:
            return render(request, "main/login.html", {
                "message": "Invalid username and/or password.",
                "username": username,
                "rerun": True
            })
    else:
        if request.user.is_authenticated:
            return HttpResponseRedirect(reverse('index'))
        else:
            return render(request, "main/login.html", {
                "rerun": False
            })

def logout_fn(request):
    logout(request)
    return HttpResponseRedirect(reverse("index"))

def scans(request):
    if request.user.is_authenticated:
        results = Scan.objects.filter(user=request.user).order_by('-scan_date')
        return render(request, "main/scans.html", {
            "scans": results
        })
    return HttpResponseRedirect(reverse("login"))

@csrf_exempt
def delete_scan(request, scan_id):
    if request.user.is_authenticated:
        if request.method == "DELETE":
            scan = Scan.objects.get(id=scan_id)
            if scan.user == request.user:
                scan.delete()
                return JsonResponse({
                    'success': True,
                    'description': "Scan deleted"
                })
            else:
                return JsonResponse({
                    'success': False,
                    'description': "Scan not found"
                })
        else:
            return JsonResponse({
                'success': False,
                'description': "Request method must be DELETE"
            })
    return HttpResponseRedirect(reverse("login"))