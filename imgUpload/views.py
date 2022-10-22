from django.shortcuts import render
from .forms import ImageUploadForm
import tensorflow as tf
# isort: off
from tensorflow.python.util.tf_export import keras_export
import numpy as np

# Create your views here.
def handle_uploaded_file(f):
    with open('img.jpg','wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


def home(request):
    return render(request,'home.html')

def imageprocess(request):
    form =ImageUploadForm(request.POST,request.FILES)
    if form.is_valid():

        handle_uploaded_file(request.FILES['image'])


        model = tf.keras.applications.EfficientNetB4(
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax",
            )

        img_path='img.jpg'

        image = tf.keras.preprocessing.image.load_img(img_path,target_size=(380, 380))
        input = np.array([tf.keras.preprocessing.image.img_to_array(image)])
        input = tf.keras.applications.efficientnet.preprocess_input(input)
        predict = model.predict(input)
        #print('predict:' ,decode_predictions(predict, top=5)[0])
        html =tf.keras.applications.efficientnet.decode_predictions(predict, top=3)[0]
        res =[]
        for e in html:
            res.append((e[1],np.round(e[2]*100,2)))

        return render(request,'result.html',{'res':res})
    return render(request,'home.html')
