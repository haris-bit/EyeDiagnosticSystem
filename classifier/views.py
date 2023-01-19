# classifier/views.py

from django.shortcuts import render
from django.contrib import messages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

def page1(request):
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' not in request.FILES:
            messages.error(request, 'No image selected')
            return render(request, 'classifier/page1.html')

        # Load the deep learning model
        model = load_model('hehe6401.h5')

        # Convert the uploaded image to a numpy array
        image = request.FILES['image']
        image = img_to_array(Image.open(image).resize((64, 64)))
        image = np.expand_dims(image, axis=0)

        # Classify the image
        classification = model.predict(image)
        
        # Check if the probability of the predicted class is below 30%
        if np.amax(classification, axis=1)< 0.3:
            messages.error(request, 'Eye fundus scan not found')
            return render(request, 'classifier/page1.html')
        else:
            classification = np.argmax(classification, axis=1)
            prediction = None
            if classification == 0:
                prediction = 'normal eye'
            elif classification == 1:
                prediction = 'cataract'
            elif classification == 2:
                prediction = 'glaucoma'

        

        # Render the result template with the classification prediction
        return render(request, 'classifier/page2.html', {'prediction': prediction})
    else:
        # Render the upload template
        return render(request, 'classifier/page1.html')


def page2(request, prediction):
    # Render the 'page2.html' template with the prediction as a context variable
    return render(request, 'classifier/page2.html', {'prediction': prediction})