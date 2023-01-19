# app/utils.py

import tensorflow as tf
import numpy as np

def predict_eye_class(image_path):
    # Load the model from the file 'eye.h5'
    model = tf.keras.models.load_model('E:\\fyp\\hehe6401.h5')

    # Load the image from the given file path and preprocess it for model prediction
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    image = tf.keras.preprocessing.image.img_to_array(image)

    image = np.expand_dims(image, axis=0)

    # Use the model to predict the class of the image
    prediction = model.predict(image)

    # Return the predicted class
    return prediction
