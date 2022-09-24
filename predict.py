import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import argparse
import json
import os

with open('label_map.json', 'r') as f:
    class_names = json.load(f)

# Download the mobilenet model on which our h5 model is based. Must do or loading h5 model will fail
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
image_size = 224
hub.KerasLayer(URL, input_shape=(image_size, image_size,3))

# load our h5 model
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '1660148018.h5')
print(model_path)

model = tf.keras.models.load_model(
        model_path,
        custom_objects={'KerasLayer': hub.KerasLayer},
        compile=False)


# TODO: Create the process_image function
def process_image(image: np.ndarray) -> np.ndarray:
    image_tensor = tf.convert_to_tensor(image)
    image_resized = tf.image.resize(image_tensor, (224, 224))
    image_resized /= 255

    return image_resized.numpy()


# TODO: Create the predict function
def predict(image_path, top_k=5):
    im = Image.open(image_path)
    image_arr = np.asarray(im)

    processed_image = process_image(image_arr)

    processed_image = np.expand_dims(processed_image, 0)

    ps = model.predict(processed_image)

    #print(ps[0])

    top_k_indices = np.argsort(ps[0])[-top_k:]
    top_k_indices = top_k_indices[::-1]
    #print(top_k_indices)

    top_ps = ps[0][top_k_indices]

    # classes are zero based, add 1
    top_classes = []
    for index in top_k_indices:
        top_classes.append(str(int(index) + 1))

    classes_names_list = []
    for key in top_classes:
        classes_names_list.append(class_names[key])

    result = {}

    for class_name, ps in zip(classes_names_list, top_ps.tolist()):
        result[class_name] = round(ps, 4)

    return result
