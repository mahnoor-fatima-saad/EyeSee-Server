from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import numpy as np
from PIL import Image, ImageOps

import os

from source.utilities.directories import project_path


class Disorders:

    @staticmethod
    def get_model_path():
        model_path = 'models\\disorders\\disorder_detection.h5'
        return model_path

    def __init__(self):
        self.model = load_model(os.path.join(project_path, Disorders.get_model_path()), compile=False)
        self.json_file = {'result': '0', 'percentage': '0', 'predicted': False}
