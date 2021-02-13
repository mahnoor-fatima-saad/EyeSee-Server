from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import numpy as np
from PIL import Image

import os

from source.utilities.directories import project_path


class Disorders:

    @staticmethod
    def get_model_path():
        model_path = 'models\\disorders\\disorder_detection.h5'
        return model_path

    def __init__(self):
        self.model = load_model(os.path.join(project_path, Disorders.get_model_path()), compile=False)
        self.json_file = {'result': '0', 'percentage': '0', 'predicted': True}

    @staticmethod
    def get_analysis_label(label):
        label = int(label)
        if label == 0:
            return 'Bulging Eye'
        elif label == 1:
            return 'Cataract'
        elif label == 2:
            return 'Crossed Eye'
        else:
            return 'Undefined'

    def preprocess_image(self, image):
        image = Image.open(image)
        image.save('disorder_img.jpg')
        processed_image = load_img('./disorder_img.jpg', target_size=(151, 332))
        processed_image = img_to_array(processed_image)
        processed_image = processed_image.reshape((1, processed_image.shape[0],
                                                   processed_image.shape[1], processed_image.shape[2]))
        processed_image = preprocess_input(processed_image)
        return processed_image

    def check_disorders(self, image):
        prediction = self.model.predict(image)
        index = int(np.argmax(prediction[0]))
        # get label through index
        label = Disorders.get_analysis_label(index)
        # update json
        self.json_file['result'] = label
        # get percentage
        percentage = prediction[0][index]
        percentage = round(percentage * 100, 4)
        if percentage < 80.0:
            self.json_file['predicted'] = False
            print('Percentage: ' + str(percentage))
            return
        self.json_file['percentage'] = str(percentage)

    def prediction(self, image):
        # Pre process image
        preprocess_for_detection = self.preprocess_image(image)
        # Predict
        self.check_disorders(preprocess_for_detection)
        return self.json_file
