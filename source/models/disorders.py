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

    @staticmethod
    def get_detection_model_path():
        detection_model_path = 'models\\disorders\\disorder_is_eye.h5'
        return detection_model_path

    def __init__(self):
        self.detection_model = load_model(os.path.join(project_path, Disorders.get_detection_model_path()), compile=False)
        self.model = load_model(os.path.join(project_path, Disorders.get_model_path()), compile=False)
        self.json_file = {'result': '0', 'percentage': '0', 'predicted': True, 'is_eye': True}

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

    def preprocess_image_for_detection(self, image):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = Image.open(image)
        processed_image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(processed_image)
        normalised_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalised_array
        return data

    def preprocess_image(self, image):
        image = Image.open(image)
        image.save('disorder_img.jpg')
        processed_image = load_img('./disorder_img.jpg', target_size=(151, 332))
        processed_image = img_to_array(processed_image)
        processed_image = processed_image.reshape((1, processed_image.shape[0],
                                                   processed_image.shape[1], processed_image.shape[2]))
        processed_image = preprocess_input(processed_image)
        return processed_image

    def check_if_eyes(self, image):
        # returns a 2D array
        prediction = self.detection_model.predict(image)

        # get label - 0 eye & 1 not-eye
        label = np.argmax(prediction[0])

        if label == 0:
            if np.max(prediction[0]) > 0.80:
                return True
        else:
            return False

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
        # Pre process image for detection
        preprocess_for_detection = self.preprocess_image_for_detection(image)
        # check if it's an eye image
        flag = self.check_if_eyes(preprocess_for_detection)
        if not flag:
            self.json_file['is_eye'] = False
            return self.json_file
        else:
            # Pre process image
            preprocess_for_detection = self.preprocess_image(image)
            # Predict
            self.check_disorders(preprocess_for_detection)
            return self.json_file
