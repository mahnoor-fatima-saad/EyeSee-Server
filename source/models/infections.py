from source.utilities.directories import project_path

import tensorflow as tf
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageOps


class Infection:

    @staticmethod
    def get_detection_model_path():
        detection_model_path = 'models\\infections\\is_eye_infection_model.h5'
        return detection_model_path

    @staticmethod
    def get_model_path():
        model_path = 'models\\infections\\infection_dectection.h5'
        return model_path

    def __init__(self):

        self.eye_detection_model = tf.keras.models.load_model(
            os.path.join(project_path, Infection.get_detection_model_path()), compile=False)
        self.infection_model = tf.keras.models.load_model(os.path.join(project_path, Infection.get_model_path()),
                                                            compile=False)
        self.json_file = {'is_eye': 'false', 'result': '0', 'percentage': '0'}

    def preprocess_image_for_eye_check(self, image):
        image = Image.open(image)
        image.save('infection_image.jpg')
        preprocessed_image = load_img('./infection_image.jpg', target_size=(150, 150))
        preprocessed_image = img_to_array(preprocessed_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        preprocessed_image = np.array(preprocessed_image) / 255.0
        return preprocessed_image

    def check_eye(self, image):
        processed_image = self.preprocess_image_for_eye_check(image)
        eye_pred = self.eye_detection_model.predict(processed_image)
        print(eye_pred[0])
        # 0: eye, 1: not_eye
        label = np.argmax(eye_pred[0])
        print(eye_pred[0][label])

        if label == 0:
            if np.max(eye_pred[0][label]) > 0.80:
                self.json_file['is_eye'] = 'true'
                return True
        # elif label == 0:
        #     if np.max(eye_pred[0][label]) > 0.80:
        #         self.json_file['is_closed'] = 'true'
        #         return False
        else:
            return False

    def preprocess_image_for_infection(self, image):
        preprocessed_image = load_img('./infection_image.jpg', target_size=(150, 150))
        preprocessed_image = img_to_array(preprocessed_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        preprocessed_image = np.array(preprocessed_image) / 255.0
        return preprocessed_image

    @staticmethod
    def get_infection_label(label):
        label = int(label)
        if label == 0:
            return 'Blepharitis'
        elif label == 1:
            return 'Cellulitis'
        elif label == 2:
            return 'Conjunctivitis'
        elif label == 3:
            return 'Normal'
        elif label == 4:
            return 'Stye'
        else:
            return 'Undefined'

    def check_infection(self, image):

        self.json_file = {'is_eye': 'false', 'result': '0', 'percentage': '0'}

        eye_flag = self.check_eye(image)
        if eye_flag:
            processed_image = self.preprocess_image_for_infection(image)
            infection_pred = self.infection_model.predict(processed_image)
            print(infection_pred)
            label = np.argmax(infection_pred[0])
            print(label)
            if infection_pred[0][label] > 0.80:
                infection = Infection.get_infection_label(label)
                self.json_file['result'] = infection
                percentage = infection_pred[0][label]
                percentage = round(percentage * 100, 4)
                self.json_file['percentage'] = percentage
            else:
                self.json_file['result'] = 'Unknown'
                self.json_file['percentage'] = '0.0'
        return self.json_file
