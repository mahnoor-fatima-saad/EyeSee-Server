from source.utilities.directories import project_path

import tensorflow as tf
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageOps


class Disease:

    @staticmethod
    def get_detection_model_path():
        detection_model_path = 'models\\diseases\\detection_model.h5'
        return detection_model_path

    @staticmethod
    def get_model_path():
        model_path = 'models\\diseases\\eye_disease_detection.h5'
        return model_path

    def __init__(self):

        self.eye_detection_model = tf.keras.models.load_model(os.path.join(project_path, Disease.get_detection_model_path()), compile = False)
        self.eye_disease_model = tf.keras.models.load_model(os.path.join(project_path, Disease.get_model_path()), compile=False)
        self.json_file = {'is_eye': 'false', 'is_closed': 'false', 'result': '0', 'percentage': '0'}

    def preprocess_image_for_eye_check(self, image):
        image = Image.open(image)
        image.save('eye_image.jpg')
        preprocessed_image = load_img('./eye_image.jpg', target_size=(224, 224))
        preprocessed_image = img_to_array(preprocessed_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        preprocessed_image = np.array(preprocessed_image) / 255.0
        return (preprocessed_image)

    def check_eye(self, image):
        processed_image = self.preprocess_image_for_eye_check(image)
        eye_pred = self.eye_detection_model.predict(processed_image)
        print(eye_pred[0])
        # 0: Closed, 1: Eye, 2: Not eye
        label = np.argmax(eye_pred[0])
        print(eye_pred[0][label])

        if label == 1:
            if np.max(eye_pred[0][label]) > 0.80:
                self.json_file['is_eye'] = 'true'
                return True
        elif label == 0:
            if np.max(eye_pred[0][label]) > 80:
                self.json_file['closed'] = 'true'
                return False
        else:
            return False

    def preprocess_image_for_disease(self,image):
        preprocessed_image = load_img('./eye_image.jpg', target_size=(200, 200))
        preprocessed_image = img_to_array(preprocessed_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        preprocessed_image = np.array(preprocessed_image) / 255.0
        return (preprocessed_image)

    def get_disease_label(self, label):
        label = int(label)
        if label == 0:
            return 'Arcus Senilis'
        elif label == 1:
            return 'Cataracts'
        elif label == 2:
            return 'Endophthalmitis'
        elif label == 3:
            return 'Hemangioma'
        elif label == 4:
            return 'Hyphema'
        else:
            return 'Undefined'

    def check_disease(self, image):
        eye_flag = self.check_eye(image)
        if eye_flag == True:
            processed_image = self.preprocess_image_for_disease(image)
            disease_pred = self.eye_disease_model.predict(processed_image)
            print(disease_pred)
            label = np.argmax(disease_pred[0])
            if disease_pred[0][label] >0.80:
                disease = self.get_disease_label(label)
                self.json_file['result'] = disease
                percentage = disease_pred[0][label]
                percentage = round(percentage * 100, 4)
                self.json_file['percentage'] = percentage
            else:
                self.json_file['result'] = 'unknown'
        return self.json_file


