from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import applications as keras_models

import numpy as np
from PIL import Image, ImageOps

import os

from source.utilities.directories import project_path


class Fundus:

    @staticmethod
    def get_detection_model_path():
        detection_model_path = 'models\\fundus\\fundus_detection.h5'
        return detection_model_path

    @staticmethod
    def get_model_path():
        # model_path = 'models\\fundus\\fundus_disease_detection.h5'
        model_path = 'models\\fundus\\fundus_disease_xception.h5'
        return model_path

    def __init__(self):
        self.detection_model = load_model(os.path.join(project_path, Fundus.get_detection_model_path()), compile=False)

        self.model = load_model(os.path.join(project_path, Fundus.get_model_path()), compile=False)
        self.json_file = {'is_fundus': 'false', 'result': '0', 'percentage': '0', 'predicted': True}

    def preprocess_image_for_analysis(self, image):
        image = Image.open(image)
        image.save('fundus_img.jpg')
        processed_image = load_img('./fundus_img.jpg', target_size=(224, 224))
        processed_image = img_to_array(processed_image)
        # only for Xception model as in data gen rescaling 1./255 was used
        processed_image = processed_image / 255.0
        processed_image = processed_image.reshape((1, processed_image.shape[0],
                                                   processed_image.shape[1], processed_image.shape[2]))
        # processed_image = keras_models.resnet50.preprocess_input(processed_image)
        # processed_image = keras_models.xception.preprocess_input(processed_image)
        return processed_image

    def preprocess_image_for_detection(self, image):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = Image.open(image)
        processed_image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(processed_image)
        normalised_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalised_array

        return data

    def check_if_fundus(self, image):
        # returns a 2D array
        prediction = self.detection_model.predict(image)

        # get label - 0 fundus & 1 not-fundus
        label = np.argmax(prediction[0])

        if label == 0:
            if np.max(prediction[0]) > 0.80:
                self.json_file['is_fundus'] = 'true'
                return True
        else:
            return False

    @staticmethod
    def get_analysis_label(label):
        label = int(label)
        if label == 0:
            return 'Cataract'
        elif label == 1:
            return 'Glaucoma'
        elif label == 2:
            return 'Myopia'
        elif label == 3:
            return 'Normal'
        else:
            return 'Undefined'

    def check_fundus_diseases(self, image):
        prediction = self.model.predict(image)
        index = int(np.argmax(prediction[0]))
        # get label through index
        label = Fundus.get_analysis_label(index)
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
        preprocess_for_detection = self.preprocess_image_for_detection(image)
        # check if it's an fundus image
        flag = self.check_if_fundus(preprocess_for_detection)

        if not flag:
            return self.json_file
        else:
            # Pre process image
            preprocess_for_analysis = self.preprocess_image_for_analysis(image)

            self.check_fundus_diseases(preprocess_for_analysis)
            return self.json_file
