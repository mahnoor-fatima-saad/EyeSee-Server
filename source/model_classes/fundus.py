from source.utilities.singleton import Singleton

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input


@Singleton
class Fundus:

    def __init__(self):
        self.detection_model = load_model('../../models/fundus/fundus_detection.h5')
        self.model = load_model('../../models/fundus/fundus_disease_detection.h5')
        self.json_file = {'is_fundus': 'false', 'cataract': '0', 'glaucoma':'0', 'myopia': '0', 'normal':'0'}

    def preprocess_image(self, image):
        processed_image = load_img(image, target_size=(224, 224))
        processed_image = img_to_array(processed_image)
        processed_image = processed_image.reshape((1, processed_image.shape[0],
                                         processed_image.shape[1], processed_image.shape[2]))
        processed_image = preprocess_input(processed_image)

        return processed_image

    def check_if_fundus(self, image):
        return True if self.detection_model.predict(image) > 0.80 else False

    def prediction(self, image):
        # Pre process image
        finalised_image = self.preprocess_image(image)

        # check if it's an fundus image


