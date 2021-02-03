from tensorflow import keras
from source.utilities.singleton import Singleton


@Singleton
class Fundus:
    def __init__(self):
        detection_model = keras.models.load_model('../../models/fundus/fundus_detection.h5')
        model = keras.models.load_model('../../models/fundus/fundus_disease_detection.h5')

    def check_if_fundus(self, image):
