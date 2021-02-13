from source.utilities.directories import project_path
import os
import tensorflow as tf


class Infection():
    @staticmethod
    def get_model_path():
        model_path = 'models\\infections\\infection_detection.h5'
        return model_path

    def __init__(self):
        self.model = tf.keras.Models.load_model(os.path.join(project_path, Infection.get_model_path()), compile=False)
        self.json_file = {'result': '0', 'percentage': '0', 'predicted': False}

