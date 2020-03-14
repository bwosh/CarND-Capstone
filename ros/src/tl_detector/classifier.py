import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, model_from_json

class LightClassifier():
    def __init__(self):
        # Set TF configuration
        config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25))
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        # Load model
        K.set_learning_phase(0)
        with open("./data/model.json", 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        weights_path = "./data/model.h5"
        model.load_weights(weights_path)

        # keras fix
        config = model.get_config()
        model = Sequential.from_config(config)        
        
        # compile requirement for inrefence...
        model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['acc'])
        K.set_learning_phase(0)

        self.model = model

    def classify(self, input):
        result = model.predict(input)[0]
        return result