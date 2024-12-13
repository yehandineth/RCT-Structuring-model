import tf_keras as keras
import tensorflow as tf

from processing.preprocessing import Dataset
from training.model import create_model
from config.config import *

name = 'model_test'

def main():

    test_dataset = Dataset.create_test_pipeline(DF)
    keras.mixed_precision.set_global_policy('mixed_float16')

    model = keras.models.load_model(SERIALIZATION_DIR.joinpath(f'{name}.keras'))

    predictions = model.predict(test_dataset)

if __name__ == '__main__':

    main()