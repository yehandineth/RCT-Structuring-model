import tf_keras as keras
import tensorflow as tf
import pickle

from processing.preprocessing import Dataset
from training.model import create_model
from config.config import *

name = 'model_test'


#TODO 
def main():

    test_dataset = Dataset.create_test_pipeline(DF)
    keras.mixed_precision.set_global_policy('mixed_float16')

    model = keras.models.load_model(
        filepath=SERIALIZATION_DIR.joinpath(f'{name}.keras')
        )

    predictions = test_dataset.predict(model=model)

    

if __name__ == '__main__':

    main()