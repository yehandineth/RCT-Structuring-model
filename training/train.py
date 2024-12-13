import tf_keras as keras
import tensorflow as tf

from processing.preprocessing import Dataset
from training.model import create_model
from config.config import *

def main():

    train_dataset = Dataset.create_training_pipeline(DF)
    val_dataset = Dataset.create_validation_pipeline(DF)
    keras.mixed_precision.set_global_policy('mixed_float16')

    model = create_model()

    history = model.fit(
        train_dataset.pipeline.prefetch(tf.data.AUTOTUNE),
        validation_data=val_dataset.pipeline.prefetch(tf.data.AUTOTUNE),
        initial_epoch=0,
        epochs=3,
        shuffle=False,
        batch_size= BATCH_SIZE,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=CHECKPOINTS_DIR.joinpath(f'{model.name}.weights.h5'), 
                save_best_only=True, 
                save_weights_only=True, 
                monitor='val_accuracy', 
                verbose=1
            )
        ],
    )

if __name__ == '__main__':

    main()