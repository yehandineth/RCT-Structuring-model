import tf_keras as keras
import tensorflow as tf
import pickle

from processing.preprocessing import Dataset
from model import create_model
from testing import evaluation
from config.config import *

def main():

    train_dataset = Dataset.create_training_pipeline(DF)
    val_dataset = Dataset.create_validation_pipeline(DF)
    keras.mixed_precision.set_global_policy('mixed_float16')

    model = create_model()

    history = model.fit(
        train_dataset.pipeline.prefetch(tf.data.AUTOTUNE),
        validation_data=val_dataset.pipeline.prefetch(tf.data.AUTOTUNE),
        epochs=NUM_EPOCHS,
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
    with open(TRAIN_DIR.joinpath('history.pkl'), mode='wb') as file:
        pickle.dump(history, file)

    predictions = val_dataset.predict(model=model)

    cm,report,mets =evaluation.get_cm_and_final_results(predictions, val_dataset.y)
    print('Results by evaluation of dev set\n', mets)
    cm.im_

if __name__ == '__main__':

    main()