import tf_keras as keras
import tensorflow as tf
import numpy as np
import pickle
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(Path(__file__).parent.parent))

from processing.preprocessing import Dataset
from model import create_model
from testing import evaluation
from config.config import *

import warnings
warnings.filterwarnings("ignore")

def main():

    train_dataset = Dataset.create_training_pipeline(DF)
    val_dataset = Dataset.create_validation_pipeline(DF)
    keras.mixed_precision.set_global_policy('mixed_float16')

    model :keras.Model = create_model(name=NAME)

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
                verbose=1,
            )
        ],
    )
    model.load_weights(CHECKPOINTS_DIR.joinpath(f'{model.name}.weights.h5'))

    model.save(SERIALIZATION_DIR.joinpath(f'{model.name}.keras'))

    with open(TRAINING_DATA_DIR.joinpath(f'training_history_{model.name}.pkl'), mode='wb') as file:
        pickle.dump(history, file)

    predictions = val_dataset.predict(model=model)

    cm,report,mets =evaluation.get_cm_and_final_results(predictions, val_dataset.y)
    print('Results by evaluation of dev set\n', mets)
    report.to_csv(TRAINING_DATA_DIR.joinpath(f'classification_report_{model.name}.csv'))
    evaluation.confusion_matrix_save(cm,model)

    print('Checking Saved model integrity.....')

    loaded = keras.models.load_model(
        filepath=SERIALIZATION_DIR.joinpath(f'{model.name}.keras')
        )
    loaded_preds = val_dataset.predict(model=loaded)
    close = np.isclose(predictions,loaded_preds)
    print('Integrity test:', 'passed' if close.all() else 'fail')



if __name__ == '__main__':

    main()