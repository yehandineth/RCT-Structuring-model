import tf_keras as keras
from tf_keras.optimizers import Adam
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).parent.parent))


from processing.preprocessing import Dataset
from config.config import *
import evaluation
from training.model import create_model, load_model


def main():

    test_dataset = Dataset.create_test_pipeline(DF)
    keras.mixed_precision.set_global_policy('mixed_float16')

    model = load_model(name=NAME)

    predictions = test_dataset.predict(model=model)

    cm,report,mets =evaluation.get_cm_and_final_results(predictions, test_dataset.y)
    print('Results for test set\n', mets)
    report.to_csv(TEST_DIR.joinpath(f'classification_report_{model.name}.csv'))
    evaluation.confusion_matrix_save(cm,model, location=TEST_DIR)
    
if __name__ == '__main__':

    main()