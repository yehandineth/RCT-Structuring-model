from config.config import *
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import tf_keras as keras
import matplotlib.pyplot as plt

def get_cm_and_final_results(predictions, truth) -> tuple[ConfusionMatrixDisplay, pd.DataFrame, dict[str, float]]:
    
    """
    Generate classification metrics, confusion matrix and detailed performance report from model predictions.

    Parameters:
        predictions: Union[list, np.ndarray]
            Model predictions array/list containing predicted class labels
            Must have same length as truth
            Values must be valid class indices (0 to NUM_CLASSES-1)
        
        truth: Union[list, np.ndarray]  
            Ground truth array/list containing actual class labels
            Must have same length as predictions
            Values must be valid class indices (0 to NUM_CLASSES-1)

    Returns:
        Tuple containing:
            - ConfusionMatrixDisplay: Plotted confusion matrix visualization
            - pd.DataFrame: Detailed classification report with per-class metrics
            - Dict[str, float]: Overall performance metrics including:
                - accuracy: Overall accuracy percentage
                - precision: Weighted average precision
                - recall: Weighted average recall
                - f1: Weighted average F1 score

    Raises:
        ValueError: If:
            - predictions and truth have different lengths
            - arrays contain invalid class indices
            - empty arrays are provided
        TypeError: If inputs are not lists or numpy arrays
        NameError: If CLASS_NAMES or NUM_CLASSES are not defined
        AttributeError: If sklearn functions fail due to input format
        KeyError: If expected metrics are missing from classification report

    Example:
        >>> preds = [0, 1, 2, 1, 0]
        >>> truth = [0, 1, 1, 1, 0]
        >>> cm, metrics_df, overall = get_cm_and_final_results(preds, truth)
        >>> print(overall['accuracy'])
        80.0
        >>> metrics_df.head()
            class0  class1  class2  ...  weighted avg
        precision  1.0    0.67    0.0    0.87
        recall    1.0    1.0     0.0    0.80
        f1-score  1.0    0.80    0.0    0.83

    Notes:
        - Requires global CLASS_NAMES list/tuple for confusion matrix labels
        - Requires global NUM_CLASSES integer for validation
        - Uses sklearn.metrics for calculations
        - Returns accuracy as percentage (0-100), other metrics as decimals (0-1)
        - Confusion matrix plot uses 50 degree rotation for x-axis labels
    """

    report = classification_report(truth, predictions, output_dict=True)
    df = pd.DataFrame(report)
    names = list(CLASS_NAMES.copy())
    for i in range(NUM_CLASSES, len(df.columns)):
        names.append(df.columns[i])
    df.columns = names
    baseline_cm = confusion_matrix(truth, predictions)
    cm_display = ConfusionMatrixDisplay(baseline_cm, display_labels=CLASS_NAMES)
    cm_display.plot(xticks_rotation=50)
    output = {
        'accuracy': 100 * report['accuracy'],
        'precision': df['weighted avg']['precision'],
        'recall': df['weighted avg']['recall'],
        'f1': df['weighted avg']['f1-score']
    }
    return cm_display, df, output

def confusion_matrix_save(cm, model, location=TRAINING_DATA_DIR):
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    fig.suptitle('Confusion matrix for Data', fontsize=20)
    ax.set_title(f'Model : {model.name}', color=(0.3,0.3,0.3))
    cm.plot(ax=ax)
    ax.set_xticklabels(CLASS_NAMES,
                    fontsize=8)
    ax.set_yticklabels(CLASS_NAMES,
                    fontsize=8)
    ax.set_xlabel(xlabel='Predicted label',
                    fontsize=10,
                    color='red')
    ax.set_ylabel(
        ylabel='True label',
        fontsize=10,
        color='red'
    )
    fig.savefig(location.joinpath(f'confusion_matrix_{model.name}.png'))