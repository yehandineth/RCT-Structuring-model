import sys
import os
from pathlib import Path
from tf_keras.mixed_precision import set_global_policy
from tf_keras.models import load_model
import numpy as np

sys.path.append(os.path.abspath(Path(__file__).parent.parent))

from processing.preprocessing import Abstract
from config.config import *

def main():
    abstract = Abstract.from_terminal_input()
    final = classify(abstract)
    print(*final)

def classify(abstract: Abstract):
        
    outputs = get_labels(dataset=abstract)
    classified = {}
    for x,y in zip(abstract.text, np.array(CLASS_NAMES)[outputs]):
        if y not in classified.keys():
            classified[y] = [x]
        else:
            classified[y].append(x)
    return classified


def get_labels(dataset :Abstract):
    set_global_policy('mixed_float16')

    model = load_model(
        filepath=SERIALIZATION_DIR.joinpath(f'{NAME}.keras')
        )

    return dataset.predict(model=model)

if __name__ == '__main__':
    
    main()