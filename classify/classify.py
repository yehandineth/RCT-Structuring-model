import sys
import os
from pathlib import Path
from tf_keras.mixed_precision import set_global_policy
from tf_keras.models import load_model
import numpy as np
from argparse import ArgumentParser
import datetime

sys.path.append(os.path.abspath(Path(__file__).parent.parent))

from processing.preprocessing import Abstract
from config.config import *

#TODO #preserve capitalizations as before
def main():
    args = arguments()
    abstract = Abstract.from_terminal_input()
    classified = classify(abstract)
    if args.output:
        to_file(classified, MAIN_DIR.joinpath('classify').joinpath('classified').joinpath(args.output))
        print('Successfully saved to', args.output)
    else:
        print(classified)

def to_file(classified: dict , path: Path):
    with open(path, 'w') as f:
        for cls, texts in classified.items():
            f.write(f"{cls}\n\n")
            f.write("\n".join(texts))
            f.write('\n')
            f.write('\n')

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

def arguments():
    parser = ArgumentParser(
        description='Classify abstracts',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Output file path',
        nargs='?',
        const=f'classified_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.txt',
        default=None
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()