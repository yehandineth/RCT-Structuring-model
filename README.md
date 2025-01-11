# Deep Neural Network for adding structure to RCT Abstracts of Medical Research Papers

Welcome to the RCT Abstract Classification Model repository! This project focuses on classifying sentences within research abstracts into predefined categories using deep learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This project aims to give me hands-on experience building models by replicating SOTA deep learning model, **Neural Networks for Joint Sentence Classification
in Medical Paper Abstracts**. 


## Features

- **Sentence Classification**: Categorizes each sentence in a research abstract into predefined labels.
- **User-Friendly Interface**: Designed for ease of use, allowing users to input text and receive structured outputs.
- **Extensibility**: Modular design enables easy updates and integration with other tools.
- **Visualizations and metrics**: Great visualizations depicting the model architecture and model performance.

## Setup

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yehandineth/NLP_Learning/RCTabstractmodel.git
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download necessary datasets and pretrained models**:
   - Ensure you have the datasets in the `datasets/` directory.
   - Place pretrained models in the `serialization/` directory.

## Usage

After setting up, you can use the model as follows:

1. **Train the model** (if training from scratch):
   ```bash
   python train.py 
   ```

2. **Classify sentences in a new abstract**:
   ```bash
   python classify.py --input data/sample_abstract.txt --output results/output.json
   ```

3. **Evaluate the model**:
   ```bash
   python evaluate.py --config configs/eval_config.yaml
   ```

## Project Structure

```
RCTabstractmodel/
├── cache/                  # Cache dir
├── config/                 # Configuration files before usage
├── datasets/               # Datasets for training dev and testing
├── processing/             # Pipeline for data and preprocessing 
├── serialization/          # Saved components after training 
├── testing/                # Testing model
├── training/               # Training model
├       ├── checkpoints/               # Training checkpoints
├       ├── training_data/             # Training history and data
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
└── LICENSE                 # License information
```

## Technologies Used

- **Python**: Programming language
- **PyTorch**: Deep learning framework
- **NLTK**: Natural Language Toolkit for text processing
- **Scikit-learn**: Machine learning utilities
- **Jupyter Notebook**: For experimentation and prototyping

## Acknowledgments

- The dataset for this project was sourced from [PubMed 200k RCT](https://arxiv.org/pdf/1710.06071).
- The model structure was inspired by the architecture described in [Neural Networks for Joint Sentence Classification
in Medical Paper Abstracts
](https://arxiv.org/pdf/1612.05251).
- The whole project is inspired by a hands-on lesson from [Tensorflow for Deep Learning Bootcamp](https://www.udemy.com/share/104ssS3@pZvYAlDoowD1INBPOwpbcXzvGkDMfmvOq6RrSKFxcHN0ocwWgWyIWZMscDlGYzCWeg==/)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue or contact [Yehan Dineth](mailto:ydinethw@gmail.com).


>All content is for learning purposes only.