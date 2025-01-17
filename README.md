# RCT Abstract Classification Model

A deep learning approach to identify and categorize sentences in Randomized Controlled Trial (RCT) abstracts.

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

This project focuses on classifying sentences in medical research abstracts into predefined categories. Inspired by research on neural networks for sentence classification, it aims to streamline abstract structuring.

## Features

- Automated sentence classification
- Configurable for multiple labels
- Extensible codebase for broader use cases
- Convenient visualizations of performance

## Setup

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yehandineth/RCT-Structuring-model
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
   python training/train.py 
   ```

2.  **Testing the model** (if training from scratch)
   ```bash
   python testing/test.py 
   ```

3. **Classify sentences in a new abstract** (Usage from terminal)
   ```bash
   python classify/classify.py  --output file_name.txt
   ```
   And type in your abstract in the terminal

4. **Running the web app** (You can view the saved metrics model images and much more using this web app)
   ```bash
   streamlit run classifier.py
   ```

## Project Structure

```
RCT-Structuring-model/
├── classify/
├── config/
├── datasets/
├── pages/
├── processing/
├── serialization/
├── testing/
├── training/
├── classifier.py
├── CONTRIBUTING.md
├── requirements.txt
└── README.md
```

## Technologies Used

- **Python**: Programming language
- **Tensorflow and Keras**: Deep learning framework
- **Spacy**: NLP framework 

## Acknowledgments

- The dataset for this project was sourced from [PubMed 200k RCT](https://arxiv.org/pdf/1710.06071).
- The model structure was inspired by the architecture described in [Neural Networks for Joint Sentence Classificationin Medical Paper Abstracts](https://arxiv.org/pdf/1612.05251).
- The whole project is inspired by a hands-on lesson from [Tensorflow for Deep Learning Bootcamp](https://www.udemy.com/share/104ssS3@pZvYAlDoowD1INBPOwpbcXzvGkDMfmvOq6RrSKFxcHN0ocwWgWyIWZMscDlGYzCWeg==/)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## Contact

Open an issue or email [Yehan Dineth](mailto:ydinethw@gmail.com) with questions.

>All content is for learning purposes only.