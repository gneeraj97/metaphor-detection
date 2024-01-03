# Metaphor_Detection

## Introduction

Welcome to the project! This repository contains code for training and testing a machine learning model. Follow the instructions below to set up the environment, train the model, and test it.

## Installation

To install the required dependencies, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

This command installs the necessary Python packages listed in the requirements.txt file.

## Training
To train the model, follow these steps:

Make sure you have installed the required dependencies using the command mentioned in the "Installation" section.

Run the training script using the following command:

```bash
python run_train.py train.csv
```

Replace train.csv with the path to your training data file.

The training process will download four files into the current working directory:

trained_model.pkl: This file contains the trained machine learning model.
trained_LDA.pkl: This file contains the trained Latent Dirichlet Allocation (LDA) model.
vectorizer.joblib: This file contains the vectorizer used to transform text data.
vocabulary.joblib: This file contains the vocabulary learned during training.

## Testing
To test the trained model, follow these steps:

Make sure you have installed the required dependencies using the command mentioned in the "Installation" section.

Run the testing script using the following command:

```bash
python run_test.py test.csv
```
Replace test.csv with the path to your testing data file.

This command will evaluate the trained model on the testing data and output a data frame with the column name - label_boolean, similar to the training data.

Feel free to reach out if you encounter any issues or have questions. 
