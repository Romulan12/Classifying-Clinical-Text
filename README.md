# Classifying-Clinical-Text
This repository contains code for classifying medical transcriptions based on subspecialty of medicine.


`data_exploration.ipynb`
- This notebook explores the data and examines the labels. It helps in selecting the final labels to be used in the classification model.

`classification_notebook.ipynb`
- This notebook implements the main classification logic. The following steps are performed in this notebook:
    - Preprocess the input texts using NLTK and spaCy libraries.
    - Use Tf-IDF to create vectors representing the texts.
    - Perform dimensionality reduction using PCA.
    - Handle class imbalance through oversampling using SMOTE.
    - Train a logistic regression classifier to predict the classes.
    - Visualize the confusion matrix to evaluate the performance of the classifier on the test data.


## Steps to Run:
  - Install the necessary requirements by executing pip install -r requirements.txt.
  - Download the [dataset](https://www.kaggle.com/datasets/louiscia/transcription-samples-mtsamples). 
  - Run the classification_notebook.ipynb notebook to execute the classification pipeline.
  > Please refer to the respective notebooks for detailed explanations and code implementation.