# Classifying-Clinical-Text
Classify medical transcription by subspecialty of medicine.

`data_exploration.ipynb`
- This notebook explores the data and examines the labels.
- This notebook selects the final labels we will use in the classifcation model.

  
`classification_notebook.ipynb`
- This notebook runs the main classfication logic.
- Here we performed the following steps:
  -  Preprocess the input texts using `nltk` and `spacy`
  -  Used `Tf-IDF` to create the vectors.
  -  Performed dimensionality reduction using `PCA`
  -  Performed oversampling using `SMOTE` to handle class imbalance.
  -  Trained a logistic regression classifier to predict the classes.
  -  Finally visualized the confusion matrix of the prediction results on test data.
 
## Steps to Run:
- Install the necessary requirements following `pip install -r requirements.txt`
- Download the [dataset](https://www.kaggle.com/datasets/louiscia/transcription-samples-mtsamples)
- Run the `classification_notebook.ipynb` notebook.