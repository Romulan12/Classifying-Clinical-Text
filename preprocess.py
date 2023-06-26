import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

from imblearn.over_sampling import SMOTE

import spacy

nlp = spacy.load(
    "en_ner_bionlp13cg_md"
)  ##using the sci-spacy library to extract biomedical entities


def sanitize_text(text):
    """
    Args:
        text (str): the text to be sanitized
    """
    text = text.translate(str.maketrans("", "", string.punctuation))
    text1 = "".join([w for w in text if not w.isdigit()])
    REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")

    text2 = text1.lower()
    text2 = REPLACE_BY_SPACE_RE.sub(
        "", text2
    )  # replace REPLACE_BY_SPACE_RE symbols by space in text
    return text2


def lemmatize_text(text):
    """
    Args:
        text (str): the text to be lemmatized
    """
    wordlist = []
    lemmatizer = WordNetLemmatizer()
    sentences = sent_tokenize(text)

    intial_sentences = sentences[0:1]
    final_sentences = sentences[len(sentences) - 2 : len(sentences) - 1]

    for sentence in intial_sentences:
        words = word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    for sentence in final_sentences:
        words = word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    return " ".join(wordlist)


def extract_entities(text):
    """
    Args:
        text (str): the text to extract entities from
    """
    wordlist = []
    doc = nlp(text)

    for ent in doc.ents:
        wordlist.append(ent.text)

    return " ".join(wordlist)


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess_text(text):
    preprocessed_keywords = []
    text = decontracted(text)
    text = re.sub("\S*\d\S*", "", text).strip()
    text = re.sub("[^A-Za-z]+", " ", text)
    text = " ".join(e.lower() for e in text.split() if e.lower() not in stopwords)
    text = lemmatize_sentence(sentence)

    return text.strip()
