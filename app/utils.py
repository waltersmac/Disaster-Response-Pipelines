# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd

import pickle
import time

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def tokenize(text):

    """
    Function for process the text data
    so that it is ready for the training

    Parameters:

    Text

    Returns:

    Cleaned data

    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
