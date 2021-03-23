# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd

import pickle
from datetime import datetime

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import set_config    # To change display


def tokenize(text):

    '''

    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def model_pipeline(tokenize):

    '''

    '''

    # Random Forest Classifier - pipeline
    pipeline_rf = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=5)))
    ])

    # LGBMC
    pipeline_lgb = Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
        ('multiclassifier',MultiOutputClassifier(LGBMClassifier(n_jobs=5)))
    ])

    #
    pipeline_chain_lgb = Pipeline([
        ('tfidfvect', TfidfVectorizer(tokenizer=tokenize)),
        ('classifierchain',ClassifierChain(LGBMClassifier(n_jobs=5)))
    ])

    # Pipeline dictionary
    pipeline_dict = {'pipeline_rf': pipeline_rf, \
                     'pipeline_lgb ': pipeline_lgb, \
                     'pipeline_chain_lgb': pipeline_chain_lgb}

    return pipeline_dict



def display_scores(model_dict, X_train, y_train):

    """
    Function for applying the cross validation score

    Parameters:

    model, X_train, y_train

    Returns:

    f1 - Scores and Average

    """

    # cross validation for F1 score
    results = {}

    for pipename, pipevalue in pipeline_dict.items() :
        start_time = time.time()
        print ("Training pipeline : {} ...".format(pipename))
        scores = cross_val_score(pipevalue, X_train, y_train, scoring='f1_weighted', cv=5)
        results[pipename] = scores.mean()
        print ("Pipeline : {} F1 mean score {}".format(pipename, scores.mean()))
        time_taken = round(((time.time() - start_time) / 60),3)
        print("--- " + str(time_taken) + " minutes ---")

    return results



def train_data(f1_scores, best_pipeline, X_train, y_train):

    '''


    '''

    myscoring = make_scorer(f1_scores,average='weighted')

    parameters = {
        'tfidfvect__ngram_range': ((1, 1), (1, 2)),
        'tfidfvect__max_df': (0.5, 0.75, 1.0),
        'tfidfvect__max_features': (None, 100, 500, 2000)
        }

        # create grid search object
        search = RandomizedSearchCV(best_pipeline, parameters, scoring=myscoring, verbose = 2)

        search.fit(X_train, y_train)

        best_model_tuned = search.best_estimator_

        return best_model_tuned



def save_model(best_model_tuned):

    '''


    '''

    model_dir = '../models/'

    #Get the date to time stamp model
    today = datetime.now()
    timestamp = today.strftime("%b-%d-%y-%Hh%M")

    model_file = best_pipeline_name+'-'+timestamp+'.pkl'
    print("Saving model {} to directory {}".format(model_file,model_dir))
    pickle.dump(best_model_tuned, open(model_dir+model_file, 'wb'))



def process_data():

    '''


    '''

    # Load data
    db_path = os.path.join(os.path.abspath('../data/processed/'), 'ResponseDatabase.db')
    db_uri = 'sqlite:///{}'.format(db_path)

    # load data from database
    engine = create_engine(db_uri)
    df = pd.read_sql_table('ResponseTable', con=engine)
    df = df.set_index('id')
    X = df['message']
    y = df.iloc[:,3:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline_dict = pipeline_dict()

    f1_scores = display_scores(pipeline_dict, X_train, y_train)

    # Train best pipline
    best_pipeline = pipeline_dict[best_pipeline_name]
    best_model_tuned = train_data(f1_scores, best_pipeline, X_train, y_train)

    save_model(best_model_tuned)
