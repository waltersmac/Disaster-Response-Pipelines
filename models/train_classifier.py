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



def load_data(database_filepath):

    '''

    '''

    # Load data
    db_uri = 'sqlite:///{}'.format(database_filepath)

    # load data from database
    engine = create_engine(db_uri)
    df = pd.read_sql_table('ResponseTable', con=engine)

    df = df.set_index('id')
    X = df['message']
    y = df.iloc[:,3:]

    category_names = list(y.columns)

    return X, y, category_names


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



def build_model(X_train, y_train):

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

    # cross validation for F1 score
    f1_results = {}

    # Cross-validation of each pipeline
    for pipename, pipevalue in pipeline_dict.items() :
        start_time = time.time()
        print ("Training pipeline : {} ...".format(pipename))
        scores = cross_val_score(pipevalue, X_train, y_train, scoring='f1_weighted', cv=5)
        f1_results[pipename] = scores.mean()
        print ("Pipeline : {} F1 mean score {}".format(pipename, scores.mean()))
        time_taken = round(((time.time() - start_time) / 60),3)
        print("--- " + str(time_taken) + " minutes ---")


    # Best pipeline
    best_pipeline_name = max(f1_results, key=f1_results.get)
    best_pipeline = pipeline_dict[best_pipeline_name]


    return best_model_tuned


def evaluate_model(model, X_test, y_test, category_names):

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
    best_model_tuned = search.best_estimator_

    y_pred = search.predict(X_test)

    class_report = classification_report(y_test, y_pred, target_names = category_names)

    print(class_report)

    return best_model_tuned



def save_model(model, model_filepath):

    '''

    '''

    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model(X_train, y_train)

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()