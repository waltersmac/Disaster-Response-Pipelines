#!/usr/bin/env python3
# import libraries
import os
import sys
import json
import plotly
import numpy as np
import pandas as pd
import pickle
import boto3
import boto3.session
from secret import access_key, secret_access_key

import warnings
warnings.filterwarnings('ignore')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine
from flask_sqlalchemy import SQLAlchemy
from utils import tokenize


application = Flask(__name__)

# load data
print("loading messages from database ...")
db_path = os.path.realpath('data/DisasterResponse.db')
db_uri = 'sqlite:///{}'.format(db_path)

application.config['SQLALCHEMY_DATABASE_URI'] = db_uri
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(application)

db.init_app(application)
conn = db.engine.connect().connection

sql = "SELECT * from ResponseTable"
df = pd.read_sql(sql, conn)

# load model
print("loading model ...")
model_path = os.path.realpath('models/classifier.pkl')
model = pickle.load(open(model_path, 'rb'))

# index webpage displays cool visuals and receives user input text for model
@application.route('/')
@application.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message'].sort_values()
    genre_names = list(genre_counts.index)

    col = df.columns[4:].to_list()
    categories_counts = ((df[col].sum()).sort_values())[-10:]
    categories_names = list(categories_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    rotation=180,
                    marker=dict(colors=['gold', 'mediumturquoise', 'darkorange', 'lightgreen'],
                    line=dict(color='#000000', width=2))
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genre',
                'height': 400,
                'width': 600,
                'legend': dict(yanchor="top",y=0.99,xanchor="left",x=0.01),
                'margin': dict(l=50)
            }
        }
    ]

    graphs += [
        {
            'data': [
                Bar(
                    y=categories_names,
                    x=categories_counts,
                    name='Count',
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'The Top 10 Message Categories',
                'height': 400,
                'width': 700,
                'showlegend': True,
                'legend': dict(yanchor="top",y=0.20,xanchor="right",x=0.99),
                'margin': dict(l=110,r=100,pad=5)
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@application.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    application.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
