import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('ResponseTable', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
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
@app.route('/go')
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
