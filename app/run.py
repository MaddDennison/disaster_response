import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Scatter, Histogram
from sklearn.externals import joblib
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
engine = create_engine('sqlite:///../data/disaster_response_tweets.db')
df = pd.read_sql_table('categorized_tweets', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    cat_names = list(df.iloc[:0, 4:])
    cat_counts = df[cat_names].sum()

    df['msg_len'] = df.message.str.len()
    msg_lengths = list(df['msg_len'])


    genre_msg_length = df.groupby('genre').mean()['msg_len']


    # create visuals
    graphs = [
        #Graph 1 ####################################################################################
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    textinfo='label+percent',
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',

            }
        },
        #Graph 2 ####################################################################################
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of messages by category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories",
                    'categoryorder': "total descending"
                }
            }
        },
        #Graph 3 ####################################################################################
        {
            'data': [
                Histogram(
                    x=msg_lengths,
                    xbins=dict(
                        start=-0,
                        end=500,
                        size=5
                    )
                )
            ],

            'layout': {
                'title': 'Histogram of message lengths',
                'yaxis': {
                    'title': "Counts"

                },
                'xaxis': {
                    'title': "Value"
                }
            }
        },
        #Graph 4 ####################################################################################
         {
            'data': [
                Scatter(
                    x=genre_counts,
                    y=genre_msg_length,
                    mode='markers',
                    text=genre_names
                )
            ],

            'layout': {
                'title': 'Message length and count by genre',
                'yaxis': {
                    'title': "Message Lengths"
                },
                'xaxis': {
                    'title': "Message Counts"
                }
            }
        },
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
