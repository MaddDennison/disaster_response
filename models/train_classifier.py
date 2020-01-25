import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    Load the data to be

    Args:
    File path to sqlite database.

    Returns:
    Values for X and y to be loaded into the model.
    '''
    db_str ='sqlite:///'+ database_filepath
    engine = create_engine(db_str)
    df = sql_DF = pd.read_sql_table('categorized_tweets', con=engine)
    X = df['message'].values
    y = df.iloc[:,4:].values
    category_names = list(df.iloc[:0, 4:])
    return X,y,category_names

def tokenize(text):
    '''
    Takes a document of text and tokenizes all the words.

    Args:
    A string of the content to be tokenized.

    Returns:
    A list of the clean tokens.
    '''
    #tokenize text
    tokens = word_tokenize(text)

    #initiate Lemmatizer
    lemmatizer =WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)


    return clean_tokens


def build_model():
    '''
    This function sets up the ML pipeline and initializes the transformers
    and classifier.

    Args:
    None

    Returns:
    The classifier to be used in evaluation.
    '''
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5, max_features=10000)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier())),
                ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Fits the data to the model.

    Args:
    The classifier to be fitted.
    The X_test data.
    The y_test data.
    A list of the catagory names.

    Returns:
    A list of the clean tokens.
    '''

    #train classifier
    model.fit(X_test, Y_test)
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[i], Y_pred[i]))


def save_model(model, model_filepath):
    '''
    Saves the classifer from the model.

    Args:
    The classifier model to be saved.
    The path to save the file to.

    Returns:
    None
    '''
    save_classifier = open(model_filepath, 'wb')
    pickle.dump(model, save_classifier)
    save_classifier.close()
    return

def main():
    '''
    This function runs the applicaiton.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
