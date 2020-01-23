import sys


def load_data(database_filepath):
    '''
    Load the data to be

    Args:
    File path to sqlite database.

    Returns:
    Values for X and y to be loaded into the model.
    '''

    engine = create_engine('sqlite:///disaster_response_tweets.db')
    df = sql_DF = pd.read_sql_table('categorized_tweets', con=engine)
    X = df['message'].values
    y = df.iloc[:,4:].values
    return X,y

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
    pipeline.fit(X_train, y_train)



def save_model(model, model_filepath):
    '''
    Saves the classifer from the model.

    Args:
    The classifier model to be saved.
    The path to save the file to.

    Returns:
    None
    '''
    save_classifier = open("classifier.pkl", 'wb')
    pickle.dump(pipeline, save_classifier)
    save_classifier.close()
    return

def main():
    '''
    This function runs the applicaiton.
    '''
    if len(sys.argv) == 3:
        'sqlite:///disaster_response_tweets.db', "/models/classifier.pkl" = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format('sqlite:///disaster_response_tweets.db'))
        X, Y, category_names = load_data('sqlite:///disaster_response_tweets.db')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(pipeline, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format("/models/classifier.pkl"))
        save_model(pipeline, "/models/classifier.pkl")

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
