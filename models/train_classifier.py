import sys


def load_data(database_filepath):
    '''


    '''

    engine = create_engine('sqlite:///disaster_response_tweets.db')
    df = sql_DF = pd.read_sql_table('categorized_tweets', con=engine)
    X = df['message'].values
    y = df.iloc[:,4:].values
    return X,y

def tokenize(text):
    '''

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

    '''
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier())),
            ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''

    '''
    #train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #train classifier
    pipeline.fit(X_train, y_train)



def save_model(model, model_filepath):
    '''

    '''
    save_classifier = open("classifier.pkl", 'wb')
    pickle.dump(pipeline, save_classifier)
    save_classifier.close()


def main():
    '''
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
