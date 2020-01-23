import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''


    '''
    # load messages dataset
    messages = pd.read_csv('messages.csv')
    # load categories dataset
    categories = pd.read_csv('categories.csv')
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id')
    return df


def clean_data(df):
    '''
    '''
    #a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    #extract a list of new column names for categories.
    row = categories.loc[:0]
    category_colnames = list(row.apply(lambda x: str(x[0])[:-2]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # merge the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    '''
    engine = create_engine('sqlite:///disaster_response.db')
    df.to_sql('categorized_tweets', engine, index=False)
    categories.to_sql('tweet_categories', engine, index=False)
    return


def main():
    '''
    '''

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
