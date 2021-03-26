# import libraries
import sys
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_csv, categories_csv):

    """
    Function for loading the csv files
    and merging the data together

    Parameters:

    csv file

    Returns:

    Merged dataframe

    """

    # load messages dataset
    messages = pd.read_csv(messages_csv)

    # load categories dataset
    categories = pd.read_csv(categories_csv)

    # merge datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):

    """
    Function for cleaning the data

    Parameters:

    dataframe

    Returns:

    Cleaned dataframe

    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True) \
                                .rename(columns=df['categories'] \
                                .str.split(';',expand=True) \
                                .iloc[0])

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [x[:-2] for x in row]


    # rename the columns of `categories`
    categories.columns = category_colnames


    # Converting category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)


    # Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])


    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Binarize Categories - checking data to ensure only 1's and 0's are used
    cat_columns = list(df.iloc[:,4:].columns)

    for col in cat_columns:
        if (df[col].nunique()) > 2:
            print (col, df[col].nunique())

    mask = (df['related'] == 2)
    df.loc[mask, 'related'] = 1

    ## Remove duplicates.
    df = df.drop_duplicates()

    ## Remove related entries that have values greater than 1.
    df = df.drop(df[df.related == 2].index)

    return df


def save_data(df, db_name):

    """
    Function saves the data to a sqlite database and table

    Parameters:

    dataframe and database name

    """

    # Save the clean dataset into an sqlite database
    db_uri = 'sqlite:///{}'.format(db_name)

    engine = create_engine(db_uri)
    df.to_sql('ResponseTable', engine, index=False, if_exists='replace')


def main():

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
