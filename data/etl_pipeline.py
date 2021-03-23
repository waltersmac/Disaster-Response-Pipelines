# import libraries
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_csv, categories_csv):

    '''

    '''

    # load messages dataset
    messages = pd.read_csv(messages_csv)

    # load categories dataset
    categories = pd.read_csv(categories_csv)

    return messages, categories


def merge_df(messages, categories):

    '''

    '''

    # merge datasets
    df = messages.merge(categories, on='id')

    return df


def cat_columns(df):

    '''

    '''

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

    return df


def dup_data(df):

    '''


    '''

    # Binarize Categories - checking data to ensure only 1's and 0's are used
    for col in df.iloc[:,4:].columns.to_list():
        if (df[col].nunique()) > 2:
            print (col, df[col].nunique())

    mask = (df['related'] == 2)
    df.loc[mask, 'related'] = 1

    ## Remove duplicates.
    df = df.drop_duplicates()

    ## Remove related entries that have values greater than 1.
    df = df.drop(df[df.related == 2].index)

    return df


def process_data(messages_csv, categories_csv):

    ## Extract
    messages, categories = load_data(messages_csv, categories_csv)

    ## Transform
    df = merge_df(messages, categories)
    df = cat_columns(df_merge)
    df = dup_data(df)

    ## Load
    # Save the clean dataset into an sqlite database
    db_path = os.path.join(os.path.abspath('../data/processed/'), 'ResponseDatabase.db')
    db_uri = 'sqlite:///{}'.format(db_path)

    engine = create_engine(db_uri)
    df.to_sql('ResponseTable', engine, index=False, if_exists='replace')
