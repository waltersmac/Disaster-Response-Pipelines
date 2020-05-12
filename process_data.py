# import libraries
import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('messages.csv')


# load categories dataset
categories = pd.read_csv('categories.csv')


# merge datasets
df = messages.merge(categories, on='id')


# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';',expand=True).rename(columns=df['categories'].str.split(';',expand=True).iloc[0])


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


## Remove duplicates.
df = df.drop_duplicates()

## Remove related entries that have values greater than 1.
df = df.drop(df[df.related == 2].index)


## Save the clean dataset into an sqlite database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df.to_sql('InsertTableName', engine, index=False)