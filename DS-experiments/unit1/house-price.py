## are markdown comments
# are simple comments
### two line space means that you have to change block of code

## Introduction to Data Science
## Classic house price prediction dataset analysis


# import libraries of data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import libraries for model creation
import sklearn


## Loading dataset 
## dataset from https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data 


# path to the dataset
df = pd.read_csv('house_price_prediction.csv')

# view first 10 rows of the dataset
df.head()


## The last column 'Sales Price' is the target variable
## Rest of the columns are either features or bloated data.


# seperating target variale
target = df['SalesPrice']
# dropping target from Data Frame
df.drop()


## Checking missing values
## Only Columns with missing will be displayed 


# Total nan values
df.isnull().sum().sum()
# get all the columns with nan values
nan_cols = [i for i in df.columns if df[i].isnull().any()]

## OBJECTIVE:
## house price prediction analysis is conducted to find the suitable price for a given house 


## Sperate dataset into numerical and categorical dataset.


num_df = df.select_dtypes(include=[np.numerical])
cat_df = df.select_dtypes(exclude=[np.numerical])

## Statistical analysis of numerical columns


# Basic statistics of data frame 
num_df.describe()


## Top 5 Most frequent categories in categorical column.

## Remove duplicates 
cat_df = cat_df.remove_duplicates()


## Training and testing split

# merging cleanded data
clean_data = df.cat(num_df, cat_df)

from sklearn.metrics import train_test_split

X_train, y_train, X_test, y_test = train_test_split()

## Histogram plot of Column

## Correlation between numerical dataset
