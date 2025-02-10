import pandas as pd
import numpy as np

# evenly spaced array 0 to 10.

es_array = np.linespace(0,10, 20)

# loading csv dataset

df = pd.read_csv('house-price.csv')
df.columns

# random dataframe

data = {
        "a": [np.randrange(10)],
        "b": [np.randrange(10)],
        "c": [np.randrange(10)]
}

df_rand = pd.DataFrame(data)

# calculate mean of df_rand
df_rand.mean()

# filter row 

# Concatenate two dataset horizontally
df_v1 = pd.DataFrame()
df_v2 = pd.DataFrame()
df_concat = pd.concat(df_v1, df_v2)

# new column
sum_apart = "todo later"

# drop nan values
df.dropna()

# join operation

# save
df.to_csv('clean_house_price.csv')
