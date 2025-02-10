import pandas as pd
import numpy as np

# create numpy array
# random integer
# import https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user
df = pd.read_csv()
df.head(25)

# unique occupation
df.unique

# function for calculating mean and std of specific column
def stats(data: pd.DataFrame) -> list[float]:
    mean = data.mean()
    std = data.std()
    return [mean, std]

# replace all occurance of 0 with NaN
df.replace()

# sort the column
def sortrow(data):
    pass

# extract rows
# column
