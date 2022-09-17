import numpy as np
import pandas as pd
import os

dir="data/"

df=pd.read_csv(dir+"GOOG.csv")  # read the data
df.head()  # print the summary of the data