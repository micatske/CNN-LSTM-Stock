import numpy as np
import pandas as pd
import os
from datetime import datetime
import math
import seaborn as sns
import datetime as dt
from datetime import datetime    
sns.set_style("whitegrid")
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt
#matplotlib inline
plt.style.use("ggplot")

dir="data/"
df=pd.read_csv(dir+"GOOG.csv")  # read the data
df.head()  # print the summary of the data

# %%
df.describe()

# %%
df.isnull().sum().sum()  # check if there is any missing value

# %%
df.plot(legend=True,subplots=True, figsize=(10,10))  # plot the data
plt.show()

from sklearn.model_selection import train_test_split

X_close=[]
Y_close=[]
window_size=100

for i in range(1,len(df)-window_size):
    first=df.iloc[i,3]
    temp=[]
    temp2=[]
    for j in range(window_size):
        temp.append((df.iloc[i+j,3]-first)/first)
    temp2.append((df.iloc[i+window_size,3]-first)/first)
    X_close.append(np.array(temp).reshape(window_size,1))
    Y_close.append(np.array(temp2).reshape(1,1))

# %%
import tensorflow as tf

# %%
x_close_train, x_close_test, y_close_train, y_close_test = train_test_split(X_close, Y_close, test_size=0.2, random_state=42,shuffle=True)
x_close_train, x_close_test, y_close_train, y_close_test=np.array(x_close_train),np.array(x_close_test),np.array(y_close_train),np.array(y_close_test)
x_close_train=x_close_train.reshape(x_close_train.shape[0],1,window_size,1)
x_close_test=x_close_test.reshape(x_close_test.shape[0],1,window_size,1)

# %%
x_close_train.shape

# %%
len(x_close_train)

# %%
##Train the model

# %%