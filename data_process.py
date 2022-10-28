import numpy as np
import pandas as pd
import os
import sys
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

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'results/')
dir="data/"

df_type=sys.argv[1]
if df_type=="stock":
    stock_name=sys.argv[2]
    if stock_name=="GOOG":
        df=pd.read_csv(dir+"GOOG.csv")
    elif stock_name=="TSLA":
        df=pd.read_csv(dir+"TSLA.csv")
    else:
        sys.exit('Not Found')
if  df_type=="index":
    index_name=sys.argv[2]
    if index_name=="SP500":
           df=pd.read_csv(dir+"SP500.csv") 
           df['Close']=df['Close'].apply(lambda x: x.replace(',','')).astype(float)
 # read the data

#df.head()  # print the summary of the data

# %%
#df.describe()

# %%
#df.isnull().sum().sum()  # check if there is any missing value

# %%
#df.plot(legend=True,subplots=True, figsize=(10,10))  # plot the data
#plt.show()

from sklearn.model_selection import train_test_split

window_size=100
def data_window(df):
    X_close=[]
    Y_close=[]
    global window_size
    for i in range(1,len(df)-window_size):
        first=df.iloc[i,4]
        temp=[]
        temp2=[]
        for j in range(window_size):
            temp.append((df.iloc[i+j,4]-first)/first)
        temp2.append((df.iloc[i+window_size,4]-first)/first)
        X_close.append(np.array(temp).reshape(window_size,1))
        Y_close.append(np.array(temp2).reshape(1,1))
    return X_close,Y_close


X_close,Y_close=data_window(df)
# %%

n=len(X_close)
split=int(n*0.8)
x_close_train, x_close_test, y_close_train, y_close_test =X_close[:split],X_close[split:],Y_close[:split],Y_close[split:]

var=int(split*0.8)
x_close_train, x_close_val, y_close_train, y_close_val = x_close_train[:var],x_close_train[var:],y_close_train[:var],y_close_train[var:]
    
x_close_train, x_close_test, y_close_train, y_close_test=np.array(x_close_train),np.array(x_close_test),np.array(y_close_train),np.array(y_close_test)

x_close_train=x_close_train.reshape(x_close_train.shape[0],1,window_size,1)
x_close_test=x_close_test.reshape(x_close_test.shape[0],1,window_size,1)

x_close_val,y_close_val=np.array(x_close_val),np.array(y_close_val)
x_close_val=x_close_val.reshape(x_close_val.shape[0],1,window_size,1)