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

import model as *
import data_process as *

import tensorflow as tf
from tensorflow.keras.utils import plot_model
import pydot

mymodel = model.build_model()

# %%
plot_model(mymodel, show_shapes=True, show_layer_names=True,to_file='model.png')

# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# %%
mymodel.compile(optimizer=optimizer, loss='mse', metrics=['mse', 'mae'])

# %%
history = mymodel.fit(x_close_train, y_close_train, validation_data=(x_close_test,y_close_test), epochs=50,batch_size=40, verbose=1, shuffle =True)

# %%
predicted=mymodel.predict(x_close_test)

# %%
predicted

# %%
loss,rmse,mae=mymodel.evaluate(x_close_test,y_close_test)

# %%
def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)


plot_metrics(metric_name='loss',title="Loss",ylim=0.02)

# %%
predicted=mymodel.predict(x_close_test)
test_label=y_close_test.reshape(-1,1)
predicted=np.array(predicted[:,0]).reshape(-1,1)
len_train=len(x_close_train)

for j in range(len_train,len_train+len(x_close_test)):
    temp=df.iloc[j,3]
    predicted[j-len_train]=(predicted[j-len_train]+1)*temp
    test_label[j-len_train]=(test_label[j-len_train]+1)*temp

# %%
predicted

# %%
plt.plot(predicted, color = 'green', label = 'Predicted  Stock Price')
plt.plot(test_label, color = 'red', label = 'Real Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()