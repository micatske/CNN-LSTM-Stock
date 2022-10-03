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

import model
from data_process import *

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

#earlystopping 
es  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
# %%
history = mymodel.fit(x_close_train, y_close_train, validation_data=(x_close_val,y_close_val), epochs=50,batch_size=40, verbose=1, shuffle =True,callbacks=[es])


# %%
def plot_metrics(metric_name, title, ylim=2):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.title(sys.argv[2]+' Train Metrics')
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)
    plt.legend()
    plt.savefig(results_dir+sys.argv[2]+'train.png')

plt.figure(1)
plot_metrics(metric_name='loss',title="Loss",ylim=0.05)

# %%


