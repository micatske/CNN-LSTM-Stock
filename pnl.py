from train import mymodel
from data_process import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from packaging import version
import os

#t+1 data

for i in range(len(x_close_test-1)):
    x_close_forwards[i]=x_close_test[i+1]
    y_close_forwards[i]=y_close_test[i+1]