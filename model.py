import tensorflow as tf
print(tf. __version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed,Multiply,Permute
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.metrics import RootMeanSquaredError
import pydot
from tensorflow.keras.models import Model
from data_process import window_size
# %%
from tensorflow.keras.utils import plot_model
import pydot

# %%
def attention_3d(inputs,single_attention_vector = False):
    input_dim = int(inputs.shape[2])
    a = inputs
    # a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

# %%


# %%
def build_model():
    input_layer= tf.keras.Input(shape=(None,window_size, 1))
    
    #cov layers
    cov_1=TimeDistributed(Conv1D(64, kernel_size=3, activation='relu'))(input_layer)
    pool_1=TimeDistributed(MaxPooling1D(2))(cov_1)
    cov_2=TimeDistributed(Conv1D(128, kernel_size=3, activation='relu'))(pool_1)
    pool_2=TimeDistributed(MaxPooling1D(2))(cov_2)
    cov_3=TimeDistributed(Conv1D(64, kernel_size=3, activation='relu'))(pool_2)
    pool_3=TimeDistributed(MaxPooling1D(2))(cov_3)
    flat=TimeDistributed(Flatten())(pool_3)
    
     #attention
    attention_mul = attention_3d(flat)
    #attention_mul = Flatten()(attention_mul)
    
    
    #lstm layers    
    lstm_1=Bidirectional(LSTM(100, return_sequences=True))(attention_mul)
    drop_1=Dropout(0.5)(lstm_1)
    lstm_2=Bidirectional(LSTM(100, return_sequences=False))(drop_1)
    drop_2=Dropout(0.5)(lstm_2)
    
    
   
    output_layer=Dense(1, activation='linear')(drop_2)
    
    func_model = Model(inputs=input_layer, outputs=output_layer)
    
    return func_model
    

