import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras.regularizers import l2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import masked_mape

def build_lstm(n_past, n_features, units=(64, 128), dropout_rate=0.2, l2_lambda=1e-4):
    """
    Builds the improved baseline LSTM model for precipitation nowcasting.
    """
    inputs = Input(shape=(n_past, n_features))
    
    x = LSTM(units[0], return_sequences=True, kernel_regularizer=l2(l2_lambda))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = LSTM(units[1], return_sequences=False, kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=[masked_mape])
    
    return model
