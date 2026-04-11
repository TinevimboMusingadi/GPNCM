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

def grow_lstm_model(old_model, additional_units, dropout_rate=0.2, l2_lambda=1e-4):
    """
    Expands an existing LSTM model by adding a new Dense layer before the output.
    This allows the model to increase capacity as it sees more diverse station data.
    """
    # Get current architecture
    old_layers = old_model.layers
    
    # We reconstruct the model to add a layer before the final Dense(1)
    # The output of the second to last layer (before final Dense)
    x = old_layers[-2].output 
    
    # Add new capacity
    x = Dense(additional_units, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout_rate)(x)
    
    # New output layer
    new_output = Dense(1, name=f'output_{len(old_layers)}')(x)
    
    new_model = Model(inputs=old_model.input, outputs=new_output)
    
    # Compile with default settings
    new_model.compile(optimizer='adam', loss='mse', metrics=[masked_mape])
    
    return new_model

def load_evolved_model(model_path):
    """
    Robustly loads an evolved model, bypassing Keras 3 deserialization issues.
    """
    try:
        # We use compile=False to avoid the common 'mse' deserialization bug in Keras 3
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=[masked_mape])
        return model
    except Exception as e:
        print(f"Warning: Could not load model from {model_path} due to: {e}")
        return None
