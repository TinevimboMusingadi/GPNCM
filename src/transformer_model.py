import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Layer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import masked_mape

class PositionalEncoding(Layer):
    """Adds sinusoidal positional encoding to the input sequence."""
    def __init__(self, max_len=500, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def call(self, x):
        seq_len  = tf.shape(x)[1]
        d_model  = tf.shape(x)[2]
        position = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        div_term = tf.exp(
            tf.cast(tf.range(0, d_model, 2), tf.float32) *
            -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32))
        )
        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)
        pe = tf.reshape(tf.stack([pe_sin, pe_cos], axis=-1),
                        [1, seq_len, -1])[:, :, :d_model]
        return x + tf.cast(pe, x.dtype)

class TransformerEncoderBlock(Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn    = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ff1     = Dense(ff_dim, activation='gelu')
        self.ff2     = Dense(d_model)
        self.norm1   = LayerNormalization(epsilon=1e-6)
        self.norm2   = LayerNormalization(epsilon=1e-6)
        self.drop1   = Dropout(dropout_rate)
        self.drop2   = Dropout(dropout_rate)

    def call(self, x, training=False):
        # Multi-head self-attention + residual
        attn_out = self.attn(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        # Feed-forward + residual
        ff_out = self.ff2(self.ff1(x))
        ff_out = self.drop2(ff_out, training=training)
        x = self.norm2(x + ff_out)
        return x

def build_transformer(n_past, n_features, d_model=64, num_heads=4, ff_dim=128, num_blocks=2, dropout_rate=0.1):
    inputs = Input(shape=(n_past, n_features))

    # Project input features to d_model dimensions
    x = Dense(d_model)(inputs)
    x = PositionalEncoding()(x)
    x = Dropout(dropout_rate)(x)

    # Stack Transformer encoder blocks
    for _ in range(num_blocks):
        x = TransformerEncoderBlock(d_model, num_heads, ff_dim, dropout_rate)(x)

    # Pool over time dimension and predict
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='gelu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=[masked_mape])
    return model
