import tensorflow as tf
import tensorflow.keras.backend as K

def masked_mape(y_true, y_pred, epsilon=0.1):
    """
    MAPE ignoring near-zero rainfall values to avoid division by zero.
    """
    mask = tf.abs(y_true) > epsilon
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.reduce_mean(tf.abs((y_true_masked - y_pred_masked) / (y_true_masked + epsilon))) * 100.0

# Robust serialization registration for both Keras 2 and Keras 3
try:
    if hasattr(tf.keras.utils, 'register_keras_serializable'):
        tf.keras.utils.register_keras_serializable(package="CustomMetrics")(masked_mape)
    elif hasattr(tf.keras, 'saving') and hasattr(tf.keras.saving, 'register_keras_serializable'):
        tf.keras.saving.register_keras_serializable(package="CustomMetrics")(masked_mape)
except Exception:
    pass
