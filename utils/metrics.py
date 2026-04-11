import tensorflow as tf
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable(package="CustomMetrics")
def masked_mape(y_true, y_pred, epsilon=0.1):
    """
    MAPE ignoring near-zero rainfall values to avoid division by zero.
    """
    mask = tf.abs(y_true) > epsilon
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    return tf.reduce_mean(tf.abs((y_true_masked - y_pred_masked) / (y_true_masked + epsilon))) * 100.0
