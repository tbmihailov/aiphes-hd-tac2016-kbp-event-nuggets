import tensorflow as tf

class tf_helpers(object):

    @staticmethod
    def tf_nan_to_zeros_float64(tensor):
        """
            Mask NaN values with zeros
        :param tensor: Tensor that might have Nan values
        :return: Tensor with replaced Nan values with zeros
        """
        return tf.select(tf.is_nan(tensor), tf.zeros(tf.shape(tensor), dtype=tf.float64), tensor)

    @staticmethod
    def tf_nan_to_zeros_float32(tensor):
        """
            Mask NaN values with zeros
        :param tensor: Tensor that might have Nan values
        :return: Tensor with replaced Nan values with zeros
        """
        return tf.select(tf.is_nan(tensor), tf.zeros(tf.shape(tensor), dtype=tf.float32), tensor)