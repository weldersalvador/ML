import tensorflow as tf


class DepthPool(tf.keras.layers.Layer):
    def __init__(self, pool_size = 2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
    def call(self,inputs):
        shape = tf.shape(inputs)
        groups = shape[-1] // self.pool_size #number of channels groups
        new_shape = tf.concat([shape[:-1], [groups,self.pool_size]],axis = 0)
        return tf.reduce_max(tf.reshape(inputs,new_shape),axis = -1)