import tensorflow as tf

class MaxReLU(tf.keras.layers.Layer):
    def __init__(self, units, init_max_val = 100, **kwargs):
        super(MaxReLU, self).__init__(**kwargs)
        self.units = units
        print(init_max_val)
        self.max_values = self.add_weight(name='max_values', shape=(self.units,), initializer=tf.keras.initializers.Constant(init_max_val), trainable=True)
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "max_values": self.max_values.numpy()
        })
        return config
    def build(self, input_shape):
#         self.max_values = self.add_weight(name='max_values', shape=(self.units,), initializer=tf.keras.initializers.Constant(100), trainable=True)
        super(MaxReLU, self).build(input_shape)

    def call(self, inputs):
        return tf.minimum(tf.maximum(inputs, 0 ), tf.maximum(self.max_values, 0))

    @classmethod
    def from_config(cls, config):
        max_values = config.pop("max_values")  # Deserialize max_values from the config
        instance = cls(**config)
        instance.max_values = tf.Variable(max_values)  # Restore max_values as a Variable
        return instance

class MaxReLUConv2D(tf.keras.layers.Layer):
    def __init__(self, units, kernel_size, trainable_max_values=True, **kwargs):
        super(MaxReLUConv2D, self).__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.trainable_max_values = trainable_max_values

    def build(self, input_shape):
        self.max_values = self.add_weight(
            name='max_values',
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(100.0),
            trainable=self.trainable_max_values
        )
        super(MaxReLUConv2D, self).build(input_shape)

    def call(self, inputs):
        # Apply the Conv2D operation
        conv_output = tf.keras.layers.Conv2D(self.units, self.kernel_size, padding='same')(inputs)

        # Apply the ReLU activation with trainable max values
        return tf.minimum(tf.maximum(conv_output, 0), self.max_values)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "kernel_size": self.kernel_size,
            "trainable_max_values": self.trainable_max_values,
        })
        return config
