import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ReLU, Dropout
from MaxReLU import MaxReLU, MaxReLUConv2D
class all_models:
    def __init__(self, n_classes, max_value):
        self.n_classes = n_classes
        self.max_value = max_value
    def create_dense_model(self, input_shape, location="end", init_max_val = 100, activation = "mrelu"):
        flatten_size = input_shape[0] * input_shape[1] * input_shape[2]
        if(activation == "mrelu"):
            model = tf.keras.Sequential([
            Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(flatten_size // 2),
            MaxReLU(flatten_size // 2, init_max_val=self.max_value),  # First hidden layer with learnable max values
            tf.keras.layers.Dense(flatten_size // 4),
            MaxReLU(flatten_size // 4, init_max_val=self.max_value),  # Second hidden layer with learnable max values
            tf.keras.layers.Dense(self.n_classes),  # Output layer for multi-label classification
            tf.keras.layers.Activation('softmax')
            ])
        elif(activation == "relu"):
            model = tf.keras.Sequential([
            Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(flatten_size // 2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(flatten_size // 4),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.n_classes),  # Output layer for multi-label classification
            tf.keras.layers.Activation('softmax')
            ])
        return model

    def create_shallow_cnn_model(self,input_shape, location="end", init_max_val = 100, activation = "mrelu"):
        model = tf.keras.Sequential()
        if(location == "beginning"):
            model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
            if(activation == "mrelu"):
                model.add(MaxReLU(input_shape[2], init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        if(location == "end"):
            model.add(tf.keras.layers.Dense(256))
            if(activation == "mrelu"):
                model.add(MaxReLU(256, init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(self.n_classes))
        model.add(tf.keras.layers.Activation('softmax'))
        return model

    def create_vgg16_model(self, input_shape, location = "end", init_max_val = 100, activation = "mrelu"):
        model = tf.keras.Sequential()
        if(location == "beginning"):
            model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
            if(activation == "mrelu"):
                model.add(MaxReLU(input_shape[2], init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        backbone = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=input_shape,
            pooling='max',
        )
        model.add(backbone)
        model.add(Dropout(0.3))
        if(location == "end"):
            model.add(tf.keras.layers.Dense(256))
            if(activation == "mrelu"):
                model.add(MaxReLU(256, init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(self.n_classes))
        model.add(tf.keras.layers.Activation('softmax'))
        return model

    def create_resnet50_model(self, input_shape, location = "end", init_max_val = 100, activation = "mrelu"):
        model = tf.keras.Sequential()
        if(location == "beginning"):
            model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
            if(activation == "mrelu"):
                model.add(MaxReLU(input_shape[2], init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        backbone = tf.keras.applications.resnet.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=input_shape,
            pooling='max',
        )
        model.add(backbone)
        model.add(Dropout(0.3))
        if(location == "end"):
            model.add(tf.keras.layers.Dense(256))
            if(activation == "mrelu"):
                model.add(MaxReLU(256, init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(self.n_classes))
        model.add(tf.keras.layers.Activation('softmax'))
        return model


    def create_resnet101_model(self, input_shape, location = "end", init_max_val = 100, activation = "mrelu"):
        model = tf.keras.Sequential()
        if(location == "beginning"):
            model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
            if(activation == "mrelu"):
                model.add(MaxReLU(input_shape[2], init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        backbone = tf.keras.applications.resnet.ResNet101(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=input_shape,
            pooling='max',
        )
        model.add(backbone)
        model.add(Dropout(0.3))
        if(location == "end"):
            model.add(tf.keras.layers.Dense(256))
            if(activation == "mrelu"):
                model.add(MaxReLU(256, init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(self.n_classes))
        model.add(tf.keras.layers.Activation('softmax'))
        return model

    def create_mobilenetv2_model(self, input_shape, location = "end", init_max_val = 100, activation = "mrelu"):
        model = tf.keras.Sequential()
        if(location == "beginning"):
            model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
            if(activation == "mrelu"):
                model.add(MaxReLU(input_shape[2], init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=input_shape,
            pooling='max',
        )
        model.add(backbone)
        model.add(Dropout(0.3))
        if(location == "end"):
            model.add(tf.keras.layers.Dense(256))
            if(activation == "mrelu"):
                model.add(MaxReLU(256, init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(self.n_classes))
        model.add(tf.keras.layers.Activation('softmax'))
        return model

    def create_inceptionv3_model(self, input_shape, location = "end", init_max_val = 100, activation = "mrelu"):
        model = tf.keras.Sequential()
        if(location == "beginning"):
            model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
            if(activation == "mrelu"):
                model.add(MaxReLU(input_shape[2], init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Input(shape=input_shape))
        model.add(tf.keras.layers.UpSampling2D(size=(3,3)))
        backbone = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(96,96,3),
            pooling='max',
        )
        model.add(backbone)
        model.add(Dropout(0.3))
        if(location == "end"):
            model.add(tf.keras.layers.Dense(256))
            if(activation == "mrelu"):
                model.add(MaxReLU(256, init_max_val=self.max_value))
            elif(activation == "relu"):
                model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(self.n_classes))
        model.add(tf.keras.layers.Activation('softmax'))
        return model
