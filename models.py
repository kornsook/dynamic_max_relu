import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ReLU, Dropout
from MaxReLU import MaxReLU, MaxReLUConv2D

def create_dense_model(input_shape, location="end", init_max_val = 100):
    flatten_size = input_shape[0] * input_shape[1] * input_shape[2]
    model = tf.keras.Sequential([
      Flatten(input_shape=input_shape),
      tf.keras.layers.Dense(flatten_size // 2),
      MaxReLU(flatten_size // 2, init_max_val=init_max_val),  # First hidden layer with learnable max values
      tf.keras.layers.Dense(flatten_size // 4),
      MaxReLU(flatten_size // 4, init_max_val=init_max_val),  # Second hidden layer with learnable max values
      tf.keras.layers.Dense(10),  # Output layer for multi-label classification
      tf.keras.layers.Activation('softmax')
    ])
    return model

def create_shallow_cnn_model(input_shape, location="end", init_max_val = 100):
    model = tf.keras.Sequential()
    if(location == "beginning"):
        model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
        model.add(MaxReLU(input_shape[2], init_max_val=init_max_val))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    if(location == "end"):
        model.add(tf.keras.layers.Dense(256))
        model.add(MaxReLU(256, init_max_val=init_max_val))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

def create_vgg16_model(input_shape, location = "end", init_max_val = 100):
    model = tf.keras.Sequential()
    if(location == "beginning"):
        model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
        model.add(MaxReLU(input_shape[2], init_max_val=init_max_val))
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
        model.add(MaxReLU(256, init_max_val=init_max_val))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

def create_resnet50_model(input_shape, location = "end", init_max_val = 100):
    model = tf.keras.Sequential()
    if(location == "beginning"):
        model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
        model.add(MaxReLU(input_shape[2], init_max_val=init_max_val))
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
        model.add(MaxReLU(256, init_max_val=init_max_val))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model


def create_resnet101_model(input_shape, location = "end", init_max_val = 100):
    model = tf.keras.Sequential()
    if(location == "beginning"):
        model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
        model.add(MaxReLU(input_shape[2], init_max_val=init_max_val))
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
        model.add(MaxReLU(256, init_max_val=init_max_val))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

def create_mobilenetv2_model(input_shape, location = "end", init_max_val = 100):
    model = tf.keras.Sequential()
    if(location == "beginning"):
        model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
        model.add(MaxReLU(input_shape[2], init_max_val=init_max_val))
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
        model.add(MaxReLU(256, init_max_val=init_max_val))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model

def create_inceptionv3_model(input_shape, location = "end", init_max_val = 100):
    model = tf.keras.Sequential()
    if(location == "beginning"):
        model.add(Conv2D(filters=input_shape[2], kernel_size=(3, 3), input_shape = input_shape, padding="same"))
        model.add(MaxReLU(input_shape[2], init_max_val=init_max_val))
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
        model.add(MaxReLU(256, init_max_val=init_max_val))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    return model
