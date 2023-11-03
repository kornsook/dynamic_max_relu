import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ReLU, Dropout
from MaxReLU import MaxReLU

def create_dense_model(input_shape):
    flatten_size = input_shape[0] * input_shape[1] * input_shape[2]
    model = tf.keras.Sequential([
      Flatten(input_shape=input_shape),
      tf.keras.layers.Dense(flatten_size // 2),
      MaxReLU(flatten_size // 2),  # First hidden layer with learnable max values
      tf.keras.layers.Dense(flatten_size // 4),
      MaxReLU(flatten_size // 4),  # Second hidden layer with learnable max values
      tf.keras.layers.Dense(10),  # Output layer for multi-label classification
      tf.keras.layers.Activation('softmax')
    ])
    return model

def create_shallow_cnn_model(input_shape):
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        tf.keras.layers.Dense(256),
        MaxReLU(256),  # Second hidden layer with learnable max values
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax')# Output layer for multi-label classification
    ])
    return model

def create_vgg16_model(input_shape):
    backbone = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling='max',
    )
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.Dense(256),
        MaxReLU(256),  # Second hidden layer with learnable max values
        tf.keras.layers.Dense(10),  
        tf.keras.layers.Activation('softmax')# Output layer for multi-label classification
    ])
    return model

def create_resnet50_model(input_shape):
    backbone = tf.keras.applications.resnet.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling='max',
    )
    model = tf.keras.Sequential([
        backbone,
        Dropout(0.3),
        tf.keras.layers.Dense(256),
        MaxReLU(256),  # Second hidden layer with learnable max values
        tf.keras.layers.Dense(10),  
        tf.keras.layers.Activation('softmax')# Output layer for multi-label classification
    ])
    return model

def create_resnet101_model(input_shape):
    backbone = tf.keras.applications.resnet.ResNet101(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling='max',
    )
    model = tf.keras.Sequential([
        backbone,
        Dropout(0.3),
        tf.keras.layers.Dense(256),
        MaxReLU(256),  # Second hidden layer with learnable max values
        tf.keras.layers.Dense(10),  
        tf.keras.layers.Activation('softmax')# Output layer for multi-label classification
    ])
    return model

def create_mobilenetv2_model(input_shape):
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling='max',
    )
    model = tf.keras.Sequential([
        backbone,
        Dropout(0.3),
        tf.keras.layers.Dense(256),
        MaxReLU(256),  # Second hidden layer with learnable max values
        tf.keras.layers.Dense(10),  
        tf.keras.layers.Activation('softmax')# Output layer for multi-label classification
    ])
    return model

def create_inceptionv3_model(input_shape):
    backbone = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(96,96,3),
        pooling='max',
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.UpSampling2D(size=(3,3)),
        backbone,
        Dropout(0.3),
        tf.keras.layers.Dense(256),
        MaxReLU(256),  # Second hidden layer with learnable max values
        tf.keras.layers.Dense(10),
        tf.keras.layers.Activation('softmax')  # Output layer for multi-label classification
    ])
    return model
