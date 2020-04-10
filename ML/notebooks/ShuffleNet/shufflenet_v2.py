import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def shufflenet_v2(nclass):
    inputs = layers.Input(shape=(224, 224, 3))
    
    # stage 1
    x = layers.Conv2D(24, 3, 2, 'same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(3, 2, 'same')(x)

    # stage 2
    
    
    

    return keras.Model(inputs=inputs, outputs=x)


if __name__ == "__main__":
    model = shufflenet_v2(20)
    model.summary()