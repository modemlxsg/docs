import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Bidirectional, LSTM, Dense, Reshape


class CRNN(keras.models.Model):

    def __init__(self, nclass, **kwargs):
        super(CRNN, self).__init__()

        self.conv1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')
        self.conv5 = Conv2D(256, (3, 3), strides=(2, 1), padding='same', activation='relu')
        self.conv6 = Conv2D(256, (3, 3), strides=(2, 1), padding='same', activation='relu')
        self.avg_pool = AveragePooling2D((8, 1))

        self.rnn = Bidirectional(LSTM(256, return_sequences=True))
        self.fc  = Dense(nclass)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.avg_pool(x)

        assert x.shape[1] == 1

        x = tf.squeeze(x, axis=1)
        # x = tf.transpose(x, perm=(1, 0, 2))
        x = self.rnn(x)
        x = self.fc(x)

        return x
        
if __name__ == "__main__":
    model = CRNN(63)
    inputs = tf.random.normal([2, 32, 100, 1], dtype=tf.float32)
    out = model(inputs)
    print(out.shape)

