import tensorflow as tf
from tensorflow import keras
import yaml
import numpy as np
from dataset import Mj_Dataset
from model import CRNN
from losses import CTCLoss
from utils import decode

#config
config_file = open('config.yaml', 'r', encoding='utf-8')
config = config_file.read()
config_file.close()
config = yaml.full_load(config)

# data
dataset = Mj_Dataset('train')
train_ds = dataset.getDS().batch(16)

val_ds = Mj_Dataset('val').getDS().batch(16)

nclass = config['crnn']['nClass']
model = CRNN(nclass)



optimizer = keras.optimizers.Adam(learning_rate=0.003)
criterion = CTCLoss(logits_time_major=False)


epochs = 10
for epoch in range(epochs):
    print(f'Start of epoch {epoch}')

    # train
    for step, (imgs, labels) in enumerate(train_ds):
        y_true = dataset.encode(labels)  # sparse_tensor

        with tf.GradientTape() as tape:
            y_pred = model(imgs)
            loss = criterion(y_true, y_pred)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    print(f"loss : {loss}")

    # val
    for step, (imgs, labels) in enumerate(val_ds):
        out = model(imgs)
        decoded = decode(out)
        print(decoded)




    
