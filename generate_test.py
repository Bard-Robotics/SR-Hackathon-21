#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

#----- CelebA Progressive GAN 128 
generate = hub.KerasLayer("https://tfhub.dev/google/progan-128/1")
single = tf.random.normal([1, 510])
tiled = tf.tile(single, [20, 1])

path = tf.expand_dims(tf.range(-3, 3, 0.3), 1)

added = tf.tile(path, [1, 2])
dev = tf.concat([tiled, added], 1)

images = generate(dev)

for i in range(images.shape[0]):
    tf.keras.preprocessing.image.save_img(f"sample{i}.png", images[i])

