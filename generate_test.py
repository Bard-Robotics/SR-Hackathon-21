#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import librosa
import sys

#----- CelebA Progressive GAN 128 
generate = hub.KerasLayer("https://tfhub.dev/google/progan-128/1")
# dim is 512

if __name__ == '__main__':
    (wave, sr) = librosa.load(sys.argv[-1], mono=True)

    # Calculate fourier transform 
    spect = librosa.stft(wave, 1024)[:-1,:100]
    amp = np.abs(spect)**2
    gan_in = amp.T

    print(gan_in.shape)
    images = generate(gan_in)

    #for i in range(images.shape[0]):
    #    tf.keras.preprocessing.image.save_img(f"images/sample{i}.png", images[i])

