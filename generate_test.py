#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import librosa
import soundfile
import subprocess
import sys
import os

#----- CelebA Progressive GAN 128 
generate = hub.KerasLayer("https://tfhub.dev/google/progan-128/1")
# dim is 512

if __name__ == '__main__':
    (wave, sr) = librosa.load(sys.argv[-1], 11025, mono=True)
    wave = wave[:1024*200]
    
    soundfile.write("output_trunc.wav", wave, sr)

    # Calculate fourier transform 
    spect = librosa.stft(wave, 1024, 128)[:-1,]
    amp = np.abs(spect)/2.5
    gan_in = amp.T

    num_images = gan_in.shape[0]
    batch_size = 64

    os.remove("images/")
    os.mkdir("images")

    def run_batch(start, end):
        ims = generate(gan_in[start:end,:])
        for i in range(end-start):
            tf.keras.preprocessing.image.save_img(f"images/sample{i+start:05d}.png", ims[i])

    batches = list(range(0, num_images, batch_size)) + [num_images]
    batches = zip(batches, batches[1:])
    for (s, e) in batches:
        print(f"Processing image {s}/{num_images}")
        run_batch(s, e)

    length = (1024 * 100) / sr
    fps = num_images / length

    os.remove("output.mp4")
    subprocess.run(["ffmpeg", "-framerate", str(fps), "-i", "images/sample%05d.png", "-i", "output_trunc.wav", "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "output.mp4"])
