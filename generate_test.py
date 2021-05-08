#!/usr/bin/python3

import torch
import torchvision
import numpy as np

import requests
import pickle

import librosa
import soundfile

import subprocess
import sys
import os

def curl(url, fname = None):
    if fname is None:
        fname = url.split("/")[-1]

    if not os.path.exists(fname):
        with open(fname, 'wb') as outfile:
            data = requests.get(url)
            outfile.write(data.content)


FFT_SIZE = 1024
NUM_FRAMES = 100
BATCH_SIZE = 16

if __name__ == "__main__":

    # download pretrained netwok
    sys.path.append("./stylegan2-ada-pytorch")

    PRETRAINED_NETWORK = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    curl(PRETRAINED_NETWORK, "ffhq.pkl")

    # open network
    with open('ffhq.pkl', 'rb') as f:
        generator = pickle.load(f)['G_ema'].cuda()

    print(generator.z_dim)

    #z = torch.randn([1, generator.z_dim]).cuda()
    #img = generator(z, None)

    # Load soundfile in librosa
    (wave, sr) = librosa.load(sys.argv[-1], 11025, mono=True)
    wave = wave[:FFT_SIZE*NUM_FRAMES]
    
    soundfile.write("output_trunc.wav", wave, sr)

    # Calculate fourier transform 
    spect = librosa.stft(wave, FFT_SIZE, FFT_SIZE//8)[:-1,:]
    amp = np.abs(spect)/2.5
    gan_in = amp.T

    num_images = gan_in.shape[0]

    try:
        os.mkdir("images")
    except FileExistsError:
        pass

    def run_batch(start, end):
        tens = torch.from_numpy(gan_in[start:end,:]).cuda()
        ims = generator(tens, None)
        for i in range(end-start):
            torchvision.utils.save_image(ims[i], f"images/sample{i+start:05d}.png")

    batches = list(range(0, num_images, BATCH_SIZE)) + [num_images]
    batches = zip(batches, batches[1:])
    for (s, e) in batches:
        print(f"Processing image {s}/{num_images}")
        run_batch(s, e)

    length = (FFT_SIZE*NUM_FRAMES) / sr
    fps = num_images / length

    subprocess.run(["ffmpeg", "-framerate", str(fps), "-i", "images/sample%05d.png", "-i", "output_trunc.wav", "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-vf", "format=yuv420p", "output.mp4"])



