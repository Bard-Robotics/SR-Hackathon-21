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
import argparse
import shutil

from getmp3 import download_audio


def curl(url, fname=None):
    if fname is None:
        fname = url.split("/")[-1]

    if not os.path.exists(fname):
        print(f"Downloading {fname}")
        with open(fname, 'wb') as outfile:
            data = requests.get(url)
            outfile.write(data.content)
    else:
        print(f"Using cached {fname}")


def handle_mp3(infile):
    (wave, sr) = librosa.load(infile, 11025, mono=True)
    print("Loaded image and downsampled to 11025 hz")
    num_samples = len(wave)
    num_samples_round = FFT_SIZE * (num_samples // FFT_SIZE)
    wave = wave[:num_samples_round]
    soundfile.write("output_trunc.wav", wave, sr)

    # Calculate fourier transform
    spect = librosa.stft(wave, FFT_SIZE, FFT_SIZE // 4)
    print(f"Spectrogram has shape {spect.shape}")
    spect = spect[:512, :] / np.sqrt(FFT_SIZE)
    amp = np.abs(spect) ** 2
    amp = amp / np.max(amp)
    gan_in = amp.T
    return gan_in, sr, num_samples_round


def run_batch(gen ,gan_in, start, end):
    tens = torch.from_numpy(gan_in[start:end, :]).cuda()
    ims = gen(tens, None)
    for i in range(end - start):
        torchvision.utils.save_image(ims[i], f"images/sample{i + start:05d}.png")


def get_images(gen, gan_input):
    num_images = gan_input.shape[0]

    try:
        shutil.rmtree("images")
        os.mkdir("images")
    except FileExistsError:
        pass

    batches = list(range(0, num_images, BATCH_SIZE)) + [num_images]
    batches = zip(batches, batches[1:])
    for (s, e) in batches:
        print(f"Processing image {s}/{num_images}")
        run_batch(gen, gan_input, s, e)


FFT_SIZE = 1024
BATCH_SIZE = 16
# Can use ffhq (humans), metfaces (paintings), afhq[dog/cat/wild], brecahad (breast cancer)
NETWORK = "ffhq.pkl"

if __name__ == "__main__":

    # Parse arg
    parser = argparse.ArgumentParser(description="audio-based Gan Image generator (from audio)")
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        type=str,
        help="The relevant input path to an mp3 file or a youtube link"
    )

    args = parser.parse_args()

    # download pretrained network
    sys.path.append("./stylegan2-ada-pytorch")

    PRETRAINED_NETWORK = f"https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/{NETWORK}"
    curl(PRETRAINED_NETWORK, NETWORK)

    # open network
    with open(NETWORK, 'rb') as f:
        generator = pickle.load(f)['G_ema'].cuda()

    # If link provided - download
    filename = args.input
    if "http" in args.input:
        downloadAudio(args.input)
        max_mtime = 0
        for file in os.listdir(os.getcwd()):
            if file.endswith(".mp3"):  # get the most recently added mp3
                mtime = os.stat(file).st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
                    chosen = file
        filename = args.input
        gan_in, sr, num_samples_round = handle_mp3(chosen)
    else:
        gan_in, sr, num_samples_round = handle_mp3(args.input)

    get_images(generator, gan_in)

    num_images = gan_in.shape[0]
    length = num_samples_round / sr
    fps = num_images / length

    print(f"Have {length} seconds of audio.")
    print(f"Have {num_images} frames")
    print(f"Framerate: {fps}")
    outname = ".".join(filename.split(".")[:-1]) + ".mp4"
    ffmpeg_command = ["ffmpeg",
                    "-framerate", str(fps),
                    "-i", "images/sample%05d.png",
                    "-i", "output_trunc.wav",
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:a", "mp3",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-vcodec", "libx264",
                    "-ar", "44100",
                    outname]

    print(f"Calling {' '.join(ffmpeg_command)}")
    subprocess.run(ffmpeg_command)

    os.remove("output_trunc.wav")
