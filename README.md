# SR-Hackathon-21
Ganvis project for HackSR 2021.

To run:  
`python generate_test.py -i <youtube link or mp3 file>`.  
To use a different network, edit line 36 (`NETWORK = 'ffhq'`).

**Environment Setup:**

The command `conda env create -f environment.yml` will create the environment from the preset file.
Be warned that the environment is a little finicky! If the environment file doesn't work for you, here are the requirements:

 - Python 3.7
 - [Pytorch 1.7.1 with CUDA support](https://pytorch.org/get-started/previous-versions/)
 - Torchvision
 - Numpy
 - Librosa
 - FFMPEG
 - Requests
 - youtube_dl

We use pretrained networks from [here](https://github.com/NVlabs/stylegan2-ada-pytorch).
