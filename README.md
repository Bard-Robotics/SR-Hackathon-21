# SR-Hackathon-21
Ganvis project for HackSR 2021.

### Design Concept 
We convert audio frames into images by using slices of the spectrogram as latent vectors. By overlapping the FFT slices, the change in the spectrogram from is relatively small on each iterator. Feeding them into a pre-trained StyleGAN, we are able to obtain a music video with (hopefully) smooth transitions!

To run: 
`python generate_test.py -i <youtube link or mp3 file>`.   
To use a different network, edit line 78 (`NETWORK = 'ffhq'`).   

**TODOs:**
 - Add more CLI parameters to the script, such as FFT size and framerate.

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

