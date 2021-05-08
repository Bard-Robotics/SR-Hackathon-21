import requests
import torch
import pickle
import os
import sys

def curl(url, fname = None):
    if fname is None:
        fname = url.split("/")[-1]

    if not os.path.exists(fname):
        with open(fname, 'wb') as outfile:
            data = requests.get(url)
            outfile.write(data.content)


if __name__ == "__main__":

    sys.path.append("./stylegan2-ada-pytorch")

    PRETRAINED_NETWORK = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    curl(PRETRAINED_NETWORK, "ffhq.pkl")

    device = torch.device('cpu')
    with open('ffhq.pkl', 'rb') as f:
        generator = pickle.load(f)['G_ema'].cuda()

    z = torch.randn([1, generator.z_dim]).cuda()
    img = generator(z, None)


