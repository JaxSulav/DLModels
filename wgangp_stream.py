import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import gdown
from PIL import Image

class MyGenerator(nn.Module):
    def __init__(self, gen_initial_dim, in_channels, out_channels, k, s, p, b):
        super(MyGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(gen_initial_dim, out_channels*16, k, s, padding=0, bias=b),
                nn.BatchNorm2d(out_channels*16),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(out_channels*16, out_channels*8, k, s, p, bias=b),
                nn.BatchNorm2d(out_channels*8),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(out_channels*8, out_channels*4, k, s, p, bias=b),
                nn.BatchNorm2d(out_channels*4),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(out_channels*4, out_channels*2, k, s, p, bias=b),
                nn.BatchNorm2d(out_channels*2),
                nn.ReLU()
            ),
            nn.ConvTranspose2d(out_channels*2, in_channels, k, s, p),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)

params = {
    "gen_initial_dimension": 100, 
    "input_channels": 3, 
    "out_channels_gen": 64,
    "kernel_gen": 4,
    "stride_gen": 2,
    "pad_gen": 1,
    "bias_gen": False, 
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

my_generator = MyGenerator(
    params["gen_initial_dimension"], 
    params["input_channels"],
    params["out_channels_gen"],
    params["kernel_gen"],
    params["stride_gen"],
    params["pad_gen"],
    params["bias_gen"]
).to(device)

os.makedirs("models", exist_ok=True)
if not os.path.isfile('models/wgangp1.2.pt'):
    url = 'https://drive.google.com/uc?export=download&id=1gR5G_KFWP1Hl8iaHl4w2gLcqmA91raiZ'
    output = 'models/wgangp1.2.pt'
    gdown.download(url, output, quiet=False)
else:
    print("Model already exists")

checkpoint_path = 'models/wgangp1.2.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
my_generator.load_state_dict(checkpoint['generator_state_dict'])

my_generator.eval()

st.title("WGAN-GP Face Generation")

if st.button("Generate Face"):
    noise_dim = 100  
    num_images = 1
    noise = torch.randn(num_images, noise_dim, 1, 1, device=device)
    save_dir = 'gen_faces'
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        fake_images = my_generator(noise)
    fake_images = (fake_images + 1) / 2 
    image_file_path = os.path.join(save_dir, 'fake_image.png')
    save_image(fake_images, image_file_path)
    
    image = Image.open(image_file_path)
    st.image(image, caption="Generated Face", use_column_width=True)
