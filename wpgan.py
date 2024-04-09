import torch
import torch.nn as nn
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.utils import save_image



class MyGenerator(nn.Module):
    def __init__(self, gen_initial_dim, in_channels, out_channels, k, s, p, b):
        super(MyGenerator, self).__init__()

        self.generator = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(gen_initial_dim, out_channels*16, k, s, padding=0, bias=b),
                nn.BatchNorm2d(out_channels*16), #Batch norm in generator layer norm in critic
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
            nn.Tanh(), # for discriminator
        )


    def forward(self, x):
        return self.generator(x)



params = { 
    "batch_size": 64, 
    "gen_initial_dimension": 100, 
    "input_channels": 3, 
    
    "out_channels_gen": 64,
    "kernel_gen": 4,
    "stride_gen": 2,
    "pad_gen": 1,
    "bias_gen": False, 
    "learning_rate_gen": 0.0004,
    
    "out_channels_dis": 64,
    "kernel_dis": 4,
    "stride_dis": 2,
    "pad_dis": 1,
    "bias_dis": False,
    "leaky_relu_slope": 0.2,
    "learning_rate_dis": 0.0004,
    "dis_extention_factor": 5,
    "lambda": 10,

    "num_epochs": 226,    
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


checkpoint_path = 'models/checkpoint_epoch_215.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
my_generator.load_state_dict(checkpoint['generator_state_dict'])

my_generator.eval()

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
# plt.imshow(fake_images.permute(1, 2, 0).numpy())
# # fake_images_grid = make_grid(fake_images, nrow=5, normalize=True).cpu()
# # plt.figure(figsize=(10, 5))
# # plt.imshow(fake_images_grid.permute(1, 2, 0).numpy())
# plt.axis('off')
plt.show()