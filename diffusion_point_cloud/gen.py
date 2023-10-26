import os

from .models.vae_flow import *

ckpt = './diffusion_point_cloud/trained/GEN_airplane.pt'
device = os.getenv("DEVICE", "cpu")
print(f'Device: [{device}]')
sample_num_points = 10000
batch_size = 1

print('Loading model')
ckpt = torch.load(ckpt, map_location=device)

model = FlowVAE(ckpt['args']).to(device)

print('Loading state dict')
model.load_state_dict(ckpt['state_dict'])


def gen_diffusion_point_cloud():
    print('Generating 3D model with diffusion_point_cloud')
    with torch.no_grad():
        z = torch.randn([batch_size, ckpt['args'].latent_dim]).to(device)
        x = model.sample(z, sample_num_points, flexibility=ckpt['args'].flexibility, stream_diffusion=True)
    return x