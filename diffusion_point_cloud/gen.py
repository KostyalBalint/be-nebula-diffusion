import os

from .models.vae_flow import *

# From paper
ckpt_airplane = './diffusion_point_cloud/trained/GEN_airplane.pt'

# Chair
ckpt_chair = './diffusion_point_cloud/trained/GEN_chair.pt'

ckpt_knife_file = './diffusion_point_cloud/trained/objaverse/knife/ckpt_0.000000_5600.pt'
ckpt_human_file = './diffusion_point_cloud/trained/objaverse/human/ckpt_0.000000_6500.pt'

device = os.getenv("DEVICE", "cpu")
print(f'Device: [{device}]')
sample_num_points = 10000
batch_size = 1


def get_model(mod_ckpt):
    print('Loading model')
    ckpt = torch.load(mod_ckpt, map_location=device)
    model = FlowVAE(ckpt['args']).to(device)
    print('Loading state dict')
    model.load_state_dict(ckpt['state_dict'])
    return model, ckpt


chair_model, chair_ckpt = get_model(ckpt_chair)
airplane_model, airplane_ckpt = get_model(ckpt_airplane)

knife_model, knife_ckpt = get_model(ckpt_knife_file)
human_model, human_ckpt = get_model(ckpt_human_file)


def get_diffusion_model_stats(model):
    if model == 'airplane':
        return airplane_ckpt['args'].__dict__
    elif model == 'chair':
        return chair_ckpt['args'].__dict__
    elif model == 'human':
        return human_ckpt['args'].__dict__
    elif model == 'knife':
        return knife_ckpt['args'].__dict__
    else:
        return None


def gen_diffusion_point_cloud(mod):
    print('Generating 3D model with diffusion_point_cloud')
    if mod == 'chair':
        with torch.no_grad():
            z = torch.randn([batch_size, chair_ckpt['args'].latent_dim]).to(device)
            x = chair_model.sample(z, sample_num_points, flexibility=chair_ckpt['args'].flexibility,
                                   stream_diffusion=True)
        return x
    if mod == 'human':
        with torch.no_grad():
            z = torch.randn([batch_size, human_ckpt['args'].latent_dim]).to(device)
            x = human_model.sample(z, sample_num_points, flexibility=human_ckpt['args'].flexibility,
                                   stream_diffusion=True)
        return x
    if mod == 'knife':
        with torch.no_grad():
            z = torch.randn([batch_size, knife_ckpt['args'].latent_dim]).to(device)
            x = knife_model.sample(z, sample_num_points, flexibility=knife_ckpt['args'].flexibility,
                                   stream_diffusion=True)
        return x
    else:  # airplane
        with torch.no_grad():
            z = torch.randn([batch_size, airplane_ckpt['args'].latent_dim]).to(device)
            x = airplane_model.sample(z, sample_num_points, flexibility=airplane_ckpt['args'].flexibility,
                                      stream_diffusion=True)
        return x
