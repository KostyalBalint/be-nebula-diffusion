import os

import numpy as np
from transformers import BertTokenizerFast, BertModel
from .models.vae_flow import *

device = os.getenv("DEVICE", "cpu")
print(f'Device: [{device}]')
sample_num_points = 10000
batch_size = 1


def tokenize_sentences(sentence):
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-small")
    tokenizer_model = BertModel.from_pretrained("setu4993/LEALLA-small").to(device)
    tokenizer_model = tokenizer_model.eval()
    english_inputs = tokenizer([sentence], return_tensors="pt", padding=True, max_length=512, truncation=True).to(
        device)
    with torch.no_grad():
        english_outputs = tokenizer_model(**english_inputs).pooler_output

    return english_outputs.cpu().numpy()[0]


def get_model(mod_ckpt):
    print('Loading model')
    ckpt = torch.load(mod_ckpt, map_location=device)
    model = FlowVAE(ckpt['args']).to(device)
    print('Loading state dict')
    model.load_state_dict(ckpt['state_dict'])
    return model, ckpt


objaverse_v1 = './nebula_diffusion/trained/conditioned/ckpt_objaverse_V1_45000.pt'  # Objaverse V1
objaverse_v2 = './nebula_diffusion/trained/conditioned/ckpt_objaverse_V2_86000.pt'  # Objaverse V2
shapenet_v1 = './nebula_diffusion/trained/conditioned_shapenet/ckpt_0.000000_100000.pt'  # Shapenet - small
shapenet_v2 = './nebula_diffusion/trained/conditioned_shapenet/ckpt_0.000000_124000.pt' # Shapenet - big

#ckpt_test = './nebula_diffusion/trained/conditioned_shapenet/ckpt_0.000000_35000.pt'

objaverse_v1_model, objaverse_v1_ckpt = get_model(objaverse_v1)
objaverse_v2_model, objaverse_v2_ckpt = get_model(objaverse_v2)
shapenet_v1_model, shapenet_v1_ckpt = get_model(shapenet_v1)
shapenet_v2_model, shapenet_v2_ckpt = get_model(shapenet_v2)

#_, test_ckpt = get_model(ckpt_test)


def get_nebula_model_stats(model):
    if model == 'objaverse_v1':
        return objaverse_v1_ckpt['args'].__dict__
    if model == 'objaverse_v2':
        return objaverse_v2_ckpt['args'].__dict__
    elif model == 'shapenet_v1':
        return shapenet_v1_ckpt['args'].__dict__
    elif model == 'shapenet_v2':
        return shapenet_v2_ckpt['args'].__dict__
    else:
        return None

def gen_conditioned(model_variant, text):
    print('Generating 3D model with nebula diffusion')
    with torch.no_grad():
        encoded_text = tokenize_sentences(text)
        encoded_text = torch.tensor(np.resize(encoded_text, (batch_size, encoded_text.shape[0]))).to(device)

        if model_variant == 'objaverse_v1':
            z = torch.randn([batch_size, objaverse_v1_ckpt['args'].latent_dim]).to(device)
            x = objaverse_v1_model.sample(z, encoded_text, sample_num_points,
                                          flexibility=objaverse_v1_ckpt['args'].flexibility, stream_diffusion=True)
        elif model_variant == 'objaverse_v2':
            z = torch.randn([batch_size, objaverse_v2_ckpt['args'].latent_dim]).to(device)
            x = objaverse_v2_model.sample(z, encoded_text, sample_num_points,
                                          flexibility=objaverse_v2_ckpt['args'].flexibility, stream_diffusion=True)
        elif model_variant == 'shapenet_v1':  # shapenet_v1
            z = torch.randn([batch_size, shapenet_v1_ckpt['args'].latent_dim]).to(device)
            x = shapenet_v1_model.sample(z, encoded_text, sample_num_points,
                                      flexibility=shapenet_v1_ckpt['args'].flexibility, stream_diffusion=True)
        else:  # shapenet_v2
            z = torch.randn([batch_size, shapenet_v2_ckpt['args'].latent_dim]).to(device)
            x = shapenet_v2_model.sample(z, encoded_text, sample_num_points,
                                      flexibility=shapenet_v2_ckpt['args'].flexibility, stream_diffusion=True)

    return x
