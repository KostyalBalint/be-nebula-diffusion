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
    english_inputs = tokenizer([sentence], return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
    with torch.no_grad():
        english_outputs = tokenizer_model(**english_inputs).pooler_output

    return english_outputs.cpu().numpy()[0]

ckpt = './nebula_diffusion/trained/conditioned/ckpt_0.000000_45000.pt'

print('Loading model')
ckpt = torch.load(ckpt, map_location=device)

model = FlowVAE(ckpt['args']).to(device)

print('Loading state dict')
model.load_state_dict(ckpt['state_dict'])


def gen_conditioned(text):
    print('Generating 3D model')
    with torch.no_grad():
        encoded_text = tokenize_sentences(text)
        encoded_text = torch.tensor(np.resize(encoded_text, (batch_size, encoded_text.shape[0]))).to(device)

        z = torch.randn([batch_size, ckpt['args'].latent_dim]).to(device)
        x = model.sample(z, encoded_text, sample_num_points, flexibility=ckpt['args'].flexibility, stream_diffusion=True)

    return x