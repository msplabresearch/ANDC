import torch
import os
from src.dataset import Audiovisual_dataset

def save_model(args, model, name=''):
    torch.save(model, f'saved_models/{name}.pth')


def load_model(args, name=''):
    model = torch.load(name, map_location=torch.device('cpu'))
    return model


def get_data(dataset, split='train'):
    data = Audiovisual_dataset( dataset, split)
 
    return data
