import os
import torch
import time
import numpy as np
import json
def set_deterministic(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # torch.set_deterministic(True)
    torch.manual_seed(seed)
    np.random.seed(seed)

def print_config_description(conf_path):
    with open(conf_path, 'r') as f:
        config_dict = json.load(f)
    description = config_dict.get("description", None)
    if description is not None:
        print(description)
    else:
        print("Configuration file does not contain a description")
        print("We highly recommend you to add a description to the configuration file for the debugging")

def reverse_dict(original_dict):
    return {v: k for k, v in original_dict.items()}


def print_decoded(decoder, emission):
    result = decoder(emission)
    transcript = " ".join(result[0][0].words).lower().strip()
    score = result[0][0].score
    return transcript, score
    