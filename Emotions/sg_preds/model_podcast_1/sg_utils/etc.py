import os
import torch
import numpy as np
import json


def set_deterministic(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # For Pytorch version >= 1.10
    #torch.use_deterministic_algorithms(True)
    # For Pytorch version < 1.10
    torch.set_deterministic(True)
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
        