import gdown
import os
from .main import MAIA2Model
from .utils import get_all_possible_moves, create_elo_dict, parse_args
from .train import load_model_state_dict, resolve_device
import torch
import warnings
warnings.filterwarnings("ignore")
import pdb

def from_pretrained(type, device="auto", save_root = "./maia2_models"):

    device = resolve_device(device)
    
    if os.path.exists(save_root) == False:
        os.makedirs(save_root)
    
    if type == "blitz":
        url = "https://drive.google.com/uc?id=1X-Z4J3PX3MQFJoa8gRt3aL8CIH0PWoyt"
        output_path = os.path.join(save_root, "blitz_model.pt")
    
    elif type == "rapid":
        url = "https://drive.google.com/uc?id=1gbC1-c7c0EQOPPAVpGWubezeEW8grVwc"
        output_path = os.path.join(save_root, "rapid_model.pt")
    
    else:
        raise ValueError("Invalid model type. Choose between 'blitz' and 'rapid'.")

    if os.path.exists(output_path):
        print(f"Model for {type} games already downloaded.")
    else:
        print(f"Downloading model for {type} games.")
        gdown.download(url, output_path, quiet=False)

    cfg_url = "https://drive.google.com/uc?id=1GQTskYMVMubNwZH2Bi6AmevI15CS6gk0"
    cfg_path = os.path.join(save_root, "config.yaml")
    if not os.path.exists(cfg_path):
        gdown.download(cfg_url, cfg_path, quiet=False)

    cfg = parse_args(cfg_path)

    all_moves = get_all_possible_moves()
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    
    checkpoint = torch.load(output_path, map_location='cpu')
    load_model_state_dict(model, checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model for {type} games loaded to {device}.")
    
    return model
    
    
    
    
    
    
