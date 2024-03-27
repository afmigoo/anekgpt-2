from mingpt.model import GPT
import torch

from .config import (
    get_model_config,
    model_path
)

model = None

def load():
    global model
    if model is not None:
        print('Model is already loaded.')
        return
    model = GPT(get_model_config())
    model.load_state_dict(torch.load(model_path))
