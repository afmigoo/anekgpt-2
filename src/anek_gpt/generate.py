from mingpt.model import GPT
import torch

from .config import (
    max_anek_size,
    get_model_config,
    begin_tkn,
    model_path
)
from . import tokenizer

def generate(model=None, prompt=begin_tkn, steps=1, do_sample=True):
    # takes string prompt, tokenizes it and feets into the model.
    # returns prompt + `steps` next tokens all converted to text
    if model is None:
        model = GPT(get_model_config())
        model.load_state_dict(torch.load(model_path))
    
    encoded = tokenizer.encode_from_str(prompt)
    # it's from stack overflow, I dont know what's happening here
    x = torch.tensor(encoded, dtype=torch.long)[None,...].to('cpu')

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample)
    
    return tokenizer.decode_to_str(list(map(int, y[0])))

def main(model = None):
    # load model if None
    if model is None:
        model = GPT(get_model_config())
        model.load_state_dict(torch.load(model_path))
    
    # start with separator token
    anek = begin_tkn
    print(anek)
    # generating `max_anek_size` tokens
    for _ in range(max_anek_size):
        # generating next token
        new_anek = generate(model=model, prompt=anek)
        # printing only last recieved token
        print(new_anek.removeprefix(anek), end='' , flush=True)
        anek = new_anek
        # stop if met separator token
        if anek.endswith(begin_tkn):
            break
    print()

if  __name__ == "__main__":
    main()
