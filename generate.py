from json import load
import torch
from mingpt.model import GPT

from train import model_path
from config import (
    max_anek_size,
    get_model_config,
    begin_tkn
)
import tokenizer

def main():
    model = GPT(get_model_config())
    model.load_state_dict(torch.load(model_path))

    def generate(prompt=begin_tkn, steps=1, do_sample=True):
        x = torch.tensor(tokenizer.encode_from_str(prompt), dtype=torch.long)[None,...].to('cpu')

        # forward the model `steps` times to get samples, in a batch
        y = model.generate(x, max_new_tokens=steps, do_sample=do_sample) #top_k=40
        
        return tokenizer.decode_to_str(list(map(int, y[0])))
    
    anek = begin_tkn
    print(anek)
    for _ in range(max_anek_size):
        new_anek = generate(anek)
        print(new_anek.removeprefix(anek), end='' , flush=True)
        anek = new_anek

if  __name__ == "__main__":
    main()
