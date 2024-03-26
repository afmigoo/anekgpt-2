from json import load
import torch
from mingpt.model import GPT

from anekdataset import itos_file, stoi_file
from train import model_path
from config import (
    max_anek_size,
    get_model_config,
    begin_flag
)

def main():
    model = GPT(get_model_config())
    model.load_state_dict(torch.load(model_path))

    with open(stoi_file, 'r', encoding='utf-8') as f:
        stoi = load(f)
    with open(itos_file, 'r', encoding='utf-8') as f:
        itos = load(f)

    def generate(prompt=begin_flag, steps=1, do_sample=True):
        x = torch.tensor([stoi[tkn] for tkn in prompt], dtype=torch.long)[None,...].to('cpu')

        # forward the model `steps` times to get samples, in a batch
        y = model.generate(x, max_new_tokens=steps, do_sample=do_sample) #top_k=40
        
        all_tokens = [itos[str(int(i))] for i in y[0]]
        out = ''.join(all_tokens)
        return out
    
    anek = begin_flag
    print(anek)
    for _ in range(max_anek_size):
        new_anek = generate(anek)
        print(new_anek.removeprefix(anek), end='' , flush=True)
        anek = new_anek

if  __name__ == "__main__":
    main()
