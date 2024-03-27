from json import load
import torch
from mingpt.model import GPT

from config import (
    max_anek_size,
    get_model_config,
    begin_tkn, end_tkn,
    model_path
)
import tokenizer

def main(model = None):
    if model is None:
        model = GPT(get_model_config())
        model.load_state_dict(torch.load(model_path))

    def generate(prompt=begin_tkn, steps=1, do_sample=True):
        encoded = tokenizer.encode_from_str(prompt)
        x = torch.tensor(encoded, dtype=torch.long)[None,...].to('cpu')

        # forward the model `steps` times to get samples, in a batch
        y = model.generate(x, max_new_tokens=steps, do_sample=do_sample) #top_k=40
        
        return tokenizer.decode_to_str(list(map(int, y[0])))
    
    anek = begin_tkn
    print(anek)
    for _ in range(max_anek_size):
        new_anek = generate(anek)
        print(new_anek.removeprefix(anek), end='' , flush=True)
        anek = new_anek
    print()

if  __name__ == "__main__":
    main()
