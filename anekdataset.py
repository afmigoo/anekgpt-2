from torch.utils.data.dataset import Dataset
from json import dump
from pathlib import Path
import torch
import logging

from config import (
    max_anek_size,
    begin_tkn, end_tkn, filler
)
import tokenizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("AnekDataset")

class AnekDataset(Dataset):
    def __init__(self, file_name: str, max_aneks: int = -1):
        # init
        self.aneks = []
        self.__load_dataset(file_name, max_aneks)

    def __load_dataset(self, file_name: str | Path, max_aneks: int):
        with open(file_name, 'r', encoding='utf-8') as f:
            anek = ''
            for line in f:
                if line =='\n':
                    if anek:
                        self.aneks.append(anek)
                        anek = ''
                        if max_aneks > 0 and \
                           len(self.aneks) >= max_aneks:
                            break
                    continue
                anek += line
            print(f"Loaded {len(self.aneks)} anekdotes")

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        # encode every character to an integer
        anek = self.aneks[index]
        encoded = tokenizer.encode_from_str(anek, normalize_len=True)
        # convert to tensors
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        #print(tokenizer.decode_to_str(encoded))
        return x, y
    
    def __len__(self) -> int:
        return len(self.aneks)

if __name__ == '__main__':
    d = AnekDataset('anekdots.txt', -1)
