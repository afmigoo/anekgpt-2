from torch.utils.data.dataset import Dataset
from json import dump
from pathlib import Path
import torch
import logging

from config import (
    max_anek_size,
    begin_flag, end_flag, filler
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("AnekDataset")
data_dir = Path('data')
stoi_file = data_dir.joinpath('stoi.json')
itos_file = data_dir.joinpath('itos.json')

class AnekDataset(Dataset):
    def __init__(self, file_name: str, max_aneks: int = -1):
        # init
        self.aneks = []
        self.unique_chars = set()
        self.unique_chars.update([begin_flag, end_flag, filler])
        # loading
        logger.info(f"Loading dataset...")
        self.__load_dataset(file_name, max_aneks)
        # forming lookup dicts
        self.itos = dict(enumerate(self.unique_chars))
        self.stoi = {v: k for k, v in self.itos.items()}
        # saving lookup dicts
        with open(stoi_file, 'w', encoding='utf-8') as f: 
            dump(self.stoi, f, ensure_ascii=False, indent=1)
        with open(itos_file, 'w', encoding='utf-8') as f: 
            dump(self.itos, f, ensure_ascii=False, indent=1)

    def __load_dataset(self, file_name: str | Path, max_aneks: int):
        with open(file_name, 'r', encoding='utf-8') as f:
            anek = ''
            total_chars = 0
            for line in f:
                if line =='\n':
                    if anek:
                        self.unique_chars.update(*anek)
                        self.aneks.append(anek)
                        total_chars += len(anek)
                        anek = ''
                        if max_aneks > 0 and \
                           len(self.aneks) >= max_aneks:
                            break
                    continue
                anek += line
            logger.info(f"Loaded {len(self.aneks)} anekdotes")
            logger.info(f"Total chars: {total_chars}")
            logger.info(f"Unique chars: {len(self.unique_chars)}")

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        # encode every character to an integer
        anek = self.aneks[index]
        used_len = len(begin_flag) + len(anek) + len(end_flag)
        anek = "{beg}{anek}{end}{filler}".format(
            beg = begin_flag,
            anek = anek,
            end = end_flag,
            filler = filler * (max_anek_size - used_len),
        )
        encoded = [self.stoi[s] for s in anek]
        # convert to tensors
        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)
        return x, y
    
    def __len__(self) -> int:
        return len(self.aneks)

if __name__ == '__main__':
    d = AnekDataset('anekdots.txt', -1)
