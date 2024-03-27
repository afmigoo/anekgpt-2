from torch.utils.data.dataset import Dataset
from pathlib import Path
import torch

from . import tokenizer
from . lookup import stoi
from .config import begin_tkn, end_tkn, get_model_config, max_anek_size

def load_raw(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        raw = f.read()
    return raw

class AnekDataset(Dataset):
    def __init__(self, file_name: str, max_aneks: int = -1):
        # init
        self.data = None
        self.block_size = get_model_config().block_size
        self.__load_dataset(file_name, max_aneks)

    def __load_dataset(self, file_name: str | Path, max_aneks: int):
        with open(file_name, 'r', encoding='utf-8') as f:
            self.data = f.read()
            print(f"Loaded {len(self.data)} chars")
            print("Tokenizing data...")
            self.data = self.data.replace('\n\n', begin_tkn)
            self.data = tokenizer.encode_from_str(self.data)
            print(f"{len(self.data)} total tokens.")

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[index:index + self.block_size], 
                         dtype=torch.long)
        y = torch.tensor(self.data[index + 1:index + self.block_size + 1],
                         dtype=torch.long)
        return x, y
    
    def __len__(self) -> int:
        return len(self.data) - self.block_size

if __name__ == '__main__':
    d = AnekDataset('anekdots.txt', -1)
