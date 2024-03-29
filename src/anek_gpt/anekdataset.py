from torch.utils.data.dataset import Dataset
from pathlib import Path
import torch

from . import tokenizer
from .config import begin_tkn, get_model_config

def load_raw(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        raw = f.read()
    return raw

class AnekDataset(Dataset):
    def __init__(self, file_name: str):
        # initializing the dataset
        self.data: list[int] = None
        self.block_size = get_model_config().block_size
        # loading data from file
        self.__load_dataset(file_name)

    def __load_dataset(self, file_name: str | Path):
        # loads data from file to self.data
        with open(file_name, 'r', encoding='utf-8') as f:
            self.data = f.read()
            print(f"Loaded {len(self.data)} chars")
            print("Tokenizing data...")
            # inserting split tokens between anekdotes
            self.data = self.data.replace('\n\n', begin_tkn)
            # encoding data to integer tokens
            self.data = tokenizer.encode_from_str(self.data)
            print(f"{len(self.data)} total tokens.")

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        # returning block_size tokens with shifted y
        x = torch.tensor(self.data[index:index + self.block_size], 
                         dtype=torch.long)
        y = torch.tensor(self.data[index + 1:index + self.block_size + 1],
                         dtype=torch.long)
        return x, y
    
    def __len__(self) -> int:
        return len(self.data) - self.block_size

if __name__ == '__main__':
    d = AnekDataset('anekdots.txt')
