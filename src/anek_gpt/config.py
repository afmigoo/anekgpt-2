from pathlib import Path
from mingpt.model import GPT
from mingpt.trainer import Trainer

# files
data_dir = Path(__file__).parent.joinpath('data')
raw_data = data_dir.joinpath('aneks.txt')
stoi_file = data_dir.joinpath('stoi.json')
itos_file = data_dir.joinpath('itos.json')
model_path = data_dir.joinpath('model.pt')
# special tokens
# separator token
begin_tkn = '[anek]'
# token maximum length
max_tkn_len = 10
# token anekdote size and model's block_size
max_anek_size = 16

def get_model_config():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    # amount of tokens
    model_config.vocab_size = 4096
    model_config.block_size = max_anek_size
    return model_config

def get_train_config():
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 6e-4
    train_config.max_iters = 100000
    train_config.batch_size = 32
    train_config.num_workers = 2
    return train_config
