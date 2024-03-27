from pathlib import Path
from mingpt.model import GPT
from mingpt.trainer import Trainer

# files
data_dir = Path(__file__).parent.joinpath('data')
raw_data = data_dir.joinpath('anekdots.txt')
stoi_file = data_dir.joinpath('stoi.json')
itos_file = data_dir.joinpath('itos.json')
model_path = data_dir.joinpath('model.pt')
# special tokens
begin_tkn = '[anek]'
end_tkn = '[;]'
max_tkn_len = 10

max_anek_size = 64
max_anek_count = -1

def get_model_config():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-mini'
    model_config.vocab_size = 4096
    model_config.block_size = 64
    return model_config

def get_train_config():
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 6e-4 # many possible options, see the file
    train_config.max_iters = 100_000
    train_config.batch_size = 32
    train_config.num_workers = 2
    return train_config
