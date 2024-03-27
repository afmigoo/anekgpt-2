from pathlib import Path
from mingpt.model import GPT
from mingpt.trainer import Trainer

# files
data_dir = Path('data')
raw_data = data_dir.joinpath('anekdots.txt')
stoi_file = data_dir.joinpath('stoi.json')
itos_file = data_dir.joinpath('itos.json')
model_path = data_dir.joinpath('model.pt')
# special tokens
begin_tkn = '[<BEG>]'
end_tkn = '[<END>]'
filler = '⚧'
max_tkn_len = 10

max_anek_size = 128
max_anek_count = -1

def get_model_config():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = 2048
    model_config.block_size = 128
    return model_config

def get_train_config():
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # many possible options, see the file
    train_config.max_iters = 2000
    train_config.batch_size = 4
    train_config.num_workers = 2
    return train_config
