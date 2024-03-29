from pathlib import Path
from mingpt.model import GPT
from mingpt.trainer import Trainer

# files
data_dir = Path(__file__).parent.joinpath('data')
#raw_data = data_dir.joinpath('aneks.txt')
raw_data = data_dir.joinpath('vsya-agata-kristi-v-odnom-tome.txt')
stoi_file = data_dir.joinpath('stoi.json')
itos_file = data_dir.joinpath('itos.json')
model_path = data_dir.joinpath('model.pt')
# special tokens
# separator token
begin_tkn = '[anek]'
# token maximum length
max_tkn_len = 7

def get_model_config():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    # amount of tokens
    model_config.vocab_size = 16384
    model_config.block_size = 18
    return model_config

def get_train_config():
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 6e-4
    train_config.max_iters = 100000
    train_config.batch_size = 2
    train_config.num_workers = 2
    return train_config
