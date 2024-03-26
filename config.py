from mingpt.model import GPT
from mingpt.trainer import Trainer

begin_flag = '[<BEG>]'
end_flag = '[<END>]'
filler = '⚧'

max_anek_size = 256
max_anek_count = -1

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 256
model_config.block_size = 256

#############################

train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # many possible options, see the file
train_config.max_iters = 70
train_config.batch_size = 6
train_config.num_workers = 2
