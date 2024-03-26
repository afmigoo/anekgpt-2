from mingpt.model import GPT
from mingpt.trainer import Trainer
import torch
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("trainer")

from anekdataset import AnekDataset
from config import (
    get_model_config,
    get_train_config,
    max_anek_count
)

model_path = "data/model.pt"

def main():
    model = GPT(get_model_config())
    train_dataset = AnekDataset('anekdots.txt', max_anek_count)
    trainer = Trainer(get_train_config(), model, train_dataset)

    def batch_end_callback(trainer: Trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 100 == 0:
            # save the latest model
            print("Saving the model...")
            torch.save(model.state_dict(), model_path)

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    
    print(f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    print("Saving the model...")
    torch.save(model.state_dict(), model_path)

if  __name__ == "__main__":
    main()
