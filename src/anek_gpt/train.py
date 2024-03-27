from mingpt.trainer import Trainer
from datetime import datetime
from mingpt.model import GPT
import torch

from .anekdataset import AnekDataset
from . import generate
from .config import (
    get_model_config,
    get_train_config,
    max_anek_count,
    raw_data,
    model_path
)

def main(model = None):
    if model is None:
        model = GPT(get_model_config())
        if model_path.is_file():
            model.load_state_dict(torch.load(model_path))
    
    train_dataset = AnekDataset(raw_data, max_anek_count)
    trainer = Trainer(get_train_config(), model, train_dataset)

    def batch_end_callback(trainer: Trainer):
        if trainer.iter_num % 1 == 0:
            print("{time} iter {iter}: train loss {loss:.5f}".format(
                time=datetime.now().time().strftime('%H:%M:%S'),
                iter=trainer.iter_num,
                loss=trainer.loss.item()
            ))
        if trainer.iter_num % 5 == 0:
            # save the latest model
            print("Saving the model...")
            torch.save(model.state_dict(), model_path)
            generate.main(model)

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    
    print(f"iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    print("Saving the model...")
    torch.save(model.state_dict(), model_path)

if  __name__ == "__main__":
    main()
