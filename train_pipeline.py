from src.anek_gpt import anekdataset
from src.anek_gpt import tokenizer
from src.anek_gpt import config
from src.anek_gpt import train
from src.anek_gpt.config import model_path

raw_data = anekdataset.load_raw(config.raw_data)

print('Do you want to overwrite the model? (type \'o\')')
print('Do you want to continue training the model? (type \'c\')')
print('Type anything else tp abort.')
ask = input()

if ask == 'o':
    # forming lookup dicts for the tokenizer
    print("Forming lookup dicts...")
    tokenizer.form_lookup_dicts(raw_data)
    # if overwrite then delete prev model
    model = None
    model_path.unlink(missing_ok=True)
elif ask == 'c':
    # if continue training then load prev model
    from src.anek_gpt.static_model import model
    model = model
else:
    # else exit
    print('Aborting')
    exit()

# start training
print("Training...")
train.main(model)