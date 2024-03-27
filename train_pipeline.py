from src.anek_gpt import anekdataset
from src.anek_gpt import tokenizer
from src.anek_gpt import config
from src.anek_gpt import train

def main(ask=False):
    if ask:
        print('Do you want to overwrite the model? (type \'o\')')
        print('Do you want to continue training the model? (type \'c\')')
        print('Type anything else tp abort.')
        ask = input()

    raw_data = anekdataset.load_raw(config.raw_data)

    if ask == 'o':
        print("Forming lookup dicts...")
        tokenizer.form_lookup_dicts(raw_data)
        model = None
    elif ask == 'c':
        from src.anek_gpt.static_model import model
        model = model
    else:
        print('Aborting')
        return
    
    print("Training...")
    train.main()

if __name__ == '__main__':
    main(ask=True)
