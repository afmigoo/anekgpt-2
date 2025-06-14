from json import load

from .config import stoi_file, itos_file

# global lookup dicts
# stoi = string_to_integer
# itos = integer_to_string
stoi = itos = None

def reload():
    # reloads lookup dicts from files
    global stoi, itos

    if stoi_file.is_file():
        with open(stoi_file, 'r', encoding='utf-8') as f:
            stoi = {k: int(v) for k, v in load(f).items()}
    else: stoi = None
    if itos_file.is_file():
        with open(itos_file, 'r', encoding='utf-8') as f:
            itos = {int(k): v for k, v in load(f).items()}
    else: itos = None

reload()
