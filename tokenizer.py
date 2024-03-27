from json import load, dump
from pathlib import Path
from pprint import pprint
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from config import (
    max_tkn_len,
    max_anek_size,
    filler, begin_tkn, end_tkn,
    get_model_config,
    stoi_file, itos_file, raw_data
)
import lookup

def _count_char_freq(seq: str, freq_d: dict[str, int]) -> None:
    for i in range(len(seq)):
        for j in range(i + 1, min(i + max_tkn_len, len(seq)) + 1):
            if seq[i:j] in freq_d:
                freq_d[seq[i:j]] += 1
            else:
                freq_d[seq[i:j]] = 1

def token_grade(token: str, freq: int) -> float:
    return freq * (len(token) ** 3)

def _get_best_tokens(freq_d: dict[str, int]) -> list[str]:
    # adding minimal tokens
    best = [tkn for tkn in freq_d.keys() if len(tkn) == 1]
    best.extend([filler, begin_tkn, end_tkn])
    # grading
    pairs = [(token_grade(tkn, freq), tkn)
             for tkn, freq in freq_d.items()
             if not tkn in best]
    pairs.sort(reverse=True)
    # adding best ones adjusting to the vocab size
    vocab_size = get_model_config().vocab_size
    pairs = pairs[:vocab_size - len(best)]
    best.extend([tkn for _, tkn in pairs])
    return best

def form_lookup_dicts(text: str) -> None:
    """Saves dicts to `stoi_file` and `itos_file`. Bruteforce :("""
    # adding special tokens
    substr_counter = {k: 1 for k in [' ', '\n', filler, begin_tkn, end_tkn]}
    # tokenizing
    #text = word_tokenize(text, language='russian')
    text = text.split(' ')
    # counting frequencies
    for seq in tqdm(text):
        _count_char_freq(seq.lower(), substr_counter)
    # getting optimal tokens
    tokens = _get_best_tokens(substr_counter)
    # saving lookup dicts
    with open(stoi_file, 'w', encoding='utf-8') as f: 
        dump({tkn: i for i, tkn in enumerate(tokens)}, 
             f, ensure_ascii=False, indent=1)
    with open(itos_file, 'w', encoding='utf-8') as f: 
        dump({i: tkn for i, tkn in enumerate(tokens)}, 
             f, ensure_ascii=False, indent=1)

def encode(text_tokens: list[str]) -> list[int]:
    return [lookup.stoi[tkn] for tkn in text_tokens]

def decode(int_tokens: list[int]) -> list[str]:
    return [lookup.itos[tkn] for tkn in int_tokens]

def _encode_seq(text: str) -> list[int]:
    """encodes sequence that doesnt contain spaces"""
    pass

def encode_from_str(text: str, normalize_len: bool = False) -> list[int]:
    text = text.lower()#.split(' ')
    #text = word_tokenize(text)
    int_tokens = []
    i = 0
    while i <= len(text):
        longest_tkn = ''
        for j in range(i + 1, min(i + max_tkn_len, len(text)) + 1):
            if text[i:j] in lookup.stoi and len(text[i:j]) > len(longest_tkn):
                longest_tkn = text[i:j]
        if not longest_tkn:
            i += 1
            continue
        int_tokens.append(lookup.stoi[longest_tkn])
        if len(int_tokens) >= max_anek_size - 2:
            break
        i += len(longest_tkn)
    if normalize_len:
        int_tokens += [lookup.stoi[filler]] * (max_anek_size - len(int_tokens))
    return int_tokens

def decode_to_str(int_tokens: list[int]) -> str:
    return ''.join(decode(int_tokens))

def main():
    with open(raw_data) as f:
        text = f.read()
    form_lookup_dicts(text)
    lookup.reload()

    print(decode(encode_from_str('мир жвачка!!')))

if __name__ == '__main__':
    main()
