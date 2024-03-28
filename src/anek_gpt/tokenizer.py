from nltk.tokenize import word_tokenize
from json import dump
from tqdm import tqdm

from .config import (
    max_tkn_len,
    begin_tkn,
    get_model_config,
    stoi_file, itos_file, raw_data
)
from . import lookup

def _count_char_freq(seq: str, freq_d: dict[str, int]) -> None:
    # counting substring frequencies in `seq` with max len `max_tkn_len`
    # `seq` does not contain spaces. we dont want to have tokens like 'елать прав'
    for i in range(len(seq)):
        for j in range(i + 1, min(i + max_tkn_len, len(seq)) + 1):
            if seq[i:j] in freq_d:
                freq_d[seq[i:j]] += 1
            else:
                freq_d[seq[i:j]] = 1

def token_grade(token: str, freq: int) -> float:
    # grading tokens, we want longer tokens to be more valuable
    return freq * (len(token) ** 1)

def _get_best_tokens(freq_d: dict[str, int]) -> list[str]:
    # getting the best graded tokens

    # adding minimal tokens, we always want to have single character tokens
    best = [tkn for tkn in freq_d.keys() if len(tkn) == 1]
    best.append(begin_tkn)
    # grading
    pairs = [(token_grade(tkn, freq), tkn)
             for tkn, freq in freq_d.items()
             if not tkn in best]
    # sorting so the best are in the front
    pairs.sort(reverse=True)
    # adding best ones adjusting to the vocab size
    vocab_size = get_model_config().vocab_size
    pairs = pairs[:vocab_size - len(best)]
    best.extend([tkn for _, tkn in pairs])
    return best

def form_lookup_dicts(text: str) -> None:
    # Forms lookup tken dicts and
    # saves them to `stoi_file` and `itos_file`.

    # adding special tokens
    substr_counter = {k: 1 for k in [' ', '\n', begin_tkn]}
    # tokenizing
    # dont use nltk tokenizer since it vipes '\n' chars
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
    # reloading dicts
    lookup.reload()

def encode(text_tokens: list[str]) -> list[int]:
    return [lookup.stoi[tkn] for tkn in text_tokens]

def decode(int_tokens: list[int]) -> list[str]:
    return [lookup.itos[tkn] for tkn in int_tokens]

def encode_from_str(text: str) -> list[int]:
    # takes a string and encodes it with longest possible tokens
    text = text.lower()
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
        i += len(longest_tkn)
    return int_tokens

def decode_to_str(int_tokens: list[int]) -> str:
    # decode and convert to string
    return ''.join(decode(int_tokens))

def main():
    # reading data
    with open(raw_data) as f:
        text = f.read()
    # forming lookup dicts
    form_lookup_dicts(text)
    # printing sample
    print(decode(encode_from_str('мир жвачка!!')))

if __name__ == '__main__':
    main()
