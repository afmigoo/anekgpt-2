from src.anek_gpt import tokenizer
from src.anek_gpt import datastat
from src.anek_gpt import anekdataset
from src.anek_gpt import config

#tokenizer.form_lookup_dicts(anekdataset.load_raw(config.raw_data))
ad = anekdataset.AnekDataset(config.raw_data)

c = 0
for i in range(1, 1000, 100):
    print(tokenizer.decode(list(map(int, ad[i][0]))))
    c += 1
    if c == 10: break
