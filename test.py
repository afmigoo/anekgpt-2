from src.anek_gpt import tokenizer
from src.anek_gpt import datastat
from src.anek_gpt import anekdataset
from src.anek_gpt import config

a = anekdataset.AnekDataset(config.raw_data)
#tokenizer.form_lookup_dicts()
#datastat.main()