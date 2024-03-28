def main():
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from .config import raw_data
    from .anekdataset import AnekDataset, load_raw
    from . import tokenizer

    tokenizer.form_lookup_dicts(load_raw(raw_data))
    ad = AnekDataset(raw_data)

    lenghts = []
    n_bins = 20

    for i in tqdm(range(len(ad)), total=len(ad)):
        anek = list(map(int, ad[i][0]))
        anek = tokenizer.decode_to_str(list(map(int, anek)))
        lenghts.append(len(anek))

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(lenghts, bins=n_bins)
    plt.show()

if __name__ == '__main__':
    main()
