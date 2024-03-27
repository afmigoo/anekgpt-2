def main():
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from .config import raw_data, end_tkn
    from .anekdataset import AnekDataset
    from . import tokenizer
    from . import lookup

    ad = AnekDataset(raw_data)

    lenghts = []
    tkn_lenghts = []
    n_bins = 20

    for anek in tqdm(ad):
        anek = list(map(int, anek[0][1:]))
        try:
            end_idx = anek.index(lookup.stoi[end_tkn])
        except ValueError:
            end_idx = -1
        anek = anek[:end_idx]
        tkn_lenghts.append(len(anek))
        anek = tokenizer.decode_to_str(list(map(int, anek)))
        lenghts.append(len(anek))

    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(lenghts, bins=n_bins)
    axs[1].hist(tkn_lenghts, bins=n_bins)
    plt.show()

if __name__ == '__main__':
    main()
