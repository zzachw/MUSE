import os

from src.dataset.vocab import build_vocab
from src.utils import processed_data_path, load_pickle


def main():
    all_icu_stay_dict = load_pickle(os.path.join(processed_data_path, "eicu/icu_stay_dict.pkl"))
    all_codes = []
    for icu_id in all_icu_stay_dict.keys():
        for code in all_icu_stay_dict[icu_id].trajectory[1]:
            all_codes.append(code)
    all_codes = list(set(all_codes))
    build_vocab(all_codes, os.path.join(processed_data_path, f"eicu/vocab.pkl"))
    print(f"vocab size: {len(all_codes)}")


if __name__ == "__main__":
    main()
