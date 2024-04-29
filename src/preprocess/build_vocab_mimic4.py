import os

from src.dataset.vocab import build_vocab
from src.utils import processed_data_path, load_pickle


def main():
    all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "mimic4/hosp_adm_dict.pkl"))
    all_codes = []
    for icu_id in all_hosp_adm_dict.keys():
        for code in all_hosp_adm_dict[icu_id].trajectory[1]:
            all_codes.append(code)
    all_codes = list(set(all_codes))
    build_vocab(all_codes, os.path.join(processed_data_path, f"mimic4/vocab.pkl"))
    print(f"vocab size: {len(all_codes)}")


if __name__ == "__main__":
    main()
