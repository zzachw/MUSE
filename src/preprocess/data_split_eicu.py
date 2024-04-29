import os
import random

from src.utils import processed_data_path, load_pickle, set_seed, create_directory, write_txt


def label_split(all_hosp_adm_dict, task):
    label_hosp_adm_dict = {}
    no_label_hosp_adm_dict = {}
    for admission_id, admission in all_hosp_adm_dict.items():
        label = getattr(admission, task)
        # random dropout by 50%
        if random.random() < 0.5:
            label_hosp_adm_dict[admission_id] = admission
        else:
            no_label_hosp_adm_dict[admission_id] = admission
    return label_hosp_adm_dict, no_label_hosp_adm_dict


def tvt_split(hosp_adm_dict, ratio=None):
    if ratio is None:
        ratio = [0.7, 0.1, 0.2]
    all_admission_ids = list(hosp_adm_dict.keys())
    random.shuffle(all_admission_ids)
    s1 = ratio[0]
    s2 = ratio[0] + ratio[1]
    train_admission_ids = all_admission_ids[:int(len(all_admission_ids) * s1)]
    val_admission_ids = all_admission_ids[int(len(all_admission_ids) * s1): int(len(all_admission_ids) * s2)]
    test_admission_ids = all_admission_ids[int(len(all_admission_ids) * s2):]
    return train_admission_ids, val_admission_ids, test_admission_ids


def main():
    set_seed(42)
    all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "eicu/icu_stay_dict.pkl"))
    for task in ["mortality", "readmission"]:
        label_hosp_adm_dict, no_label_hosp_adm_dict = label_split(all_hosp_adm_dict, task)
        train_admission_ids, val_admission_ids, test_admission_ids = tvt_split(label_hosp_adm_dict)
        no_label_admission_ids = list(no_label_hosp_adm_dict.keys())
        output_path = os.path.join(processed_data_path, f"eicu/task:{task}")
        create_directory(output_path)
        write_txt(os.path.join(output_path, "train_admission_ids.txt"), train_admission_ids)
        write_txt(os.path.join(output_path, "val_admission_ids.txt"), val_admission_ids)
        write_txt(os.path.join(output_path, "test_admission_ids.txt"), test_admission_ids)
        write_txt(os.path.join(output_path, "no_label_admission_ids.txt"), no_label_admission_ids)
        print(f"task: {task}")
        print(f"train: {len(train_admission_ids)}")
        print(f"val: {len(val_admission_ids)}")
        print(f"test: {len(test_admission_ids)}")
        print(f"no label: {len(no_label_admission_ids)}")
    return


if __name__ == "__main__":
    main()
