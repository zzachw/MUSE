import os

import torch
from torch.utils.data import Dataset

from src.utils import processed_data_path, read_txt, load_pickle


class ADNIDataset(Dataset):
    def __init__(self, split, task="y", load_no_label=False, dev=False):
        if dev:
            assert split == "train"
        if load_no_label:
            assert split == "train"
        self.load_no_label = load_no_label
        self.split = split
        self.task = task
        self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "adni/adni_data_dict.pkl"))
        included_admission_ids = read_txt(
            os.path.join(processed_data_path, f"adni/task:{task}/{split}_admission_ids.txt"))
        self.no_label_admission_ids = []
        if load_no_label:
            no_label_admission_ids = read_txt(
                os.path.join(processed_data_path, f"adni/task:{task}/no_label_admission_ids.txt"))
            self.no_label_admission_ids = no_label_admission_ids
            included_admission_ids += no_label_admission_ids
        self.included_admission_ids = included_admission_ids
        if dev:
            self.included_admission_ids = self.included_admission_ids[:10000]

    def __len__(self):
        return len(self.included_admission_ids)

    def __getitem__(self, index):
        icu_id = self.included_admission_ids[index]
        icu_stay = self.all_hosp_adm_dict[icu_id]

        x1 = icu_stay.x1
        x1_flag = True
        if x1 is None:
            x1 = torch.zeros(228)
            x1_flag = False
        else:
            x1 = torch.FloatTensor(x1)

        x2 = icu_stay.x2
        x2_flag = True
        if x2 is None:
            x2 = torch.zeros(227)
            x2_flag = False
        else:
            x2 = torch.FloatTensor(x2)

        x3 = icu_stay.x3
        x3_flag = True
        if x3 is None:
            x3 = torch.zeros(620)
            x3_flag = False
        else:
            x3 = torch.FloatTensor(x3)

        label = getattr(icu_stay, self.task)
        label_flag = True
        if label is None:
            label = 0
            label_flag = False
        else:
            label = int(label)

        return_dict = dict()
        return_dict["id"] = icu_id

        return_dict["x1"] = x1
        return_dict["x1_flag"] = x1_flag

        return_dict["x2"] = x2
        return_dict["x2_flag"] = x2_flag

        return_dict["x3"] = x3
        return_dict["x3_flag"] = x3_flag

        return_dict["label"] = label
        return_dict["label_flag"] = label_flag

        return return_dict


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = ADNIDataset(split="train", load_no_label=True)
    print(len(dataset))
    item = dataset[0]
    print(item["id"])
    print(item["x1"].shape)
    print(item["x2"].shape)
    print(item["x3"].shape)
    print(item["label"])

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(data_loader))
    print(batch["id"])
    print(batch["x1"].shape)
    print(batch["x1_flag"])
    print(batch["x2"].shape)
    print(batch["x2_flag"])
    print(batch["x3"].shape)
    print(batch["x3_flag"])
    print(batch["label"])
    print(batch["label_flag"])
