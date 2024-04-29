import os

import torch
from torch.utils.data import Dataset

from src.dataset.tokenizer import MIMIC4Tokenizer
from src.utils import processed_data_path, read_txt, load_pickle


class MIMIC4Dataset(Dataset):
    def __init__(self, split, task, load_no_label=False, dev=False, return_raw=False):
        if dev:
            assert split == "train"
        if load_no_label:
            assert split == "train"
        self.load_no_label = load_no_label
        self.split = split
        self.task = task
        self.all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "mimic4/hosp_adm_dict_v2.pkl"))
        included_admission_ids = read_txt(
            os.path.join(processed_data_path, f"mimic4/task:{task}/{split}_admission_ids.txt"))
        self.no_label_admission_ids = []
        if load_no_label:
            no_label_admission_ids = read_txt(
                os.path.join(processed_data_path, f"mimic4/task:{task}/no_label_admission_ids.txt"))
            self.no_label_admission_ids = no_label_admission_ids
            included_admission_ids += no_label_admission_ids
        self.included_admission_ids = included_admission_ids
        if dev:
            self.included_admission_ids = self.included_admission_ids[:10000]
        self.return_raw = return_raw
        self.tokenizer = MIMIC4Tokenizer()

    def __len__(self):
        return len(self.included_admission_ids)

    def __getitem__(self, index):
        admission_id = self.included_admission_ids[index]
        hosp_adm = self.all_hosp_adm_dict[admission_id]

        age = str(hosp_adm.age)
        gender = hosp_adm.gender
        ethnicity = hosp_adm.ethnicity
        types = hosp_adm.trajectory[0]
        codes = hosp_adm.trajectory[1]
        codes_flag = True

        labvectors = hosp_adm.labvectors
        labvectors_flag = True
        if labvectors is None:
            labvectors = torch.zeros(1, 114)
            labvectors_flag = False
        else:
            labvectors = torch.FloatTensor(labvectors)

        discharge = hosp_adm.discharge
        discharge_flag = True
        if discharge is None:
            discharge = ""
            discharge_flag = False

        label = getattr(hosp_adm, self.task)
        label_flag = True
        if label is None:
            label = 0.0
            label_flag = False
        else:
            label = float(label)

        if not self.return_raw:
            age, gender, ethnicity, types, codes = self.tokenizer(
                age, gender, ethnicity, types, codes
            )
            label = torch.tensor(label)

        return_dict = dict()
        return_dict["id"] = admission_id

        return_dict["age"] = age
        return_dict["gender"] = gender
        return_dict["ethnicity"] = ethnicity
        return_dict["types"] = types
        return_dict["codes"] = codes
        return_dict["codes_flag"] = codes_flag

        return_dict["labvectors"] = labvectors
        return_dict["labvectors_flag"] = labvectors_flag

        return_dict["discharge"] = discharge
        return_dict["discharge_flag"] = discharge_flag

        return_dict["label"] = label
        return_dict["label_flag"] = label_flag

        return return_dict


if __name__ == "__main__":
    dataset = MIMIC4Dataset(split="train", task="mortality", load_no_label=True, return_raw=True)
    print(len(dataset))
    item = dataset[0]
    print(item["id"])
    print(item["age"])
    print(item["gender"])
    print(item["ethnicity"])
    print(len(item["types"]))
    print(len(item["codes"]))
    print(item["labvectors"].shape)
    print(item["discharge"])
    print(item["label"])

    from torch.utils.data import DataLoader
    from src.dataset.utils import mimic4_collate_fn

    dataset = MIMIC4Dataset(split="train", task="mortality", load_no_label=True)
    print(len(dataset))
    item = dataset[0]
    print(item["id"])
    print(item["age"])
    print(item["gender"])
    print(item["ethnicity"])
    print(item["types"].shape)
    print(item["codes"].shape)
    print(item["label"].shape)

    data_loader = DataLoader(dataset, batch_size=32, collate_fn=mimic4_collate_fn, shuffle=True)
    batch = next(iter(data_loader))
    print(batch["age"])
    print(batch["gender"])
    print(batch["ethnicity"])
    print(batch["types"].shape)
    print(batch["codes"].shape)
    print(batch["codes_flag"])
    print(batch["labvectors"].shape)
    print(batch["labvectors_flag"])
    print(batch["discharge"])
    print(batch["discharge_flag"])
    print(batch["label"])
    print(batch["label_flag"])
