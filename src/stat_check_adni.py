import os
from collections import Counter

import numpy as np

from src.utils import processed_data_path, load_pickle


def list_occurrence_counter(lst):
    counter = Counter(lst)
    keys = sorted(counter.keys())
    message = []
    for k in keys:
        count = counter[k]
        percentage = count / len(lst)
        message.append(f"{k}: {percentage:.2f}")
    return "{" + ", ".join(message) + "}"


all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "adni/adni_data_dict.pkl"))
print('# patients', len(set([v.patient_id for v in all_hosp_adm_dict.values()])))
print('# admissions', len(all_hosp_adm_dict))
print("gender:", list_occurrence_counter([all_hosp_adm_dict[id].gender for id in all_hosp_adm_dict]))
print("ethnicity:",
      list_occurrence_counter([all_hosp_adm_dict[id].ethnicity for id in all_hosp_adm_dict]))
print(f"avg age: {np.mean([all_hosp_adm_dict[id].age for id in all_hosp_adm_dict]):.0f}")

all_labels = [all_hosp_adm_dict[id].y for id in all_hosp_adm_dict]
valid_labels = [l for l in all_labels if l is not None]
print(f"Label missing ratio: {1 - len(valid_labels) / len(all_labels):.2f} "
      f"({len(all_labels) - len(valid_labels)} / {len(all_labels)})")
print(f"Label ratio: {list_occurrence_counter(valid_labels)}")

missing_x1 = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].x1 is None:
        missing_x1 += 1
print(f"x1 missing ratio: {missing_x1 / len(all_hosp_adm_dict):.2f}")

missing_x2 = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].x2 is None:
        missing_x2 += 1
print(f"x2 missing ratio: {missing_x2 / len(all_hosp_adm_dict):.2f}")

missing_x3 = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].x3 is None:
        missing_x3 += 1
print(f"x3 missing ratio: {missing_x3 / len(all_hosp_adm_dict):.2f}")

missing_any = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].x1 is None or \
            all_hosp_adm_dict[id].x2 is None or \
            all_hosp_adm_dict[id].x3 is None:
        missing_any += 1
print(f"missing any ratio: {missing_any / len(all_hosp_adm_dict):.2f}")

missing_all = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].x1 is None and \
            all_hosp_adm_dict[id].x2 is None and \
            all_hosp_adm_dict[id].x3 is None:
        missing_all += 1
print(f"missing all ratio: {missing_all / len(all_hosp_adm_dict):.2f}")

just_one = 0
for id in all_hosp_adm_dict:
    if (int(all_hosp_adm_dict[id].x1 is not None) + \
        int(all_hosp_adm_dict[id].x2 is not None) + \
        int(all_hosp_adm_dict[id].x3 is not None)) == 1:
        just_one += 1
print(f"just one ratio: {just_one / len(all_hosp_adm_dict):.2f}")
