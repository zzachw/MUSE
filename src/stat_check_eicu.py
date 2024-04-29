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


all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "eicu/icu_stay_dict.pkl"))
print('# patients', len(set([v.patient_id for v in all_hosp_adm_dict.values()])))
print('# admissions', len(all_hosp_adm_dict))
print("gender:", list_occurrence_counter([all_hosp_adm_dict[id].gender for id in all_hosp_adm_dict]))
print("ethnicity:",
      list_occurrence_counter([all_hosp_adm_dict[id].ethnicity for id in all_hosp_adm_dict]))
print(f"avg age: {np.mean([all_hosp_adm_dict[id].age for id in all_hosp_adm_dict]):.0f}")

all_mortality_labels = [all_hosp_adm_dict[id].mortality for id in all_hosp_adm_dict]
valid_mortality_labels = [l for l in all_mortality_labels if l is not None]
print(f"Mortality missing ratio: {1 - len(valid_mortality_labels) / len(all_mortality_labels):.2f} "
      f"({len(all_mortality_labels) - len(valid_mortality_labels)} / {len(all_mortality_labels)})")
print(f"Mortality ratio: {np.sum(valid_mortality_labels) / len(valid_mortality_labels):.2f} "
      f"({np.sum(valid_mortality_labels)} / {len(valid_mortality_labels)})")

all_readmission_labels = [all_hosp_adm_dict[id].readmission for id in all_hosp_adm_dict]
valid_readmission_labels = [l for l in all_readmission_labels if l is not None]
print(f"Readmission missing ratio: {1 - len(valid_readmission_labels) / len(all_readmission_labels):.2f} "
      f"({len(all_readmission_labels) - len(valid_readmission_labels)} / {len(all_readmission_labels)})")
print(f"Readmission ratio: {np.sum(valid_readmission_labels) / len(valid_readmission_labels):.2f} "
      f"({np.sum(valid_readmission_labels)} / {len(valid_readmission_labels)})")

missing_diagnoses_icd = 0
for id in all_hosp_adm_dict:
    if len(all_hosp_adm_dict[id].diagnosis) == 0:
        missing_diagnoses_icd += 1
print(f"diagnoses_icd missing ratio: {missing_diagnoses_icd / len(all_hosp_adm_dict):.2f}")

missing_procedures_icd = 0
for id in all_hosp_adm_dict:
    if len(all_hosp_adm_dict[id].treatment) == 0:
        missing_procedures_icd += 1
print(f"procedures_icd missing ratio: {missing_procedures_icd / len(all_hosp_adm_dict):.2f}")

missing_prescriptions = 0
for id in all_hosp_adm_dict:
    if len(all_hosp_adm_dict[id].medication) == 0:
        missing_prescriptions += 1
print(f"prescriptions missing ratio: {missing_prescriptions / len(all_hosp_adm_dict):.2f}")

missing_trajectory = 0
for id in all_hosp_adm_dict:
    if len(all_hosp_adm_dict[id].trajectory) == 0:
        missing_trajectory += 1
print(f"trajectory missing ratio: {missing_trajectory / len(all_hosp_adm_dict):.2f}")

missing_labvectors = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].labvectors is None:
        missing_labvectors += 1
print(f"labvectors missing ratio: {missing_labvectors / len(all_hosp_adm_dict):.2f}")

missing_apacheapsvar = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].apacheapsvar is None:
        missing_apacheapsvar += 1
print(f"apacheapsvar missing ratio: {missing_apacheapsvar / len(all_hosp_adm_dict):.2f}")

missing_any = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].apacheapsvar is None or \
            all_hosp_adm_dict[id].labvectors is None or \
            len(all_hosp_adm_dict[id].trajectory) == 0:
        missing_any += 1
print(f"missing any ratio: {missing_any / len(all_hosp_adm_dict):.2f}")

just_one = 0
for id in all_hosp_adm_dict:
    if (int(all_hosp_adm_dict[id].apacheapsvar is not None) + \
        int(all_hosp_adm_dict[id].labvectors is not None) + \
        int(len(all_hosp_adm_dict[id].trajectory) > 1)) == 1:
        just_one += 1
print(f"just one ratio: {just_one / len(all_hosp_adm_dict):.2f}")
