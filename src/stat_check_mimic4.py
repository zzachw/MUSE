import os
from collections import Counter

import numpy as np

from src.utils import processed_data_path, load_pickle

ethnicity_map = {
    'BLACK/CARIBBEAN ISLAND': 'African American',
    'ASIAN': 'Asian',
    'HISPANIC/LATINO - SALVADORAN': 'Hispanic',
    'WHITE': 'Caucasian',
    'AMERICAN INDIAN/ALASKA NATIVE': 'Native American',
    'WHITE - EASTERN EUROPEAN': 'Caucasian',
    'WHITE - RUSSIAN': 'Caucasian',
    'OTHER': 'Other',
    'HISPANIC/LATINO - COLUMBIAN': 'Hispanic',
    'ASIAN - SOUTH EAST ASIAN': 'Asian',
    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 'Native American',
    'BLACK/CAPE VERDEAN': 'African American',
    'HISPANIC OR LATINO': 'Hispanic',
    'HISPANIC/LATINO - DOMINICAN': 'Hispanic',
    'HISPANIC/LATINO - PUERTO RICAN': 'Hispanic',
    'BLACK/AFRICAN': 'African American',
    'ASIAN - CHINESE': 'Asian',
    'HISPANIC/LATINO - HONDURAN': 'Hispanic',
    'ASIAN - KOREAN': 'Asian',
    'ASIAN - ASIAN INDIAN': 'Asian',
    'PATIENT DECLINED TO ANSWER': 'Other',
    'HISPANIC/LATINO - CENTRAL AMERICAN': 'Hispanic',
    'HISPANIC/LATINO - GUATEMALAN': 'Hispanic',
    'HISPANIC/LATINO - MEXICAN': 'Hispanic',
    'PORTUGUESE': 'Caucasian',
    'MULTIPLE RACE/ETHNICITY': 'Other',
    'WHITE - BRAZILIAN': 'Caucasian',
    'WHITE - OTHER EUROPEAN': 'Caucasian',
    'UNABLE TO OBTAIN': 'Other',
    'HISPANIC/LATINO - CUBAN': 'Hispanic',
    'SOUTH AMERICAN': 'Hispanic',
    'BLACK/AFRICAN AMERICAN': 'African American',
    'UNKNOWN': 'Other',
}


def list_occurrence_counter(lst):
    counter = Counter(lst)
    keys = sorted(counter.keys())
    message = []
    for k in keys:
        count = counter[k]
        percentage = count / len(lst)
        message.append(f"{k}: {percentage:.2f}")
    return "{" + ", ".join(message) + "}"


all_hosp_adm_dict = load_pickle(os.path.join(processed_data_path, "mimic4/hosp_adm_dict.pkl"))
print('# patients', len(set([v.patient_id for v in all_hosp_adm_dict.values()])))
print('# admissions', len(all_hosp_adm_dict))
print("gender:", list_occurrence_counter([all_hosp_adm_dict[id].gender for id in all_hosp_adm_dict]))
print("ethnicity:",
      list_occurrence_counter([ethnicity_map[all_hosp_adm_dict[id].ethnicity] for id in all_hosp_adm_dict]))
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
    if len(all_hosp_adm_dict[id].diagnoses_icd) == 0:
        missing_diagnoses_icd += 1
print(f"diagnoses_icd missing ratio: {missing_diagnoses_icd / len(all_hosp_adm_dict):.2f}")

missing_procedures_icd = 0
for id in all_hosp_adm_dict:
    if len(all_hosp_adm_dict[id].procedures_icd) == 0:
        missing_procedures_icd += 1
print(f"procedures_icd missing ratio: {missing_procedures_icd / len(all_hosp_adm_dict):.2f}")

missing_prescriptions = 0
for id in all_hosp_adm_dict:
    if len(all_hosp_adm_dict[id].prescriptions) == 0:
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

missing_discharge = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].discharge is None:
        missing_discharge += 1
print(f"discharge missing ratio: {missing_discharge / len(all_hosp_adm_dict):.2f}")

len_text = []
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].discharge is not None:
        len_text.append(len(all_hosp_adm_dict[id].discharge.split(" ")))
print(f"avg discharge length: {np.mean(len_text):.0f}")

missing_any = 0
for id in all_hosp_adm_dict:
    if all_hosp_adm_dict[id].discharge is None or \
            all_hosp_adm_dict[id].labvectors is None or \
            len(all_hosp_adm_dict[id].trajectory) == 0:
        missing_any += 1
print(f"missing any ratio: {missing_any / len(all_hosp_adm_dict):.2f}")

just_one = 0
for id in all_hosp_adm_dict:
    if (int(all_hosp_adm_dict[id].discharge is not None) + \
        int(all_hosp_adm_dict[id].labvectors is not None) + \
        int(len(all_hosp_adm_dict[id].trajectory) > 1)) == 1:
        just_one += 1
print(f"just one ratio: {just_one / len(all_hosp_adm_dict):.2f}")
