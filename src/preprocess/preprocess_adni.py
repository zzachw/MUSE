import os

import numpy as np

from src.dataset.data import ADNIData
from src.utils import dump_pickle, processed_data_path, read_json

data_path = os.path.join(processed_data_path, "adni")

y = os.path.join(data_path, "y.json")
x_DTIROI_04_30_14 = os.path.join(data_path, "x_DTIROI_04_30_14.json")
x_UCBERKELEYAV45_10_17_16 = os.path.join(data_path, "x_UCBERKELEYAV45_10_17_16.json")
x_UCSFFSL = os.path.join(data_path, "x_UCSFFSL.json")
demo = os.path.join(data_path, "demo.json")

y = read_json(y)
x_DTIROI_04_30_14 = read_json(x_DTIROI_04_30_14)
x_UCBERKELEYAV45_10_17_16 = read_json(x_UCBERKELEYAV45_10_17_16)
x_UCSFFSL = read_json(x_UCSFFSL)
demo = read_json(demo)

adni_data_dict = {}
for id, data in demo.items():
    adni_data_dict[id] = ADNIData(id, data["AGE_INT"], data["PTGENDER"], data["PTRACCAT"])

for id, data in x_DTIROI_04_30_14.items():
    adni_data_dict[id].x1 = np.array(data)

for id, data in x_UCBERKELEYAV45_10_17_16.items():
    adni_data_dict[id].x2 = np.array(data)

for id, data in x_UCSFFSL.items():
    adni_data_dict[id].x3 = np.array(data)

for id, data in y.items():
    adni_data_dict[id].y = data

valid_adni_data_dict = {}
for id, data in adni_data_dict.items():
    if data.x1 is not None or \
            data.x2 is not None or \
            data.x3 is not None:
        valid_adni_data_dict[id] = data

dump_pickle(valid_adni_data_dict, os.path.join(data_path, "adni_data_dict.pkl"))
