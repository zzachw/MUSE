import csv
import os
import sys

import numpy as np

from src.dataset.data import MIMIC4Data
from src.utils import processed_data_path, dump_pickle


def process_patients_admissions(infile, hosp_adm_dict):
    excluded_duration = 0
    excluded_age = 0
    excluded_mortality = 0
    inff = open(infile, "r")
    count = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()
        admission_id = line["hadm_id"]
        patient_id = line["subject_id"]
        duration = float(line["duration"])
        # exclude: duration > 10 days
        if duration > 10 * 24 * 60:
            excluded_duration += 1
            continue
        age = int(line["admit_age"])
        # exclude: cohort age < 18, > 89
        if (age < 18) or (age > 89):
            excluded_age += 1
            continue
        discharge_status = line["hospital_expire_flag"]
        assert discharge_status in ["0", "1"]
        if discharge_status == "1":
            excluded_mortality += 1
            continue
        if line["die_in_days"] != "":
            mortality = float(line["die_in_days"]) <= 90
        elif line["readmit_in_days"] != "" and float(line["readmit_in_days"]) > 90:
            # no mortality info, but the patient is readmitted after 90 days
            mortality = False
        else:
            mortality = None
        if line["readmit_in_days"] != "":
            readmission = float(line["readmit_in_days"]) <= 15
        else:
            readmission = None
        gender = line["gender"]
        ethnicity = line["race"]
        hosp_adm = MIMIC4Data(
            admission_id=admission_id,
            patient_id=patient_id,
            duration=duration,
            mortality=mortality,
            readmission=readmission,
            age=age,
            gender=gender,
            ethnicity=ethnicity,
        )
        if hosp_adm in hosp_adm_dict:
            print("Duplicate HADM ID!")
            sys.exit(0)
        hosp_adm_dict[admission_id] = hosp_adm
        count += 1
    inff.close()
    print("")
    print("hosp_adm excluded due to duration: %d" % excluded_duration)
    print("hosp_adm excluded due to age: %d" % excluded_age)
    print("hosp_adm excluded due to mortality: %d" % excluded_mortality)
    print("hosp_adm included: %d" % len(hosp_adm_dict))
    return hosp_adm_dict


def process_diagnoses_icd(infile, hosp_adm_dict):
    inff = open(infile, "r")
    count = 0
    missing_admission_id = 0
    out_of_hosp_adm = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        admission_id = line["hadm_id"]
        timestamp = 0
        id = line["long_title"].lower()

        if admission_id not in hosp_adm_dict:
            missing_admission_id += 1
            continue

        if timestamp > hosp_adm_dict[admission_id].duration:
            out_of_hosp_adm += 1
            continue

        hosp_adm_dict[admission_id].diagnoses_icd.append((timestamp, "diagnoses_icd", id))
        count += 1
    inff.close()

    print("")
    print("diagnoses_icd without admission ID: %d" % missing_admission_id)
    print("diagnoses_icd out of hosp_adm stay: %d" % out_of_hosp_adm)

    return hosp_adm_dict


def process_procedures_icd(infile, hosp_adm_dict):
    inff = open(infile, "r")
    count = 0
    missing_admission_id = 0
    out_of_hosp_adm = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        admission_id = line["hadm_id"]
        timestamp = int(float(line["timestamp"]))
        id = line["long_title"].lower()

        if admission_id not in hosp_adm_dict:
            missing_admission_id += 1
            continue

        if timestamp > hosp_adm_dict[admission_id].duration:
            out_of_hosp_adm += 1
            continue

        hosp_adm_dict[admission_id].procedures_icd.append((timestamp, "procedures_icd", id))
        count += 1
    inff.close()

    print("")
    print("procedures_icd without admission ID: %d" % missing_admission_id)
    print("procedures_icd out of hosp_adm stay: %d" % out_of_hosp_adm)

    return hosp_adm_dict


def process_prescriptions(infile, hosp_adm_dict):
    inff = open(infile, "r")
    count = 0
    missing_admission_id = 0
    out_of_hosp_adm = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        admission_id = line["hadm_id"]
        timestamp = int(float(line["timestamp"]))
        id = line["drug"].lower()

        if admission_id not in hosp_adm_dict:
            missing_admission_id += 1
            continue

        if timestamp > hosp_adm_dict[admission_id].duration:
            out_of_hosp_adm += 1
            continue

        hosp_adm_dict[admission_id].prescriptions.append((timestamp, "prescriptions", id))
        count += 1
    inff.close()

    print("")
    print("prescriptions without admission ID: %d" % missing_admission_id)
    print("prescriptions out of hosp_adm stay: %d" % out_of_hosp_adm)

    return hosp_adm_dict


def post_process_codes(hosp_adm_dict, max_len=50):
    max_cut = 0
    ret_hosp_adm_dict = {}
    for admission_id, hosp_adm in hosp_adm_dict.items():
        # merge codes
        merged = sorted(
            hosp_adm.procedures_icd +
            hosp_adm.prescriptions
        )
        # diagnoses_icd always comes first
        merged = hosp_adm.diagnoses_icd + merged
        types = [item[1] for item in merged]
        codes = [item[2] for item in merged]
        seq_len = len(merged)

        if seq_len > max_len:
            max_cut += 1
            types = types[-max_len:]
            codes = codes[-max_len:]

        hosp_adm.trajectory = (types, codes)

        ret_hosp_adm_dict[admission_id] = hosp_adm

    print("hosp_adm with max cut: %d" % max_cut)

    return ret_hosp_adm_dict


def process_labevents(infile, hosp_adm_dict):
    inff = open(infile, "r")
    count = 0
    missing_admission_id = 0
    out_of_hosp_adm = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        admission_id = str(int(float(line["hadm_id"])))
        timestamp = int(float(line["timestamp"]))
        itemid = line["itemid"]
        valuenum = float(line["normalized_valuenum"])
        flag = 1 if line["flag"] == "abnormal" else 0

        if admission_id not in hosp_adm_dict:
            missing_admission_id += 1
            continue

        if timestamp > hosp_adm_dict[admission_id].duration:
            out_of_hosp_adm += 1
            continue

        # ( timestamp in min (int), type (str), (item_id (str), value (float), flag (float)) )
        hosp_adm_dict[admission_id].labevents.append((timestamp, "labevents", (itemid, valuenum, flag)))
        count += 1

    inff.close()

    print("")
    print("labevents without admission ID: %d" % missing_admission_id)
    print("labevents out of hosp_adm stay: %d" % out_of_hosp_adm)

    return hosp_adm_dict


def post_process_labevents(hosp_adm_dict, max_len=50):
    itemid_to_index = {}
    for admission_id, hosp_adm in hosp_adm_dict.items():
        for timestamp, _, (itemid, valuenum, flag) in hosp_adm.labevents:
            if itemid not in itemid_to_index:
                itemid_to_index[itemid] = len(itemid_to_index) * 2

    # group by timestamp
    for admission_id, hosp_adm in hosp_adm_dict.items():
        # (timestamp, list of (item_id, value, flag))
        grouped_labevents = []
        for timestamp, _, (itemid, valuenum, flag) in hosp_adm.labevents:
            if len(grouped_labevents) == 0:
                grouped_labevents.append((timestamp, [(itemid, valuenum, flag)]))
            elif grouped_labevents[-1][0] == timestamp:
                grouped_labevents[-1][1].append((itemid, valuenum, flag))
            else:
                grouped_labevents.append((timestamp, [(itemid, valuenum, flag)]))
        hosp_adm.labevents = grouped_labevents

    # convert to numpy array
    max_cut = 0
    for admission_id, hosp_adm in hosp_adm_dict.items():
        vectors = []
        vector = np.zeros(len(itemid_to_index) * 2)
        for timestamp, labevents in hosp_adm.labevents:
            for itemid, valuenum, flag in labevents:
                vector[itemid_to_index[itemid]] = valuenum
                vector[itemid_to_index[itemid] + 1] = flag
            vectors.append(vector.copy())
        if len(vectors) > max_len:
            vectors = vectors[-max_len:]
            max_cut += 1

        if len(vectors) > 0:
            vectors = np.array(vectors)
            hosp_adm.labvectors = vectors

    print("hosp_adm with max cut: %d" % max_cut)

    return hosp_adm_dict


def process_discharge(infile, hosp_adm_dict):
    inff = open(infile, "r")
    count = 0
    missing_admission_id = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        admission_id = line["hadm_id"]
        text = line["text"]

        if admission_id not in hosp_adm_dict:
            missing_admission_id += 1
            continue

        hosp_adm_dict[admission_id].discharge = text
        count += 1
    inff.close()

    print("")
    print("discharge without admission ID: %d" % missing_admission_id)

    return hosp_adm_dict


def main():
    input_path = os.path.join(processed_data_path, "mimic4")
    output_path = os.path.join(processed_data_path, "mimic4")

    patients_admissions_file = input_path + "/patients_admissions_tmp.csv"
    diagnoses_icd_file = input_path + "/diagnoses_icd_tmp.csv"
    procedures_icd_tmp_file = input_path + "/procedures_icd_tmp.csv"
    prescriptions_file = input_path + "/prescriptions_tmp.csv"
    labevents_file = input_path + "/labevents_tmp.csv"
    discharge_file = input_path + "/discharge_tmp.csv"

    hosp_adm_dict = {}
    print("Processing patient.csv")
    hosp_adm_dict = process_patients_admissions(patients_admissions_file, hosp_adm_dict)

    print("Processing diagnoses_icd_tmp.csv")
    hosp_adm_dict = process_diagnoses_icd(diagnoses_icd_file, hosp_adm_dict)
    print("Processing procedures_icd_tmp.csv")
    hosp_adm_dict = process_procedures_icd(procedures_icd_tmp_file, hosp_adm_dict)
    print("Processing prescriptions_tmp.csv")
    hosp_adm_dict = process_prescriptions(prescriptions_file, hosp_adm_dict)
    hosp_adm_dict = post_process_codes(hosp_adm_dict)
    print("Processing labevents_tmp.csv")
    hosp_adm_dict = process_labevents(labevents_file, hosp_adm_dict)
    hosp_adm_dict = post_process_labevents(hosp_adm_dict)
    print("Processing discharge_tmp.csv")
    hosp_adm_dict = process_discharge(discharge_file, hosp_adm_dict)

    dump_pickle(hosp_adm_dict, os.path.join(output_path, "hosp_adm_dict.pkl"))


if __name__ == "__main__":
    main()
