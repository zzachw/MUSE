import csv
import os
import sys

import math
import numpy as np

from src.dataset.data import eICUData
from src.utils import create_directory, dump_pickle, raw_data_path, processed_data_path

APACHEAPSVAR = [
    "intubated",
    "vent",
    "dialysis",
    "eyes",
    "motor",
    "verbal",
    "meds",
    "urine",
    "wbc",
    "temperature",
    "respiratoryrate",
    "sodium",
    "heartrate",
    "meanbp",
    "ph",
    "hematocrit",
    "creatinine",
    "albumin",
    "pao2",
    "pco2",
    "bun",
    "glucose",
    "bilirubin",
    "fio2",
]

APACHEAPSVAR_NCAT = {
    "eyes": 4,
    "motor": 6,
    "verbal": 5,
}


def process_patient(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    admission_dict = {}
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        icu_timestamp = -int(line["hospitaladmitoffset"])  # w.r.t. hospital admission
        admission_id = line["patienthealthsystemstayid"]

        if admission_id not in admission_dict:
            admission_dict[admission_id] = []
        admission_dict[admission_id].append((icu_timestamp, icu_id))

    inff.close()
    print("")

    admission_dict_sorted = {}
    for admission_id, time_icu_tuples in admission_dict.items():
        admission_dict_sorted[admission_id] = sorted(time_icu_tuples)

    icu_readmission_dict = {}
    next_icu_id_dict = {}
    for admission_id, time_icu_tuples in admission_dict_sorted.items():
        for idx, time_icu_tuple in enumerate(time_icu_tuples[:-1]):
            curr_icu_timestamp = time_icu_tuple[0]
            curr_icu_id = time_icu_tuple[1]
            next_icu_timestamp = time_icu_tuples[idx + 1][0]
            next_icu_id = time_icu_tuples[idx + 1][1]
            if next_icu_timestamp - curr_icu_timestamp <= 15 * 24 * 60:
                # re-admitted to ICU within 15 days
                icu_readmission_dict[curr_icu_id] = True
                next_icu_id_dict[curr_icu_id] = next_icu_id
            else:
                icu_readmission_dict[curr_icu_id] = False
                next_icu_id_dict[curr_icu_id] = ""
        last_icu_id = time_icu_tuples[-1][1]
        icu_readmission_dict[last_icu_id] = False
        next_icu_id_dict[last_icu_id] = ""

    excluded_icu_duration = 0
    excluded_age = 0

    inff = open(infile, "r")
    count = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        admission_id = line["patienthealthsystemstayid"]
        patient_id = line["uniquepid"]
        icu_duration = int(line["unitdischargeoffset"])
        # exclude: icu duration > 10 days or < 12 hours
        if (icu_duration > 10 * 24 * 60) or (icu_duration < 12 * 60):
            excluded_icu_duration += 1
            continue
        hospital_id = line["hospitalid"]
        age = line["age"]
        try:
            age = int(age)
        # exclude: cohort age < 18, > 89, or unknown
        except ValueError:
            assert age in ["> 89", ""]
            excluded_age += 1
            continue
        if age < 18:
            excluded_age += 1
            continue
        discharge_status = line["unitdischargestatus"]
        mortality = True if discharge_status == "Expired" else False
        readmission = icu_readmission_dict[icu_id]
        gender = line["gender"]
        ethnicity = line["ethnicity"]

        icu_stay = eICUData(
            icu_id=icu_id,
            admission_id=admission_id,
            patient_id=patient_id,
            icu_duration=icu_duration,
            hospital_id=hospital_id,
            mortality=mortality,
            readmission=readmission,
            age=age,
            gender=gender,
            ethnicity=ethnicity
        )

        if icu_stay in icu_stay_dict:
            print("Duplicate ICU ID!")
            sys.exit(0)
        icu_stay_dict[icu_id] = icu_stay

        count += 1
    inff.close()
    print("")
    print("ICU stays excluded due to icu duration: %d" % excluded_icu_duration)
    print("ICU stays excluded due to age: %d" % excluded_age)
    print("ICU stays included: %d" % len(icu_stay_dict))

    return icu_stay_dict


def process_diagnosis(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["diagnosisoffset"])
        id = line["diagnosisstring"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].diagnosis.append((timestamp, "diagnosis", id))
        count += 1
    inff.close()

    print("")
    print("diagnosis without ICU ID: %d" % missing_icu_id)
    print("diagnosis out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_diagnosis(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.diagnosis = sorted(icu_stay.diagnosis)
    return icu_stay_dict


def process_treatment(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["treatmentoffset"])
        id = line["treatmentstring"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].treatment.append((timestamp, "treatment", id))
        count += 1
    inff.close()

    print("")
    print("treatment without ICU ID: %d" % missing_icu_id)
    print("treatment out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_treatment(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.treatment = sorted(icu_stay.treatment)
    return icu_stay_dict


def process_medication(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["drugstartoffset"])
        id = line["drugname"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].medication.append((timestamp, "medication", id))
        count += 1
    inff.close()

    print("")
    print("medication without ICU ID: %d" % missing_icu_id)
    print("medication out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_medication(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.medication = sorted(icu_stay.medication)
    return icu_stay_dict


def post_process_codes(icu_stay_dict, max_len=50):
    max_cut = 0
    ret_icu_stay_dict = {}
    for icu_id, icu_stay in icu_stay_dict.items():
        # merge codes
        merged = sorted(
            icu_stay.diagnosis +
            icu_stay.treatment +
            icu_stay.medication
        )
        timestamps = [math.ceil((item[0] + 1e-6) / 60) for item in merged]  # min -> hour
        timestamps = [t if t > 0 else 1 for t in timestamps]  # resolve negative timestamp
        types = [item[1] for item in merged]
        codes = [item[2] for item in merged]

        # make prediction at 12 hours
        num_valid_codes = sum([1 for t in timestamps if t <= 12])
        types = types[:num_valid_codes]
        codes = codes[:num_valid_codes]
        if num_valid_codes > max_len:
            max_cut += 1
            types = types[-max_len:]
            codes = codes[-max_len:]

        icu_stay.trajectory = (types, codes)

        ret_icu_stay_dict[icu_id] = icu_stay

    print("ICU stays with max cut: %d" % max_cut)

    return ret_icu_stay_dict


def process_lab(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["labresultoffset"])
        id = line["labname"]
        value = float(line["normalized_labresult"])

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].lab.append((timestamp, "lab", (id, value)))
        count += 1

    inff.close()

    print("")
    print("lab without ICU ID: %d" % missing_icu_id)
    print("lab out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_lab(icu_stay_dict, max_len=50):
    # sort
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.lab = sorted(icu_stay.lab)

    itemid_to_index = {}
    for icu_id, icu_stay in icu_stay_dict.items():
        for timestamp, _, (itemid, valuenum) in icu_stay.lab:
            if itemid not in itemid_to_index:
                itemid_to_index[itemid] = len(itemid_to_index)

    # group by timestamp
    for icu_id, icu_stay in icu_stay_dict.items():
        # (timestamp, list of (item_id, value))
        grouped_lab = []
        for timestamp, _, (itemid, valuenum) in icu_stay.lab:
            # make prediction at 12 hours
            if timestamp > 12 * 60:
                continue
            if len(grouped_lab) == 0:
                grouped_lab.append((timestamp, [(itemid, valuenum)]))
            elif grouped_lab[-1][0] == timestamp:
                grouped_lab[-1][1].append((itemid, valuenum))
            else:
                grouped_lab.append((timestamp, [(itemid, valuenum)]))
        icu_stay.lab = grouped_lab

    # convert to numpy array
    max_cut = 0
    for icu_id, icu_stay in icu_stay_dict.items():
        vectors = []
        vector = np.zeros(len(itemid_to_index))
        for timestamp, labevents in icu_stay.lab:
            for itemid, valuenum in labevents:
                vector[itemid_to_index[itemid]] = valuenum
            vectors.append(vector.copy())
        if len(vectors) > max_len:
            vectors = vectors[-max_len:]
            max_cut += 1

        if len(vectors) > 0:
            vectors = np.array(vectors)
            icu_stay.labvectors = vectors

    print("hosp_adm with max cut: %d" % max_cut)

    return icu_stay_dict


def process_apacheapsvar(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        apacheapsvar = []
        for var in APACHEAPSVAR:
            value = float(line[var])
            if var in APACHEAPSVAR_NCAT:
                ncat = APACHEAPSVAR_NCAT[var]
                cat = int(value) - 1
                vec = [0.0] * ncat
                vec[cat] = 1.0
                apacheapsvar.extend(vec)
            else:
                apacheapsvar.append(value)
        apacheapsvar = np.array(apacheapsvar)

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        icu_stay_dict[icu_id].apacheapsvar = apacheapsvar
        count += 1
    inff.close()

    print("")
    print("apacheapsvar without ICU ID: %d" % missing_icu_id)

    return icu_stay_dict


def main():
    input_path = os.path.join(raw_data_path, "physionet.org/files/eicu-crd/2.0")
    output_path = os.path.join(processed_data_path, "eicu")
    create_directory(output_path)

    patient_file = input_path + "/patient.csv"
    diagnosis_file = input_path + "/diagnosis.csv"
    treatment_file = input_path + "/treatment.csv"
    medication_file = input_path + "/medication.csv"
    lab_file = output_path + "/lab_tmp.csv"
    apacheapsvar_file = output_path + "/apacheapsvar_tmp.csv"

    icu_stay_dict = {}
    print("Processing patient.csv")
    icu_stay_dict = process_patient(patient_file, icu_stay_dict)

    print("Processing diagnosis.csv")
    icu_stay_dict = process_diagnosis(diagnosis_file, icu_stay_dict)
    print("Post-processing diagnosis")
    icu_stay_dict = post_process_diagnosis(icu_stay_dict)
    print("Processing treatment.csv")
    icu_stay_dict = process_treatment(treatment_file, icu_stay_dict)
    print("Post-processing treatment")
    icu_stay_dict = post_process_treatment(icu_stay_dict)
    print("Processing medication.csv")
    icu_stay_dict = process_medication(medication_file, icu_stay_dict)
    print("Post-processing medication")
    icu_stay_dict = post_process_medication(icu_stay_dict)
    icu_stay_dict = post_process_codes(icu_stay_dict)
    print("Processing lab.csv")
    icu_stay_dict = process_lab(lab_file, icu_stay_dict)
    print("Post-processing lab")
    icu_stay_dict = post_process_lab(icu_stay_dict)
    print("Processing apacheapsvar.csv")
    icu_stay_dict = process_apacheapsvar(apacheapsvar_file, icu_stay_dict)

    dump_pickle(icu_stay_dict, os.path.join(output_path, "icu_stay_dict.pkl"))


if __name__ == "__main__":
    main()
