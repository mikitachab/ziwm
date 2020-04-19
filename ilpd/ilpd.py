import os
import subprocess

import pandas


from .constants import (
    DATASET_CSV_FILENAME,
    DATASET_DIR,
    DATASET_URL,
    GENDER_MAP,
    ILPD_COLUMNS,
)


def get_data(filename=DATASET_CSV_FILENAME, normalized=False):
    ilpd = pandas.read_csv(filename)
    ilpd.columns = ILPD_COLUMNS
    if normalized:
        return normalize_data(ilpd)
    return ilpd


def normalize_data(data):
    normalized_data = data.copy()
    normalized_data['Gender'] = normalized_data['Gender'].map(GENDER_MAP)
    normalized_data = normalized_data.dropna()
    return normalized_data


def prepare_data(dirname=DATASET_DIR, filename=DATASET_CSV_FILENAME):
    os.makedirs(dirname, exist_ok=True)

    wget_cmd = [
        'wget',
        f'--output-document={filename}',
        DATASET_URL
    ]

    subprocess.run(wget_cmd, check=True)
