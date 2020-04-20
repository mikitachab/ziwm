import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DATASET_CSV_FILENAME = os.path.join(DATASET_DIR, 'ILPD.csv')

GENDER_MAP = {
    'Female': 0,
    'Male': 1
}

ILPD_COLUMNS = [
    'Age',
    'Gender',
    'TB',
    'DB',
    'Alkphos',
    'Sgpt',
    'Sgot',
    'TP',
    'ALB',
    'A/G',
    'Selector',
]

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv' # noqa
