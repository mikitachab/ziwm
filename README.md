# ziwm

## Prerequisites

- Python 3.7
- wget

## Dataset

[ILPD (Indian Liver Patient Dataset) Data Set](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))

## Setup

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

- Download and prepare dataset
  
```shell
./main.py prepare-data
```

- Make features ranking

print ranking

```shell
./main.py features-ranking
```

or print rankning as latex table

```shell
./main.py features-ranking --latex
```
