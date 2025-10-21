# STIfuse: Predicting AKI Deterioration Robustly by STIfuse: Fusing Dynamic and Static Representations
This respository includes codes and datasets for paper "Predicting AKI Deterioration Robustly by STIfuse: Fusing Dynamic and Static Representations".

## Environment requirements

-python = 3.10.0
-torch = 2.5.1
-numpy = 1.24.3
-cuda_version = 12.6

## Data avalability

The preprocessed MIMIC-IV and MIMIC-III datasets supporting this study are available in a controlled-access repository at:
https://drive.google.com/drive/folders/1lR9simtpo37tTD3z4cf9hufyVUWG9eS7?usp=drive_link

## Pre-train and fine-tune

- **Pre-training**: Executed via `run_pretrain.py` using the **MIMIC-IV dataset**. This phase includes model training and in-process evaluation.  
- **Fine-tuning**: Executed via `run_ft.py` using the **MIMIC-III dataset**. Includes adaptation training and evaluation on the target domain.
