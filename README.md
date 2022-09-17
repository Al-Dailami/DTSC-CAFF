# DTSC-CAFF

This repo hosts the source code scripts for training and testing DTSC-CAFF.

## Requirements
```
numpy==1.18.1
pandas==0.24.2
scipy==1.4.1
torch==1.5.0
scikit-learn==0.20.2
```

## Datasets

We use [eICU](https://physionet.org/content/eicu-crd/2.0/) and [MIMIC-IV](https://mimic.mit.edu/) datasets. We refer users to the link because MIMIC-IV and eICU datasets requires the CITI training program in order to use it. 

## Data extraction and preprocessing:
We Follow the preprocessing pipline of [TPC-LoS prediction](https://dl.acm.org/doi/10.1145/3450439.3451860).
### Mimic Preprocessing:
We refer the user to [MIMIC_preprocessing](https://github.com/EmmaRocheteau/TPC-LoS-prediction/tree/master/MIMIC_preprocessing)
### eICU Preprocessing:
We refer the user to [eICU_preprocessing](https://github.com/EmmaRocheteau/TPC-LoS-prediction/tree/master/eICU_preprocessing)

## Model training, validation and testting

Run the script in the train_val_test_tdsc-caff.ipynb file.

## Pre-trained model weight files

Can be downloaded via this [Link](https://pan.baidu.com/s/1OYvziRJo0GZ1aYmjjRUlGQ?pwd=wg72)

# References
E. Rocheteau, P. Lio, and S. Hyland, “Temporal pointwise convolutional` networks for length of stay prediction in the intensive care unit,” in ACM CHIL 2021 - Proceedings of the Conference on Health, Inference, and Learning, 2021, pp. 58–68.

