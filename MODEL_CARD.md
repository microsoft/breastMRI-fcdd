---
# Model Card for breast-MRI-fcdd
---

# Model Card for breastMRI-fcdd

This is an implementation of the models described in the publication: *"Cancer detection in breast MRI screening via explainable artificial intelligence anomaly detection"*.

## Model Details

This model detects anomalies in breast MRI scans. It is based on the FCDD architecture, which is designed for unsupervised anomaly detection in images.


### Model Description

<!-- Provide a longer summary of what this model is. -->


- **Developed by:**
Felipe Oviedo, Anum S. Kazerouni, Philipp Liznerski, Yixi Xu, Michael Hirano, Robert A. Vandermeulen, Marius Kloft, Elyse Blum, Adam M Alessio, Christopher I. Li, William B Weeks, Rahul Dodhia, Juan Lavista Ferres, Habib Rahbar, Savannah C. Partridge

- **Model type:** 
Convolutional Neural Network

- **License:**
    MIT


## Uses


### Direct Use

This model is inteded for cancer detection in 2D MIPs from breast MRI scans, pre-processed as described in the publication, focused on highly imbalanced data. This model as-is is not intended for use in clinical settings.

### Out-of-Scope Use

*Disclaimer:* This code, model and sample data are intended for research and model development exploration. The models, code and examples are not designed or intended to be deployed in clinical settings as-is nor for use in the diagnosis or treatment of any health or medical condition, and the individual models’ performances for such purposes have not been established. You bear sole responsibility and liability for any use of the models, code and examples, including verification of outputs and incorporation into any product or service intended for a medical purpose or to inform clinical decision-making, compliance with applicable healthcare laws and regulations, and obtaining any necessary clearances or approvals.

## Bias, Risks, and Limitations

See Disclaimer / Out-of-Scope Use, and Discussion in the publication.

## How to Get Started with the Model

See the [README.md](README.md) file for instructions on how to get started with the model.

## Training Details

### Training Data

Private dataset as described in the publication.

### Training Procedure

Patient grouped cross validation for balanced and imbalanced data, with two classes Benign and Malignant, as described in the publication.

#### Training Hyperparameters

See default hyperparameters in the publication.

## Evaluation

Evaluated with 5-fold cross validation with balanced and imbalanced data, as described in the publication.

Also evaluated on an internal, independent dataset, and the multi-center [ACRIN-6698](https://www.cancerimagingarchive.net/collection/acrin-6698/) dataset.

### Testing Data, Factors & Metrics

AUC and AUROC for balanced and imbalanced data, along with pixel-wise AUC for spatial model interpretability.


## Environmental Impact

Negligible

### Model Architecture and Objective

Architecture and objectives are adapted from the original FCDD model (https://github.com/liznerski/fcdd)


## Model Card Contact

Felipe Oviedo - Microsoft AI for Good Lab