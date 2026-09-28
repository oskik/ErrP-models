# EEG Error-Related Potential (ErrP) Classification: Deep Learning and Riemannian Geometry

This project implements and evaluates Brain-Computer Interface (BCI) classification models for detecting Error-Related Potentials (ErrP) using Electroencephalography (EEG) data. The analysis utilizes the MOABB framework's BNCI2015_013 dataset and Kaggle BCI Challenge @ NER 2015 to classify 'Target' (class 0) and 'NonTarget' (class 1) events across six subjects for MOQBB and sixteen subjects for NER. This project benchmarks a Deep Convolutional Neural Network (`EEGNet`) and Riemannian Geometry machine learning pipeline (`xDAWN` + `Tangent Space`) across those two public datasets in both Within-Subject and Cross-Subject scenarios. 

## Preprocessing Pipelines
Preprocessing is tailored to the specific needs of the underlying classifiers using `mne-python` and `braindecode`.

**Deep Learning (EEGNet) Preprocessing**
*   Extracts strictly EEG channels and scales data to microvolts.
*   Applies Common Average Reference (CAR).
*   Utilizes a zero-phase FIR bandpass filter between 1.0 Hz and 30.0 Hz.
*   Resamples signals to 64 Hz and applies exponential moving standardization.
*   Extracts continuous event windows from -0.2s to 0.8s (MOABB) or from -0.2s to 0.7s (NER).

**Riemannian Geometry Preprocessing**
*   Applies a narrower FIR bandpass filter (1.0 to 10.0 Hz for MOABB, 1.0 to 12.0 Hz for NER).
*   Applies Common Average Reference (CAR).
*   Extracts epochs (-0.2s to 0.8s/0.7s) and decimates the signal by a factor of 4.

## Model Architectures
### 1. PyTorch & Braindecode (EEGNet)
*   Implements the `EEGNet` architecture natively designed for BCI applications.
*   Optimized using `AdamW` and a `CosineAnnealingLR` learning rate scheduler over 250 epochs.
*   Accounts for high class imbalance by utilizing weighted `CrossEntropyLoss` computed dynamically from the training set.
*   Employs Early Stopping based on validation loss to prevent overfitting.

### 2. PyRiemann & Scikit-Learn
*   Extracts spatial features using `XdawnCovariances` optimized for event-related potentials.
*   Projects covariance matrices into a Euclidean tangent space using `TangentSpace(metric='riemann')`.
*   Classifies features using either a Linear Support Vector Classifier (SVC) with balanced weights or Logistic Regression utilizing an ElasticNet penalty (50% L1, 50% L2).

## Project Main Structure

| File Name | Description |
| :--- | :--- |
| `process_data_for_eegnet_moabb.py` | Preprocesses the BNCI2015_013 dataset for Deep Learning and saves native `.fif` window datasets. |
| `process_data_for_eegnet_ner.py` | Preprocesses Kaggle NER CSV files for Deep Learning and aggregates them into a `BaseConcatDataset`. |
| `process_data_for_riemann_moabb.py` | Preprocesses BNCI2015_013 and exports continuous `.npz` arrays for Riemannian feature extraction. |
| `process_data_for_riemann_ner.py` | Preprocesses Kaggle NER raw CSV files, synchronizes stimulus events, and exports to `.npz` format. |
| `eegnet_moabb.ipynb` | Trains and evaluates EEGNet on MOABB data, generating learning curves, cross-subject metrics, and saving CSV results. |
| `eegnet_ner.ipynb` | Evaluates EEGNet on NER data, utilizing chronological train/validation splits, and extracts spatial convolution weights. |
| `riemann_moabb.ipynb` | Fits the Riemannian SVM pipeline, computes ROC-AUC curves, and executes grid searches for optimal time windows and hyperparameters. |
| `riemann_ner.ipynb` | Fits the Riemannian Logistic Regression pipeline on NER data, executes time-series cross-validation, and performs Wilcoxon statistical significance testing. |

## Getting Started

### Requirements
The core dependencies required to run the preprocessing and classification pipelines include:
*   `mne`, `moabb`, `braindecode`
*   `torch`, `skorch`
*   `pyriemann`, `scikit-learn`
*   `numpy`, `pandas`, `matplotlib`, `seaborn`

### Execution Order
1.  **Data Preprocessing:** Run the relevant `process_data_for_*.py` script to generate the preprocessed datasets and export them to your local directory (`.fif` for Deep Learning, `.npz` for Riemannian).
2.  **Model Training & Evaluation:** Execute the corresponding model script (e.g., `eegnetmoabb.py` or `riemannner.py`) to train the models. The scripts evaluate performance chronologically to simulate online BCI steering and automatically generate evaluation plots, confusion matrices, and `.csv` metric reports.
