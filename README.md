# Malware Detection with Stacking Ensemble Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)  
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

---

## Table of Contents

- [Overview](#overview)  
- [Authors](#authors)  
- [Abstract](#abstract)  
- [Dataset](#dataset)  
  - [CIC MalMem 2022](#cic-malmem-2022)  
  - [Preprocessing](#preprocessing)  
- [Model Architectures](#model-architectures)  
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)  
  - [Random Forest](#random-forest)  
  - [XGBoost](#xgboost)  
  - [LightGBM](#lightgbm)  
- [Stacking Ensemble](#stacking-ensemble)  
- [Results](#results)  
- [Conclusion](#conclusion)  
- [Future Work](#future-work)  
- [Installation & Usage](#installation--usage)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  
- [References](#references)

---

## Overview

This repository implements a **stacking ensemble** for memory-based malware detection. By combining diverse classifiers—MLP, Random Forest, XGBoost, and LightGBM—our model leverages complementary strengths to improve overall detection accuracy and robustness against novel and adversarial malware samples.

---

## Authors

- **Gavin K. Luo**  
- **Zain J. Saquer**  

Central High School, 423 E Central St, Springfield, MO 65802

---

## Abstract

Malware detection is increasingly challenging due to polymorphism, encryption, and memory-based execution techniques. Traditional signature-based systems struggle to keep up. We propose a **stacking ensemble** that integrates:

1. Multi-Layer Perceptron (MLP)  
2. Random Forest  
3. XGBoost  
4. LightGBM  

on the **CIC MalMem 2022** dataset (memory-based behavioral features). After preprocessing (SMOTE, feature selection, z-score normalization), our ensemble achieves **87.95% accuracy**, **87.94% F1-score**, **87.94% precision**, and **87.95% recall**, outperforming any standalone classifier.

---

## Dataset

### CIC MalMem 2022

- **Source:** Canadian Institute for Cybersecurity  
- **URL:** https://www.unb.ca/cic/datasets/malmem-2022.html  
- **Samples before preprocessing:** 58,597  
- **Samples after preprocessing:** 93,536  
- **Features:** 52 (numerical + categorical)  
- **Classes:** Benign, Ransomware, Spyware, Trojan

### Preprocessing

1. **Data Cleaning**  
   - Removed duplicates  
   - Dropped three single-value categorical columns  

2. **Label Encoding**  
   - `'Benign': 0, 'Ransomware': 1, 'Spyware': 2, 'Trojan': 3`  

3. **Balancing**  
   - Applied **SMOTE** → 23,384 samples per class  

4. **Feature Selection**  
   - Retained top features via **Information Gain**  

5. **Scaling**  
   - Z-score normalization:  
     \  
     \(x'_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j}\)

---

## Model Architectures

### Multi-Layer Perceptron (MLP)

- **Layers:** 512 → 256 → 128 → 64 → 4  
- **Activations:** ReLU (hidden), Softmax (output)  
- **Regularization:** L2, Dropout  
- **Optimizer:** Adam + Learning-rate scheduler  

### Random Forest

- **Hyperparameter Tuning:** Optuna (TPE sampler)  
- **Best Params:**  
  ```json
  {
    "n_estimators": 225,
    "max_depth": 25,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "bootstrap": false
  }
