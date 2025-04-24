# Jame and Gav 2025 OSEF Project# OSEF-2025

Malware Detection with Ensemble Learning
Project Overview
This project presents a stacking ensemble approach for robust malware classification using machine learning. The model combines four powerful classifiers: Multi-Layer Perceptrons (MLP), XGBoost, LightGBM, and Random Forests. The goal is to improve the detection of malware through diverse machine learning models by leveraging their complementary strengths. Our approach is evaluated on the CIC MalMem 2022 dataset, which includes memory-based behavioral features from malware and benign samples.

Authors
Gavin K. Luo

Zain J. Saquer

Central High School, Springfield, MO 65802

Abstract
Malware detection has become a critical concern in cybersecurity, especially as new variants emerge and evade traditional signature-based detection systems. This research investigates the effectiveness of stacking ensemble methods in improving classification accuracy and robustness. By combining the strengths of multiple algorithms, the ensemble achieves superior performance over individual models in detecting memory-based malware. The framework demonstrates a high classification accuracy of 87.95% with an F1-score of 87.94%. This research emphasizes the importance of feature diversity and ensemble techniques in creating resilient cybersecurity defenses.

Dataset
The CIC MalMem 2022 dataset is used in this project. It contains dynamic behavioral features extracted from system memory during the execution of malware and benign software. The dataset includes:

58,597 samples initially (before preprocessing)

93,536 samples after preprocessing

52 features (numerical and categorical)

Malware categories: Benign, Ransomware, Spyware, Trojan

Preprocessing
SMOTE (Synthetic Minority Over-sampling Technique) was used to balance class distribution.

Features were scaled using Z-score normalization.

Information Gain was used for feature selection.

Model Architecture
1. Multi-Layer Perceptrons (MLP)
MLPs are neural networks that excel in capturing complex, non-linear patterns. The architecture consists of:

Hidden layers: 512, 256, 128, 64, 4 neurons

ReLU activation for hidden layers and Softmax for the output layer

L2 Regularization, Dropout layers, and Learning rate scheduler for improved generalization

2. Random Forest
A decision-tree-based algorithm that uses an ensemble of trees for robust predictions. Optimized using Optuna for hyperparameter tuning.

3. XGBoost
A gradient-boosted decision tree algorithm. We used Optuna for hyperparameter tuning, resulting in optimal settings for n_estimators, max_depth, and learning_rate.

4. LightGBM
A gradient-boosting framework optimized through Optuna and Bayesian optimization techniques for better model performance.

Stacking Ensemble
The models are combined using a stacking technique, where predictions from individual models serve as input features to a meta-classifier. We experimented with weighted averaging and logistic regression as combination strategies.

Results

Model | Accuracy | F1-Score | Precision | Recall
Ensemble | 87.95% | 87.94% | 87.94% | 87.95%
XGBoost | 87.76% | 87.75% | 87.74% | 87.76%
Random Forest | 87.05% | 87.04% | 87.04% | 70.50%
LightGBM | 87.90% | 87.89% | 87.88% | 87.90%
MLP | 85.09% | 85.06% | 85.21% | 85.09%
The ensemble model outperformed all individual models, showcasing its effectiveness in improving malware classification accuracy while maintaining high recall and F1-scores.

Conclusion
The stacking ensemble model significantly enhances malware detection capabilities by combining diverse model architectures. The performance improvements, especially in classifying novel malware variants and resisting adversarial attacks, validate the use of ensemble learning in cybersecurity applications.

Future Research
Bayesian optimization for fine-tuning neural network parameters.

Exploring alternative ensemble methods such as bagging and boosting.

Investigating multi-modal learning using additional data sources (e.g., network traffic, static features).


