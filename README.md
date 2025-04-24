# Robust Malware Classification via Stacking Ensemble

This project explores a robust approach to malware classification by employing a stacking ensemble of diverse machine learning models: XGBoost, LightGBM, Random Forest, and a Multi-Layer Perceptron (MLP). The goal is to leverage the complementary strengths of these models to achieve high accuracy and generalization in detecting and categorizing malware based on memory-based behavioral features.

## Dataset

The core of this research utilizes the **CIC MalMem 2024 OSEF dataset**. This dataset provides a rich collection of dynamic behavioral features extracted from system memory during the execution of both benign and malicious software samples. Key characteristics of the dataset include:

* **Dynamic Behavioral Features:** Capturing runtime activities such as process interactions, service enumeration, handle allocations, and module load characteristics. These 52 features are crucial for detecting modern malware employing evasion techniques.
* **Multi-Class Classification:** The dataset, after preprocessing, is structured for multi-class classification, distinguishing between **Benign** software and three distinct malware families: **Ransomware**, **Spyware**, and **Trojan**. These categories were numerically encoded as follows: `{'Benign': 0, 'Ransomware': 1, 'Spyware': 2, 'Trojan': 3}`.
* **Class Distribution Handling:** The initial class distribution was imbalanced. For example, the value counts before applying SMOTE showed varying numbers of samples per category. To address this, the **Synthetic Minority Over-sampling Technique (SMOTE)** was applied, resulting in a balanced training dataset.
* **Feature Preprocessing:** The data underwent several preprocessing steps:
    * **Removal of Duplicates:** Ensuring data integrity.
    * **Constant Feature Removal:** Eliminating 1 feature where all values were the same, reducing the feature count from the original dataset.
    * **Train-Test Split:** Dividing the data into training (80%) and testing (20%) sets, with stratification to maintain class proportions. The training set was further resampled using SMOTE.
    * **Feature Scaling:** Applying `StandardScaler` to normalize the numerical features in both the training and testing sets.

## Methodology

The project employs a stacking ensemble, a meta-learning technique that combines the predictions of multiple base models to form a final prediction. The key steps in our methodology include:

1.  **Individual Model Training:** Four distinct machine learning models were trained on the preprocessed training data:
    * **XGBoost:** Optimized using Optuna, with the best parameters found including `n_estimators`, `max_depth`, and `learning_rate`.
    * **LightGBM:** Also optimized using Optuna, with best parameters identified for `n_estimators`, `max_depth`, `learning_rate`, and `num_leaves`.
    * **Random Forest:** Hyperparameters tuned using Optuna with class balancing, with key parameters like `n_estimators` and `max_depth` optimized.
    * **Multi-Layer Perceptron (MLP):** A deep learning model built with TensorFlow/Keras, incorporating layers with neuron counts [512, 256, 128, 64, 4], ReLU activation (except for the final Softmax layer), L2 regularization, dropout (with rates like 0.05 and 0.1), and an Adam optimizer with an exponential decay learning rate schedule. Early stopping with a patience of 15 epochs was used during training.

2.  **Hyperparameter Optimization:** Optuna was used with the Tree-structured Parzen Estimator (TPE) sampler to efficiently search the hyperparameter space for the tree-based models through 3-fold cross-validation, optimizing for accuracy.

3.  **MLP Architecture:** The MLP model was designed with multiple dense layers, layer normalization, and dropout to prevent overfitting. The output layer used a Softmax activation to provide probability distributions over the four malware classes.

4.  **Stacking Ensemble Construction:** The predictions (specifically, predicted probabilities using `stack_method="predict_proba"`) from the trained base models (XGBoost, LightGBM, Random Forest, and a Keras-wrapped MLP) were used as input features for a `LogisticRegression` meta-classifier.

5.  **Model Evaluation:** The performance of each individual model and the final stacking ensemble was evaluated on the held-out test set using standard classification metrics: accuracy, precision (weighted), recall (weighted), and F1-score (weighted).

6.  **Feature Importance Analysis:** Feature importance scores were extracted from the tree-based models. For the MLP, permutation importance was calculated to estimate the impact of each feature on the model's accuracy. The top 10 most important features for each model were visualized.

## Results

The performance of the individual models and the stacking ensemble on the test set is summarized below:

| Model             | Accuracy | Precision | Recall   | F1-score |
| :---------------- | :------- | :-------- | :------- | :------- |
| XGBoost           | 0.8776   | 0.8774    | 0.8776   | 0.8775   |
| Random Forest     | 0.8705   | 0.8704    | 0.8705   | 0.8704   |
| LightGBM          | 0.8790   | 0.8788    | 0.8790   | 0.8789   |
| MLP               | 0.8509   | 0.8521    | 0.8509   | 0.8506   |
| **Stacking Ensemble** | **0.8797** | **0.8794** | **0.8797** | **0.8795** |

As shown in the table, the **stacking ensemble achieved the highest accuracy (0.8797)** and F1-score (0.8795) on the test data, demonstrating the effectiveness of combining diverse models. Confusion matrices for each model provide a detailed breakdown of the classification performance across the Benign and malware categories. Feature importance analysis revealed that different models prioritize different features, highlighting the benefit of leveraging diverse perspectives in the ensemble.

## Key Libraries Used

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn (`sklearn`)
* TensorFlow (`tensorflow`, `keras`)
* XGBoost (`xgboost`)
* LightGBM (`lightgbm`)
* Imbalanced-learn (`imblearn`)
* Optuna

This repository contains the code and analysis conducted for this project, showcasing a practical application of ensemble learning for robust malware classification using memory-based features.
