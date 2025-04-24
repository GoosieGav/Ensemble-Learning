# Robust Malware Classification via Stacking Ensemble

This project explores a robust approach to malware classification by employing a stacking ensemble of diverse machine learning models: XGBoost, LightGBM, Random Forest, and a Multi-Layer Perceptron (MLP). The goal is to leverage the complementary strengths of these models to achieve high accuracy and generalization in detecting and categorizing malware based on memory-based behavioral features.

## Dataset

The core of this research utilizes the **CIC MalMem 2024 OSEF dataset**. This dataset provides a rich collection of dynamic behavioral features extracted from system memory during the execution of both benign and malicious software samples. Key characteristics of the dataset include:

* **Dynamic Behavioral Features:** Capturing runtime activities such as process interactions, service enumeration, handle allocations, and module load characteristics. These features are particularly relevant for detecting modern malware that employs evasion techniques.
* **Multi-Class Classification:** The dataset, after preprocessing, is structured for multi-class classification, distinguishing between **Benign** software and three distinct malware families: **Ransomware**, **Spyware**, and **Trojan**.
* **Class Distribution Handling:** The initial class distribution was imbalanced. To address this, the **Synthetic Minority Over-sampling Technique (SMOTE)** was applied to create a more balanced training dataset.
* **Feature Preprocessing:** The data underwent several preprocessing steps, including:
    * **Removal of Duplicates:** Ensuring data integrity.
    * **Categorical Encoding:** Using `LabelEncoder` to convert malware category labels into numerical representations.
    * **Constant Feature Removal:** Eliminating features with no variance.
    * **Train-Test Split:** Dividing the data into training (80%) and testing (20%) sets, with stratification to maintain class proportions.
    * **Feature Scaling:** Applying `StandardScaler` to normalize the numerical features.

## Methodology

The project employs a stacking ensemble, a meta-learning technique that combines the predictions of multiple base models to form a final prediction. The key steps in our methodology include:

1.  **Individual Model Training:** Four distinct machine learning models were trained on the preprocessed training data:
    * **XGBoost:** Optimized using Optuna for hyperparameter tuning.
    * **LightGBM:** Also optimized using Optuna for hyperparameter tuning.
    * **Random Forest:** Hyperparameters tuned using Optuna with class balancing.
    * **Multi-Layer Perceptron (MLP):** A deep learning model built with TensorFlow/Keras, incorporating regularization, dropout, and a learning rate scheduler.

2.  **Hyperparameter Optimization:** Optuna, a powerful hyperparameter optimization framework, was used to find the best configurations for the tree-based models (XGBoost, LightGBM, Random Forest) through cross-validation on the resampled training data.

3.  **MLP Architecture:** The MLP model consists of several dense layers with ReLU activation, layer normalization, dropout for regularization, and a final Softmax layer for multi-class classification.

4.  **Stacking Ensemble Construction:** The predictions (probabilities) from the trained base models were used as input features for a meta-classifier, which in this case was a `LogisticRegression` model. The `stack_method="predict_proba"` was utilized where available to provide probabilistic inputs to the meta-learner.

5.  **Model Evaluation:** The performance of each individual model and the final stacking ensemble was evaluated on the held-out test set using standard classification metrics: accuracy, precision, recall, and F1-score. Confusion matrices were also generated to visualize the classification performance of each model.

6.  **Feature Importance Analysis:** Feature importance was explored for the tree-based models and permutation importance was calculated for the MLP to understand which features contribute most significantly to the classification task.

## Results (Summary)

The stacking ensemble model demonstrated competitive performance, often outperforming the individual base models. The evaluation metrics for each model, including the ensemble, are presented in the notebooks. Confusion matrices provide a detailed view of the classification accuracy across different malware categories. Feature importance plots highlight the most influential features identified by each model.

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
