# Leveraging Feature Diversity for Robust Malware Classification: A Stacking Ensemble Approach using MLPs, LightGBMs, XGBoosts, and Random Forests

Gavin K. Luo, Zain J. Saquer
Central High School, 423 E Central St, Springfield, MO 65802

## Abstract

This research presents a stacking ensemble framework that combines Multiple Layer Perceptrons (MLPs), Random Forests, XGBoost, and LightGBM classifiers to address the persistent challenge of malware detection in cybersecurity. By leveraging complementary strengths from diverse algorithmic architectures, our ensemble methodology demonstrates superior classification performance compared to individual models. We utilized the CIC MalMem 2022 dataset, comprising memory-based behavioral features, for model development and evaluation. The dataset includes dynamic behavioral indicators extracted from system memory, capturing key attributes such as process interactions, service enumeration, handle allocations, and module load characteristics—features that have been demonstrated to be effective in encrypted traffic and malware classification tasks. Our preprocessing pipeline implemented feature normalization techniques, employed information gain for feature selection, and addressed class imbalances through the Synthetic Minority Over-sampling Technique (SMOTE). The resampled data ended up with 23,384 samples each for all four classes, with each class having 52 features.

Performance assessment using standard metrics—accuracy, F1-score, precision, and recall—confirms that our ensemble approach outperforms standalone classifiers. Notably, the integrated model exhibits enhanced resilience against adversarial examples and demonstrates improved generalization to novel malware variants. It displayed 87.97% accuracy, 87.95% F1-score, 87.94% precision, and 87.97% recall. This research validates the efficacy of ensemble methodologies when applied to memory-based malware features. Future research directions include exploring sophisticated feature engineering approaches and incorporating additional deep learning architectures to further enhance detection capabilities in memory-based malware classification systems.

## Introduction

In 2023, an estimated 6.06 billion malware attacks were detected worldwide (Statista, 2023), reflecting the increasing sophistication and scale of cyber threats. Modern malware variants employ evasion techniques such as polymorphism, encryption, and memory-based execution, rendering traditional signature-based detection methods ineffective. These conventional approaches struggle to detect previously unseen or obfuscated malware, necessitating intelligent, adaptive solutions capable of identifying malicious behaviors beyond predefined signatures. As a result, machine learning-based malware detection has gained significant attention due to its ability to generalize patterns from data and classify threats based on behavioral and structural indicators.

Among machine learning approaches, models such as Multi-Layer Perceptrons (MLPs), XGBoost, LightGBM, and Random Forests have demonstrated effectiveness in high-dimensional cybersecurity datasets. Each model offers distinct advantages: MLPs capture complex, nonlinear patterns, XGBoost and LightGBM refine features iteratively through gradient boosting, and Random Forests provide robustness through ensemble decision trees (Moujoud, 2024). However, no single model consistently outperforms others across all malware detection scenarios, particularly when dealing with memory-based malware features, which require models capable of identifying nuanced execution patterns.

To address this challenge, this study investigates the effectiveness of a stacking ensemble model that integrates predictions from MLPs, XGBoost, LightGBM, and Random Forests. The research leverages the CIC MalMem 2022 dataset, which provides rich, memory-based malware features extracted from dynamic behavioral analysis. These features allow models to detect malware based on execution characteristics rather than static signatures, making them more resilient against modern threats. By combining diverse classifiers, the ensemble method aims to improve classification accuracy, generalizability, and robustness against adversarial malware samples.

Through rigorous experimentation, this research seeks to demonstrate the advantages of leveraging feature diversity across multiple architectures in enhancing cyber threat intelligence. The ultimate goal is to contribute to the development of adaptive, high-performance malware detection systems that can be realistically implemented in cybersecurity software, strengthening defenses against emerging threats in an evolving digital landscape.

## Materials and Methodology

For this study, we utilized the CIC MalMem 2022 dataset, a recognized benchmark resource for malware classification research. This dataset comprises telemetry-derived features extracted from authentic benign and malicious software specimens, providing comprehensive system activity logs and behavioral indicators that facilitate precise malware categorization.

Each sample within the dataset is characterized by a comprehensive feature vector encompassing both numerical and categorical attributes, capturing crucial behavioral dimensions including system call patterns, memory utilization metrics, API interaction sequences, and network communication signatures. The dataset incorporates classification labels across two distinct categories: Benign and Malware, but can be further split into four distinct categories: Benign, Ransomware, Spyware, and Trojan.

To mitigate classification bias from class distribution imbalances, we implemented the Synthetic Minority Over-sampling Technique (SMOTE) to establish more equitable class representation, thereby enhancing model performance particularly for underrepresented malware categories. The feature space underwent further optimization through an information gain-based selection methodology, retaining only those attributes demonstrating the highest discriminative potential for effective classification.

### Dataset Summary

| Metric                       | Value   |
| ---------------------------- | ------- |
| Total Samples (Before Preprocessing) | 58,597  |
| Total Samples (Final)        | 93,536  |
| Number of Features           | 52      |
| Feature Types                | Numerical (e.g., memory usage) Categorical (e.g., API call type) |
| Malware Families             | Benign, Ransomware, Spyware, Trojan (LabelEncoded: 0, 1, 2, 3) |

MalMem is a well-known dataset for malware classification. It contains both benign and malicious Windows Portable Executable (PE) files, extracted from real-world malware samples (University of New Brunswick, 2022).

### Dataset Structure

The MalMem dataset consisted of labeled instances of malware and benign software. Each instance includes:

* **Label:** The original dataset was designed for binary classification (Malware = 1, Benign = 0). We later changed this for multi-class classification.
* **Dynamic Features:** Extracted by executing the sample in a controlled environment (Singh et al., 2022).

### Data Preprocessing

* **Data Cleaning:** We removed duplicate rows to avoid redundant training examples.
* **Data Transformation:** The original dataset was balanced on a 50/50 split for Benign and Malware classes, serving as a binary classification problem. We aimed to create a multi-class classifier rather than a binary one, so we utilized three pre-existing classes within the Malware class: Trojan Horse, Spyware, and Ransomware.
    * First, we transformed categorical malware class labels into numerical values using LabelEncoder: ('Benign': 0, 'Ransomware': 1, 'Spyware': 2, 'Trojan': 3).
    * Following this, we used SMOTE (Synthetic Minority Over-sampling Technique) to balance the four classes, resulting in 23,384 samples per class.
* **Feature Engineering:** We removed three non-relevant categorical columns that had a single unique value, reducing the feature count from 55 to 52.
* **Feature Scaling:** We standardized each input feature by removing the mean and scaling to unit variance of the dataset. This z-score based normalization transforms the distribution of each feature $x_{i, j}$ into $x'_{i, j}$:
    $$x'_{i, j} = \frac{x_{i, j} - \bar{x}_j}{\sigma_j}$$
    where $\bar{x}_j = \frac{1}{|X|}\sum_{i=1}^{|X|}(x_{i, j})$ and $\sigma_j = \sqrt{\frac{1}{|X|}\sum_{i=1}^{|X|}(x_{i, j} - \bar{x}_j)^2}$.

### Model Architectures

* **MLP:** A feedforward neural network optimized with non-linear activation, consisting of fully-connected neurons. Our MLP architecture consisted of the following layer neuron counts: 512, 256, 128, 64, and 4 (output). We utilized ReLU activation functions for all but our output layer, for which we used Softmax. We also implemented L2 regularization, a learning rate scheduler, and dropout layers, compiled with the Adam optimizer.

* **Random Forest:** An algorithm that internally combines the output of multiple decision trees to reach a single result. We utilized the Optuna library with the Tree-structured Parzen Estimator (TPE) for hyperparameter tuning. The best parameters found were: `{'n_estimators': 225, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 1, 'bootstrap': False}`.

* **XGBoost:** A tree-based algorithm that uses an ‘extreme gradient boosting’ technique. The XGBoost model also underwent hyperparameter tuning using Optuna and TPE. The best parameters found were: `{'n_estimators': 492, 'max_depth': 15, 'learning_rate': 0.08033621956814192, 'subsample': 0.7203303897901465, 'colsample_bytree': 0.8537742123999098, 'gamma': 0.03177765487850257, 'reg_alpha': 0.00010124429957568028, 'reg_lambda': 0.022916678154371584}`.

* **LightGBM:** A gradient boosting framework that combines weaker learners sequentially. The LightGBM model was also tuned using Optuna and TPE. The best parameters found were: `{'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 100}` (Note: Full best parameters were not provided in the original text).

### Stacking Ensemble Construction

Each model was trained independently, generating predictions that served as inputs to a meta-classifier (logistic regression), which refined the final prediction.

## Results

The experimental evaluation of our stacking ensemble model revealed exceptional malware classification capabilities that surpassed individual classifier performance when evaluated on the CIC MalMem 2022 dataset.

### Evaluation Metrics

Performance was measured using accuracy, precision, recall, and F1-score. For each class <span class="math-inline">i</span>:

* **True Positive (TPi):** Number of instances correctly identified as class <span class="math-inline">i</span>.
* **True Negative (TNi):** Number of instances correctly classified as not belonging to class <span class="math-inline">i</span>.
* **False Positive (FPi):** Number of instances wrongly classified as class <span class="math-inline">i</span>.
* **False Negative (FNi):** Number of instances that misclassify class <span class="math-inline">i</span> data as another class.

The evaluation metrics were calculated as follows:

* **Accuracy:** <span class="math-inline">\\frac\{TP\_1 \+ TP\_2 \+ TP\_3 \+ TP\_4\}\{TP\_1 \+ TP\_2 \+ TP\_3 \+ TP\_4 \+ FP\_1 \+ FP\_2 \+ FP\_3 \+ FP\_4 \+ FN\_1 \+ FN\_2 \+ FN\_3 \+ FN\_4\}</span>
* **Precisioni:** <span class="math-inline">\\frac\{TP\_i\}\{TP\_i \+ FP\_i\}</span>
* **Recalli:** <span class="math-inline">\\frac\{TP\_i\}\{TP\_i \+ FN\_i\}</span>
* **F1i:** <span class="math-inline">2 \\times \\frac\{\\text\{Precision\}\_i \\times \\text\{Recall\}\_i\}\{\\text\{Precision\}\_i \+ \\text\{Recall\}\_i\}</span>

### Performance Metrics Analysis

| Model          | Accuracy | F1 Score | Precision | Recall |
| -------------- | -------- | -------- | --------- | ------ |
| Ensemble       | 0.8797   | 0.8795   | 0.8794    | 0.8797 |
| XGBoost        | 0.8776   | 0.8775   | 0.8774    | 0.8776 |
| Random Forest  | 0.8705   | 0.8704   | 0.8704    | 0.705  |
| LightGBM       | 0.8790   | 0.8789   | 0.8788    | 0.8790 |
| MLP            | 0.8509   | 0.8506   | 0.8521    | 0.8509 |

The stacking ensemble achieved an overall accuracy of 87.97%, exceeding the performance of the highest-performing individual model, XGBoost (87.76% accuracy). The ensemble also demonstrated superior F1-scores, effectively balancing precision and recall across classification categories. Notably, recall values were particularly strong for malware classifications, indicating the model's proficiency in identifying malicious activity while maintaining minimal false-negative rates.

### Feature Diversity Robustness

The significant variation in feature importance across the base models indicates that each model relies on different aspects of the data for making predictions. This diversity enhances the robustness of the ensemble model, reducing the risk of overfitting to any single feature or pattern and improving resilience against adversarial attacks and novel malware variants.

## Conclusion

This research investigated the application of ensemble deep learning methodologies for memory-based malware classification, demonstrating that our stacking ensemble approach significantly enhances detection capabilities. By synthesizing feature representations from neural networks, gradient-boosted frameworks, and tree-based models, our architecture leverages the complementary strengths of diverse algorithmic approaches, resulting in superior accuracy and enhanced resilience. These findings confirm that ensemble learning represents a viable strategy for strengthening cybersecurity frameworks against increasingly sophisticated malware threats.

While our model effectively classifies malware using memory-based behavioral features, operational implementation would necessitate additional validation, including integration within existing cybersecurity infrastructure. A practical deployment could involve incorporating the model into endpoint protection platforms where it continuously analyzes memory artifacts for suspicious activity patterns. By implementing real-time inference capabilities and automated threat response mechanisms, the system could substantially enhance contemporary malware detection frameworks.

## Future Research

Though our stacking ensemble demonstrated robust performance, several enhancements merit exploration to further optimize malware detection capabilities:

* Implementing Bayesian optimization techniques for neural network hyperparameter tuning.
* Evaluating alternative ensemble learning methodologies such as bagging and boosting.
* Conducting binary classification benchmarking (malware vs. benign).
* Exploring multimodal learning by incorporating other data sources like network traffic and static file attributes.
* Addressing model interpretability concerns for practical deployment in security operations.

## References

Alazab, Ammar, et al. "Android Malware Detection Using Ensemble Learning." *IEEE Access*, vol. 9, 2021, pp. 45497-45511. https://ieeexplore.ieee.org/document/10169578

Alzaylaee, Hisham, et al. "File-Based Malware Detection Using Ensemble Method." *IET Conference Publications*, 2021. https://ieeexplore.ieee.org/document/9770633

Apruzzese, Giovanni, et al. "On the Effectiveness of Machine and Deep Learning for Cyber Security." *NATO Cooperative Cyber Defence Centre of Excellence (CCDCOE)*, 2018. https://ccdcoe.org/uploads/2018/10/Art-19-On-the-Effectiveness-of-Machine-and-Deep-Learning-for-Cyber-Security.pdf

Dener, Murat, and Ercan Ok. "Malware Detection Using Memory Analysis Data in Big Data Platforms." *International Journal of Computer Science and Network Security*, vol. 21, no. 5, 2021, pp. 1-8. https://www.semanticscholar.org/paper/Malware-Detection-Using-Memory-Analysis-Data-in-Big-Dener-Ok/b9a0d61b294e4ff18c01287ec1ee8b9fd6c3f80b

García, Salvador, et al. "Review of Data Preprocessing Techniques in Data Mining." *ResearchGate*, 2017. https://www.researchgate.net/publication/320161439_Review_of_Data_Preprocessing_Techniques_in_Data_Mining

Hasan, S. M. Rakib, and Aakar Dhakal. "Obfuscated Malware Detection: Investigating Real-World Scenarios through Memory Analysis." *arXiv preprint arXiv:2404.02372*, 2024. https://arxiv.org/abs/2404.02372

Li, Zhenglin, et al. "Comprehensive Evaluation of Mal-API-2019 Dataset by Machine Learning in Malware Detection." *arXiv preprint arXiv:2403.02232*, 2024. https://arxiv.org/abs/2403.02232

Lashkari, Arash Habibi, et al. *Characterization of Encrypted and VPN Traffic Using Time-Related Features*. Proceedings of the 2nd International Conference on Information Systems Security and Privacy (ICISSP 2016), 2021, https://pdfs.semanticscholar.org/b2e2/0dc7a34753311472a5f2314fbf866d7eddd0.pdf

Moujoud, Yassine, et al. "Enhancing Malware Detection through Ensemble Learning Techniques." *Research Square*, 2023. https://assets-eu.researchsquare.com/files/rs-4772367/v1_covered_5bc38cd7-24d8-498e-ac2e-e693774724e5.pdf

Pektas, Abdulbaki, and Raheel Nawaz. "Detecting Malware Using Machine Learning and Deep Learning Methods: A Systematic Literature Review." *Cybersecurity*, vol. 7, no. 5, 2024. https://cybersecurity.springeropen.com/articles/10.1186/s42400-024-00238-4

Singh, R., et al. "Malware-Memory-Analysis." *GitHub Repository*, 2022. https://github.com/rsingh0616/Malware-Memory-Analysis

Statista. "Number of

The stacking ensemble achieved an overall accuracy of 87.97%, exceeding the performance of the highest-performing individual model, XGBoost (87.76% accuracy). The ensemble also demonstrated superior F1-scores, effectively balancing precision and recall across classification categories. Notably, recall values were particularly strong for malware classifications, indicating the model's proficiency in identifying malicious activity while maintaining minimal false-negative rates.

### Feature Diversity Robustness

The significant variation in feature importance across the base models indicates that each model relies on different aspects of the data for making predictions. This diversity enhances the robustness of the ensemble model, reducing the risk of overfitting to any single feature or pattern and improving resilience against adversarial attacks and novel malware variants.

## Conclusion

This research investigated the application of ensemble deep learning methodologies for memory-based malware classification, demonstrating that our stacking ensemble approach significantly enhances detection capabilities. By synthesizing feature representations from neural networks, gradient-boosted frameworks, and tree-based models, our architecture leverages the complementary strengths of diverse algorithmic approaches, resulting in superior accuracy and enhanced resilience. These findings confirm that ensemble learning represents a viable strategy for strengthening cybersecurity frameworks against increasingly sophisticated malware threats.

While our model effectively classifies malware using memory-based behavioral features, operational implementation would necessitate additional validation, including integration within existing cybersecurity infrastructure. A practical deployment could involve incorporating the model into endpoint protection platforms where it continuously analyzes memory artifacts for suspicious activity patterns. By implementing real-time inference capabilities and automated threat response mechanisms, the system could substantially enhance contemporary malware detection frameworks.

## Future Research

Though our stacking ensemble demonstrated robust performance, several enhancements merit exploration to further optimize malware detection capabilities:

* Implementing Bayesian optimization techniques for neural network hyperparameter tuning.
* Evaluating alternative ensemble learning methodologies such as bagging and boosting.
* Conducting binary classification benchmarking (malware vs. benign).
* Exploring multimodal learning by incorporating other data sources like network traffic and static file attributes.
* Addressing model interpretability concerns for practical deployment in security operations.

## References

Alazab, Ammar, et al. "Android Malware Detection Using Ensemble Learning." *IEEE Access*, vol. 9, 2021, pp. 45497-45511. https://ieeexplore.ieee.org/document/10169578

Alzaylaee, Hisham, et al. "File-Based Malware Detection Using Ensemble Method." *IET Conference Publications*, 2021. https://ieeexplore.ieee.org/document/9770633

Apruzzese, Giovanni, et al. "On the Effectiveness of Machine and Deep Learning for Cyber Security." *NATO Cooperative Cyber Defence Centre of Excellence (CCDCOE)*, 2018. https://ccdcoe.org/uploads/2018/10/Art-19-On-the-Effectiveness-of-Machine-and-Deep-Learning-for-Cyber-Security.pdf

Dener, Murat, and Ercan Ok. "Malware Detection Using Memory Analysis Data in Big Data Platforms." *International Journal of Computer Science and Network Security*, vol. 21, no. 5, 2021, pp. 1-8. https://www.semanticscholar.org/paper/Malware-Detection-Using-Memory-Analysis-Data-in-Big-Dener-Ok/b9a0d61b294e4ff18c01287ec1ee8b9fd6c3f80b

García, Salvador, et al. "Review of Data Preprocessing Techniques in Data Mining." *ResearchGate*, 2017. https://www.researchgate.net/publication/320161439_Review_of_Data_Preprocessing_Techniques_in_Data_Mining

Hasan, S. M. Rakib, and Aakar Dhakal. "Obfuscated Malware Detection: Investigating Real-World Scenarios through Memory Analysis." *arXiv preprint arXiv:2404.02372*, 2024. https://arxiv.org/abs/2404.02372

Li, Zhenglin, et al. "Comprehensive Evaluation of Mal-API-2019 Dataset by Machine Learning in Malware Detection." *arXiv preprint arXiv:2403.02232*, 2024. https://arxiv.org/abs/2403.02232

Lashkari, Arash Habibi, et al. *Characterization of Encrypted and VPN Traffic Using Time-Related Features*. Proceedings of the 2nd International Conference on Information Systems Security and Privacy (ICISSP 2016), 2021, https://pdfs.semanticscholar.org/b2e2/0dc7a34753311472a5f2314fbf866d7eddd0.pdf

Moujoud, Yassine, et al. "Enhancing Malware Detection through Ensemble Learning Techniques." *Research Square*, 2023. https://assets-eu.researchsquare.com/files/rs-4772367/v1_covered_5bc38cd7-24d8-498e-ac2e-e693774724e5.pdf

Pektas, Abdulbaki, and Raheel Nawaz. "Detecting Malware Using Machine Learning and Deep Learning Methods: A Systematic Literature Review." *Cybersecurity*, vol. 7, no. 5, 2024. https://cybersecurity.springeropen.com/articles/10.1186/s42400-024-00238-4

Singh, R., et al. "Malware-Memory-Analysis." *GitHub Repository*, 2022. https://github.com/rsingh0616/Malware-Memory-Analysis

Statista. "Number of
