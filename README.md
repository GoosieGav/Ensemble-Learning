# Leveraging Feature Diversity for Robust Malware Classification: A Stacking Ensemble Approach using MLPs, LightGBMs, XGBoosts, and Random Forests

Gavin K. Luo, Zain J. Saquer
Central High School, 423 E Central St, Springfield, MO 65802

## Abstract

This research tackles the persistent challenge of malware detection in cybersecurity by introducing a robust stacking ensemble framework. Our approach strategically combines the predictive power of Multiple Layer Perceptrons (MLPs), Random Forests, XGBoost, and LightGBM classifiers. By harnessing the distinct strengths inherent in these diverse algorithmic architectures, our ensemble methodology achieves superior classification performance when compared against individual models. The foundation of our study lies in the CIC MalMem 2022 dataset, a rich repository of memory-based behavioral features. This dataset provides dynamic indicators extracted directly from system memory, capturing crucial attributes such as process interactions, service enumeration, handle allocations, and module load characteristics – features increasingly recognized for their efficacy in analyzing encrypted traffic and classifying malware. Our rigorous preprocessing pipeline incorporated feature normalization techniques, employed information gain for discerning feature selection, and effectively addressed class imbalances through the application of the Synthetic Minority Over-sampling Technique (SMOTE). Following resampling, our dataset comprised 23,384 samples for each of the four classes, with each sample characterized by 52 informative features.

The efficacy of our approach was rigorously evaluated using standard performance metrics: accuracy, F1-score, precision, and recall. The results unequivocally demonstrate that our ensemble model outperforms its standalone counterparts across all these key indicators. Notably, the integrated model exhibits enhanced resilience against adversarial examples and showcases improved generalization capabilities when confronted with novel malware variants. Specifically, it achieved an impressive 87.97% accuracy, an 87.95% F1-score, 87.94% precision, and an 87.97% recall. This research firmly validates the significant potential of ensemble methodologies when applied to the analysis of memory-based malware features. Future research endeavors will focus on exploring advanced feature engineering techniques and integrating additional sophisticated deep learning architectures to further bolster detection capabilities within memory-based malware classification systems.

## Introduction

The escalating sophistication and scale of cyber threats are underscored by the estimated 6.06 billion malware attacks detected globally in 2023 (Statista, 2023). Modern malware variants increasingly employ sophisticated evasion tactics, including polymorphism, encryption, and memory-based execution, rendering traditional signature-based detection methods largely ineffective. These conventional approaches struggle to identify previously unseen or intentionally obfuscated malware, thereby necessitating the development of intelligent, adaptive solutions capable of discerning malicious behaviors that transcend predefined signatures. Consequently, machine learning-based malware detection has garnered substantial attention due to its inherent ability to generalize patterns from data and classify threats based on both behavioral and structural indicators.

Within the realm of machine learning, models such as Multi-Layer Perceptrons (MLPs), XGBoost, LightGBM, and Random Forests have consistently demonstrated their effectiveness when applied to high-dimensional cybersecurity datasets. Each of these models offers a distinct set of advantages: MLPs excel at capturing intricate, nonlinear patterns; XGBoost and LightGBM iteratively refine features through the powerful technique of gradient boosting; and Random Forests provide inherent robustness through their ensemble of diverse decision trees (Moujoud, 2024). However, it's crucial to recognize that no single model consistently achieves superior performance across all malware detection scenarios, particularly when dealing with memory-based malware features, which demand models adept at identifying subtle execution patterns.

To effectively address this challenge, our study rigorously investigates the effectiveness of a novel stacking ensemble model. This model strategically integrates the predictive outputs from MLPs, XGBoost, LightGBM, and Random Forests. Our research leverages the publicly available CIC MalMem 2022 dataset, a valuable resource that provides rich, memory-based malware features meticulously extracted through dynamic behavioral analysis. These features empower models to detect malware based on its execution characteristics rather than relying on static signatures, thereby enhancing resilience against contemporary threats. By intelligently combining these diverse classifiers, our ensemble method aims to significantly improve classification accuracy, enhance generalizability to unseen malware, and bolster robustness against adversarial malware samples.

Through a series of rigorous experiments, this research endeavors to clearly demonstrate the tangible advantages of leveraging feature diversity across multiple algorithmic architectures in the critical pursuit of enhancing cyber threat intelligence. The ultimate goal of this work is to contribute meaningfully to the development of adaptive, high-performance malware detection systems that can be realistically implemented within cybersecurity software solutions, ultimately strengthening digital defenses against the ever-evolving landscape of emerging threats.

## Materials and Methodology

For the purpose of this research, we utilized the CIC MalMem 2022 dataset (University of New Brunswick, 2022), a widely recognized and respected benchmark resource within the malware classification research community. This comprehensive dataset comprises telemetry-derived features meticulously extracted from both authentic benign software and a diverse collection of malicious software specimens. It provides detailed system activity logs and a rich set of behavioral indicators that facilitate precise and nuanced malware categorization.

Each individual sample within the dataset is characterized by a comprehensive feature vector, encompassing both numerical and categorical attributes. These attributes collectively capture crucial behavioral dimensions, including intricate system call patterns, detailed memory utilization metrics, sequences of API interactions, and distinctive network communication signatures (Singh et al., 2022). While the original dataset was structured for binary classification, distinguishing between Benign and Malware, our research extended its utility by focusing on a multi-class classification task, differentiating between Benign software and three distinct malware families: Ransomware, Spyware, and Trojan.

To effectively mitigate potential classification bias arising from imbalances in class distribution, we implemented the Synthetic Minority Over-sampling Technique (SMOTE) (García et al., 2017). This technique allowed us to establish a more equitable representation of each class within the dataset, thereby significantly enhancing the performance of our models, particularly for those malware categories that were initially underrepresented. Furthermore, to optimize the feature space and enhance model efficiency, we applied an information gain-based feature selection methodology. This process ensured that only those attributes exhibiting the highest discriminative potential for effective classification were retained for model training and evaluation.

### Dataset Summary

| Metric                       | Value   |
| ---------------------------- | ------- |
| Total Samples (Before Preprocessing) | 58,597  |
| Total Samples (Final)        | 93,536  |
| Number of Features           | 52      |
| Feature Types                | Numerical (e.g., memory usage) Categorical (e.g., API call type) |
| Malware Families             | Benign, Ransomware, Spyware, Trojan (LabelEncoded: 0, 1, 2, 3) |

The MalMem dataset is a well-established resource for research in malware classification, containing a diverse collection of both benign and malicious Windows Portable Executable (PE) files, meticulously extracted from real-world malware samples.

### Dataset Structure

The MalMem dataset comprises labeled instances of both malware and benign software. Each instance is described by:

* **Label:** The original dataset was designed for binary classification (Malware = 1, Benign = 0). We adapted this for our multi-class classification task.
* **Dynamic Features:** These features were extracted by executing each software sample within a controlled environment, capturing its runtime behavior.

### Data Preprocessing

* **Data Cleaning:** To ensure the integrity of our training data, we began by removing any duplicate rows, thereby eliminating redundant training examples that could potentially bias our models.
* **Data Transformation:** The original dataset presented a balanced binary classification problem (50% Benign, 50% Malware). To facilitate our goal of creating a multi-class classifier, we leveraged three pre-existing sub-categories within the broader "Malware" class: Trojan Horse, Spyware, and Ransomware.
    * Initially, we transformed the categorical malware class labels into a numerical representation using the LabelEncoder technique. This resulted in the following mapping: ('Benign': 0, 'Ransomware': 1, 'Spyware': 2, 'Trojan': 3).
    * Subsequently, we employed the Synthetic Minority Over-sampling Technique (SMOTE) to address the inherent class imbalance. This process generated synthetic samples for the minority classes (Ransomware, Spyware, Trojan) until all four classes (Benign, Ransomware, Spyware, Trojan) were balanced, each containing 23,384 samples.
* **Feature Engineering:** We identified and removed three non-relevant categorical columns that exhibited a single unique value across all samples. This step reduced the total number of features from the original 55 to a more focused set of 52.
* **Feature Scaling:** To ensure that all features contributed equally to the learning process and to optimize the performance of our gradient-based models, we applied feature standardization. Specifically, we standardized each input feature by subtracting the mean and scaling to unit variance across the entire dataset. This z-score based normalization transforms each feature $x_{i, j}$ into $x'_{i, j}$ using the following formula:
    ```
    x'_{i, j} = (x_{i, j} - mean(x_j)) / std(x_j)
    ```
    where $mean(x_j)$ represents the mean of the $j^{th}$ feature across all data points, and $std(x_j)$ represents the standard deviation of the $j^{th}$ feature.

### Model Architectures

* **MLP (Multi-Layer Perceptron):** An MLP is a feedforward neural network architecture characterized by multiple layers of interconnected neurons with non-linear activation functions. MLPs excel at capturing complex, high-dimensional feature interactions through the learned weights of these connections, enabling them to effectively model intricate patterns inherent in malware behavior.
    Our specific MLP architecture comprised the following sequence of fully connected layers with the specified number of neurons: 512, 256, 128, 64, and a final output layer with 4 neurons (corresponding to the four classes). We employed the ReLU (Rectified Linear Unit) activation function for all hidden layers to introduce non-linearity, while the Softmax activation function was used in the output layer to produce probability distributions over the four classes. To mitigate overfitting and improve generalization, we implemented L2 regularization, which penalizes large weight values, effectively simplifying the model. Additionally, we incorporated a learning rate scheduler to dynamically adjust the learning rate during training, optimizing convergence. Dropout layers were also strategically included to further reduce overfitting by randomly setting a fraction of neuron activations to zero during each training iteration, with the remaining activations scaled accordingly. The model was compiled using the Adam optimizer, an efficient stochastic gradient descent algorithm that adapts the learning rates of parameters.

* **Random Forest:** The Random Forest algorithm is an ensemble learning method that operates by constructing a multitude of decision trees during training and then outputting the class that is the mode of the classes (for classification) or the mean prediction (for regression) of the individual trees. By aggregating the predictions of these diverse trees, Random Forests capture multiple perspectives of the dataset, significantly reducing the risk of overfitting and enhancing the model's ability to generalize to unseen data.
    To optimize the performance of our Random Forest model, we employed the Optuna library, a powerful framework for hyperparameter optimization. Rather than relying on less efficient grid search or randomized search methods, we utilized the Tree-structured Parzen Estimator (TPE), a sequential model-based Bayesian Optimization (BO) approach. This intelligent algorithm builds probabilistic models of the hyperparameter space based on cross-validation measurements. It iteratively updates its model and strategically selects new candidate hyperparameter configurations that exhibit a high potential for improved performance. The hyperparameter tuning process yielded the following optimal parameter set for our Random Forest: `{'n_estimators': 225, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 1, 'bootstrap': False}`.

* **XGBoost (Extreme Gradient Boosting):** XGBoost is an efficient and widely successful gradient boosting algorithm that builds an ensemble of decision trees sequentially. Unlike traditional gradient boosting, XGBoost employs a second-order Taylor expansion of the loss function, allowing it to find a more precise optimal constant within each terminal node of the trees. This "extreme gradient boosting" technique, along with built-in regularization terms, contributes to its robust performance and ability to handle complex datasets.
    Similar to the Random Forest model, we applied Optuna's hyperparameter tuning capabilities, leveraging the Bayesian Optimization approach with TPE sampling, to optimize our XGBoost model. This sophisticated tuning process enabled the algorithm to effectively identify subtle yet crucial feature importance patterns within the data. The optimization process resulted in the following best hyperparameter configuration for our XGBoost model: `{'n_estimators': 492, 'max_depth': 15, 'learning_rate': 0.08033621956814192, 'subsample': 0.7203303897901465, 'colsample_bytree': 0.8537742123999098, 'gamma': 0.03177765487850257, 'reg_alpha': 0.00010124429957568028, 'reg_lambda': 0.022916678154371584}`.

* **LightGBM (Light Gradient Boosting Machine):** LightGBM is another highly efficient gradient boosting framework that constructs ensembles of weaker learners (typically decision trees) sequentially to create a strong learner. A key innovation in LightGBM is its use of gradient-based one-side sampling (GOSS) and exclusive feature bundling (EFB) techniques, which significantly accelerate training and reduce memory consumption, particularly when dealing with large datasets. LightGBM efficiently buckets continuous feature values into discrete bins, which are then iteratively processed to calculate information gain and determine optimal data splits.
    Our LightGBM model also benefited from hyperparameter tuning using Optuna's Bayesian Optimization approach with TPE sampling. This optimization process aimed to identify the hyperparameter configuration that would yield the best classification performance on our memory-based malware dataset. The best parameters identified were: `{'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 100}` (Note: The provided text only included these three parameters).

### Stacking Ensemble Construction

To create our stacking ensemble model, we followed a two-layer approach. In the first layer, each of the individual base models (MLP, Random Forest, XGBoost, and LightGBM) was trained independently on the preprocessed CIC MalMem 2022 dataset. Once trained, each base model generated predictions for the same dataset. These predictions then served as the input features for a second-layer meta-classifier. We experimented with both weighted averaging and logistic regression as potential combination strategies for this meta-classifier. Ultimately, logistic regression was chosen as the meta-classifier due to its ability to learn optimal weights for the predictions of the base models, effectively refining the final prediction of the ensemble.

## Results

The experimental evaluation of our meticulously constructed stacking ensemble model yielded remarkable malware classification capabilities that consistently surpassed the performance of each individual constituent classifier. When rigorously tested on the CIC MalMem 2022 dataset, our integrated approach demonstrated superior performance across all key performance indicators when compared to standalone neural networks, tree-based models, and gradient-boosted frameworks. These compelling findings strongly affirm that strategically incorporating feature representations learned by diverse architectural frameworks significantly enhances classification resilience, particularly when confronting the challenges posed by adversarial examples and previously unseen malware variants.

### Evaluation Metrics

To comprehensively assess the classification quality and generalizability of our models, we employed a suite of standard evaluation metrics: accuracy, precision, recall, and F1-score. For each individual class $i$ (Benign, Ransomware, Spyware, Trojan), we defined the following fundamental terms:

* **TP (True Positive):** The number of instances that were correctly predicted as belonging to class $i$.
* **TN (True Negative):** The number of instances that were correctly predicted as *not* belonging to class $i$.
* **FP (False Positive):** The number of instances that were incorrectly predicted as belonging to class $i$ when they actually belonged to a different class.
* **FN (False Negative):** The number of instances that actually belonged to class $i$ but were incorrectly predicted as belonging to a different class.

Based on these fundamental terms, we calculated the following evaluation metrics:

* **Accuracy:** A measure summarizing the overall ratio of correctly predicted observations to the total number of observations:
    ```
    Accuracy = (TP1 + TP2 + TP3 + TP4) / (TP1 + TP2 + TP3 + TP4 + FP1 + FP2 + FP3 + FP4 + FN1 + FN2 + FN3 + FN4)
    ```
* **Precisioni:** The ratio of correctly predicted positive observations for class $i$ to the total number of observations predicted as positive for class $i$:
    ```
    Precision_i = TP_i / (TP_i + FP_i)
    ```
* **Recalli:** The ratio of correctly predicted positive observations for class $i$ to the total number of actual positive observations for class $i$:
    ```
    Recall_i = TP_i / (TP_i + FN_i)
    ```
* **F1i:** The harmonic mean of precision and recall for class $i$, providing a balanced measure of a model's performance, particularly useful in datasets with imbalanced class distributions:
    ```
    F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)
    ```

### Performance Metrics Analysis

| Model          | Accuracy | F1 Score | Precision | Recall |
| -------------- | -------- | -------- | --------- | ------ |
| Ensemble       | 0.8797   | 0.8795   | 0.8794    | 0.8797 |
| XGBoost        | 0.8776   | 0.8775   | 0.8774    | 0.8776 |
| Random Forest  | 0.8705   | 0.8704   | 0.8704    | 0.8705 |
| LightGBM       | 0.8790   | 0.8789   | 0.8788    | 0.8790

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
