# ML_Metabolomics_Crohns_Disease
Machine learning applications in the analysis of GC-MS metabolomics data.

This repository contains the workflow for analyding metabolomic data associated with Crohn's Disease. The project consists of data preprocessing, exploratory data analysis (EDA), and machine learning classification using Support Vector Machine (SVM) and Random Forest (RF) models.

## Objective
The primary goal of this analysis is to determine which sample type—breath, blood, urine, or faeces—provides the most diagnostic signature for Crohn's disease (CD). By building and evaluating machine learning models, we aim to identify patterns in GC-MS metabolomics data that differentiate CD samples from healthy controls.

## Classification Outcomes Across Sample Types

Faecal samples achieved the highest classification accuracy among all sample types.
The RF model outperformed the SVM model (details below).

## Folder Structure

Below is an overview of the folder structure and the analyses performed.

### data_preprocessing_and_eda/
This folder contains the *preprocess_eda_notebook.ipynb* file, which focuses on data preprocessing and exploratory data analysis (EDA). 

### PCA/
This folder includes jupyter notebook for PCA-related analysis. 

PCA was used primarily for exploratory analysis.

*Key findings:*

PCA did not reveal clear separation between cases and controls. Outliers were identified but retained to maintain statistical power due to the small dataset size.

PCA scores did not improve classification performance compared to raw data and were excluded from further modeling.

### SVM_SVC_Model/
This folder contains the *SVM_SVC_Model_Notebook.ipynb*, which implements an SVM classifier using the Scikit-learn SVC module.

*Highlights:*

Optimal parameters: C=1, gamma=1e-04. Achieved 100% Training Accuracy and 79% Cross-Validation Accuracy.

The SVM classifier achieved a Permutation Test Score of 0.79 when applied to the real data. The p-value for the real data was found to be 0.01698, reinforcing the significance of the results. On the other hand, when applied to randomised data, the model's performance dropped with a Permutation Test Score of 0.31 and a p-value of 0.944, suggesting that the model relies heavily on genuine features rather than random noise.

Feature Importance:
Permutation importance identified key retention times around 13.87 min and 22.8 min, highlighting critical metabolic features.

### Random_Forest_Model/
This folder includes the *Random_Forest_Model_Notebook.ipynb*, which focuses on training and evaluating a Random Forest model.

*Highlights:*

Optimal configuration: max_depth=None, max_features='log2', n_estimators=200, min_samples_split=2. Achieved 100% Training Accuracy and 88% Cross-Validation Accuracy.

The Random Forest model achieved a Permutation Test Score of 0.82 on the real data, showing strong predictive power. The p-value for the real data was 0.00899, indicating the results are statistically significant. In contrast, when tested on randomized data, the model's performance was reduced with a Permutation Test Score of 0.53 and a p-value of 0.419, pointing to the importance of genuine features for accurate classification.

## Libraries and Packages Used

### Core Libraries

***Pandas (pd):*** For efficient data wrangling, manipulation, and handling of tabular data.

***NumPy (np):*** For numerical computations, linear algebra, and array manipulations.

***Matplotlib (plt):*** For creating plots and visualisations of data and model results.

***SciPy (scipy.stats, loadmat):*** 
Statistical functions and bootstrapping support.
Loading MATLAB .mat files for data import.


### Machine Learning and Model Development

***Scikit-learn (sklearn)- machine learning library, including:***

Model Development and Training:
*Support Vector Machines (svm, SVC)*
; *Random Forest (RandomForestClassifier)*

Data Preprocessing: 
*StandardScaler, MinMaxScaler:* : For feature scaling and normalisation.

Dimensionality Reduction: 
*PCA:* Principal Component Analysis to reduce dataset complexity.

Validation and Hyperparameter Tuning:
*train_test_split, GridSearchCV, cross_val_score:* For splitting data, optimising hyperparameters, and cross-validation.
; *StratifiedKFold:* Ensures class balance in cross-validation splits.

Performance Metrics:
*confusion_matrix, ConfusionMatrixDisplay, accuracy_score:* For evaluating model performance and visualising results.

Permutation Testing:
*permutation_test_score:* To test model performance against random label assignments.
; *permutation_importance:* For assessing the importance of features via shuffling.

## Acknowledgments

All libraries used are open-source and maintained by their respective contributors.

The data used in this project was derived from the study outlined in the paper: "A Comprehensive Metabolomics Study of Human Urine, Plasma, and Fecal Samples". For more details, please refer to the publication: DOI: 10.1007/s11306-014-0650-1.

I would also like to acknowledge Professor Conrad Bessant, for invaluable guidance and for the preparation of the practical assignment upon which this project is based.
