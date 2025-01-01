# ML_Metabolomics_Crohns_Disease
Machine learning applications in the analysis of GC-MS metabolomics data.

## Objective
The primary goal of this analysis is to determine which sample type—breath, blood, urine, or faeces—provides the most diagnostic signature for Crohn's disease (CD). Using machine learning (ML) techniques, we aim to identify patterns in GC-MS metabolomics data that differentiate CD samples from healthy controls.

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

***Support Vector Machines (svm, SVC)*** 

***Random Forest (RandomForestClassifier)*** 

Data Preprocessing: 
*StandardScaler, MinMaxScaler:* To normalize data, as SVMs are sensitive to feature magnitudes.

Dimensionality Reduction: 
*PCA:* Principal Component Analysis to reduce dataset complexity.

Validation and Hyperparameter Tuning:
*train_test_split, GridSearchCV, cross_val_score:* For splitting data, optimizing hyperparameters, and cross-validation.
; *StratifiedKFold:* Ensures class balance in cross-validation splits.

Performance Metrics:
*confusion_matrix, ConfusionMatrixDisplay, accuracy_score:* For evaluating model performance and visualising results.

Permutation Testing:
*permutation_test_score:* To test model performance against random label assignments.
; *permutation_importance:* For assessing the importance of features via shuffling.

Additional Utilities: 
*SciPy Bootstrap (scipy.stats.bootstrap):* For resampling methods and confidence interval estimation.

### Acknowledgments
All libraries used are open-source and maintained by their respective contributors.
