# CC Fraud Detection
Predict credit card fraud using various ML models - by using SMOTE and ADASYNC  

# Dataset
The dataset for this project was obtained from kaggle website.
[https://www.kaggle.com/mlg-ulb/creditcardfraud]

Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

# Exploratory data analysis

### 1. Data is heavily imbalanced, with fraud transactions just less than 2% of the total
Hence, the usage of oversampling/undersampling techniques to balance the data.

![alt](https://github.com/aakashsingh11/CC-fraud-Detection/blob/master/images/output_6_1.png?raw=true)!


### 2. Distribution of transaction amounts according to class
Checking for outliers. Only extreme values will be removed since we dont want the reduction of already scarce fraud transactions.

![alt](https://github.com/aakashsingh11/CC-fraud-Detection/blob/master/images/output_21_0.png?raw=true)!


### 3. Number of transactions vs time
2 peaks are visible which corresponds to the two days of data. Most of the transactions occur during the daytime, and drop to minimum  at night.

![alt](https://github.com/aakashsingh11/CC-fraud-Detection/blob/master/images/output_12_0.png?raw=true)!

### 4. Fraud & genuine transactions vs time
Fraud transactions are maximum during 0-7 hours and genuine transactions are maximum during the rest of the hours.

![alt](https://github.com/aakashsingh11/CC-fraud-Detection/blob/master/images/output_32_1.png?raw=true)

### 5. Scatter plot of unbalanced dataset 
![alt](https://github.com/aakashsingh11/CC-fraud-Detection/blob/master/images%202/output_40_0.png?raw=true) 

### 6. After using ADASYNC oversampling
![alt](https://github.com/aakashsingh11/CC-fraud-Detection/blob/master/images%202/output_42_0.png?raw=true)

# Performace of various Models ranked
It is observed that Random Forest models have the highest precision and recall followed by decision trees. the performance can be further improved by using GridsearchCV to detect the best parameters for the given models. ADASYN and SMOTE give promising results- the AUC score is improved. Each model has a high true positive rate and a low false-positive rate.

|    | Model            | Accuracy | AUC      | PrecisionScore | RecallScore | F1Score  |
|----|------------------|----------|----------|----------------|-------------|----------|
| 12 | RF Oversampling  | 0.999933 | 0.999933 | 0.999866       | 1.000000    | 0.999933 |
| 13 | RF SMOTE         | 0.999866 | 0.999865 | 0.999732       | 1.000000    | 0.999866 |
| 14 | RF ADASYN        | 0.999860 | 0.999859 | 0.999732       | 0.999988    | 0.999860 |
| 7  | DT Oversampling  | 0.999774 | 0.999774 | 0.999549       | 1.000000    | 0.999775 |
| 9  | DT ADASYN        | 0.998761 | 0.998760 | 0.998173       | 0.999354    | 0.998763 |
| 8  | DT SMOTE         | 0.998260 | 0.998259 | 0.997480       | 0.999049    | 0.998264 |
| 2  | LR Oversampling  | 0.950902 | 0.950946 | 0.978971       | 0.921753    | 0.949501 |
| 3  | LR SMOTE         | 0.948893 | 0.948939 | 0.977837       | 0.918766    | 0.947382 |
| 11 | RF Undersampling | 0.933579 | 0.935870 | 0.992000       | 0.879433    | 0.932331 |
| 1  | LR Undersampling | 0.933579 | 0.934970 | 0.969466       | 0.900709    | 0.933824 |
| 16 | NB Undersampling | 0.929889 | 0.931724 | 0.976562       | 0.886525    | 0.929368 |
| 17 | NB Oversampling  | 0.927970 | 0.928044 | 0.973744       | 0.879886    | 0.924439 |
| 18 | NB SMOTE         | 0.924124 | 0.924204 | 0.973661       | 0.872071    | 0.920070 |
| 15 | NB imbalance     | 0.978801 | 0.915976 | 0.063215       | 0.852941    | 0.117707 |
| 6  | DT Undersampling | 0.904059 | 0.903901 | 0.907801       | 0.907801    | 0.907801 |
| 10 | RF imbalance     | 0.999573 | 0.900686 | 0.931624       | 0.801471    | 0.861660 |
| 4  | LR ADASYN        | 0.891804 | 0.891831 | 0.911571       | 0.868070    | 0.889289 |
| 5  | DT imbalance     | 0.999171 | 0.885804 | 0.739437       | 0.772059    | 0.755396 |
| 0  | LR imbalance     | 0.999256 | 0.819780 | 0.878788       | 0.639706    | 0.740426 |
| 19 | NB ADASYN        | 0.735312 | 0.735569 | 0.920660       | 0.515683    | 0.661080 |
