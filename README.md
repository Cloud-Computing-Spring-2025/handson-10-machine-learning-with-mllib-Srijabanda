
# Customer Churn Prediction with MLlib

This project uses Apache Spark MLlib to predict customer churn based on structured customer data. It includes data preprocessing, model training, feature selection, and hyperparameter tuning using various machine learning classifiers.

---

## Dataset

The dataset used is `customer_churn.csv`, which includes features like:

- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (label), etc.

---

## Prerequisites

Make sure you have the following installed:

```bash
python --version
pip install pyspark
```

---

## Step-by-Step Execution

### 1. Generate Dataset

```bash
python dataset-generator.py
```

### 2. Run the Churn Prediction Pipeline

```bash
spark-submit customer-churn-analysis.py
```

---

## Tasks Breakdown

### Task 1: Data Preprocessing and Feature Engineering

**Objective:** Prepare the raw dataset for machine learning by cleaning, encoding, and assembling features.

**Steps:**
1. Handle missing values in `TotalCharges` by filling them with 0.
2. Convert categorical columns like `gender`, `PhoneService`, and `InternetService` using `StringIndexer`.
3. Apply `OneHotEncoder` to indexed columns.
4. Assemble all features into a single vector using `VectorAssembler`.

**Code Explanation:**
- Missing values are filled using `fillna(0)`.
- Categorical variables are indexed and one-hot encoded.
- `VectorAssembler` is used to combine features into a single vector.

**Sample Output:**
```
=== Data Preprocessing ===
Transformed sample (features and label):
Row(features=DenseVector([1.0, 32.0, 94.76, 3209.02, 1.0, 1.0, 0.0, 1.0]), label=0.0)
Row(features=DenseVector([1.0, 25.0, 66.16, 1697.94, 1.0, 0.0, 1.0, 0.0]), label=0.0)
Row(features=SparseVector(8, {1: 63.0, 2: 25.73, 3: 1078.91}), label=0.0)
Row(features=DenseVector([1.0, 20.0, 86.46, 1882.06, 1.0, 0.0, 1.0, 0.0]), label=0.0)
Row(features=SparseVector(8, {1: 6.0, 2: 107.77, 3: 642.2, 5: 1.0}), label=1.0)
```

---

### Task 2: Train and Evaluate Logistic Regression Model

**Objective:** Train a logistic regression model and evaluate its performance using AUC.

**Steps:**
1. Split dataset into training (80%) and test (20%) sets.
2. Train a `LogisticRegression` model.
3. Use `BinaryClassificationEvaluator` to compute AUC.

**Code Explanation:**
- `randomSplit([0.8, 0.2])` is used to split the dataset.
- Model trained using `LogisticRegression()`.
- Evaluated using AUC metric.

**Sample Output:**
```
=== Logistic Regression ===
AUC Score: 0.7290
```

---

### Task 3: Feature Selection using Chi-Square Test

**Objective:** Select top 5 relevant features using a Chi-Square test.

**Steps:**
1. Use `ChiSqSelector` to select top 5 features.
2. Display selected features and corresponding label.

**Code Explanation:**
- Chi-square test ranks features by correlation with the label.
- The top 5 features are retained.

**Sample Output:**
```
=== Chi-Square Feature Selection ===
Top 5 selected features (first 5 rows):
Row(selectedFeatures=DenseVector([1.0, 32.0, 94.76, 0.0, 1.0]), label=0.0)
Row(selectedFeatures=DenseVector([1.0, 25.0, 66.16, 1.0, 0.0]), label=0.0)
Row(selectedFeatures=SparseVector(5, {1: 63.0, 2: 25.73}), label=0.0)
Row(selectedFeatures=DenseVector([1.0, 20.0, 86.46, 1.0, 0.0]), label=0.0)
Row(selectedFeatures=SparseVector(5, {1: 6.0, 2: 107.77}), label=1.0)
```

---

### Task 4: Hyperparameter Tuning and Model Comparison

**Objective:** Tune hyperparameters and compare the performance of different ML models.

**Models Used:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosted Trees (GBT)

**Steps:**
1. Define models and parameter grids.
2. Use `CrossValidator` with 5-fold cross-validation.
3. Print AUC and best parameters for each model.

**Code Explanation:**
- `ParamGridBuilder` and `CrossValidator` used to find best parameters.
- Evaluation done using AUC.

**Sample Output:**
```
=== Model Comparison & Tuning ===
LogisticRegression AUC: 0.7328
DecisionTree AUC: 0.7637
RandomForest AUC: 0.7928
GBTClassifier AUC: 0.7532
Best Model: RandomForest, AUC = 0.7928
```

---

## Summary

This project demonstrates end-to-end churn prediction using PySpark's MLlib. It includes data cleaning, model building, evaluation, feature selection, and model comparison with hyperparameter tuning.

---

## Final Execution

```bash
spark-submit customer-churn-analysis.py
```
