# Personality Prediction
## Kaggle Playground Series - Season 5, Episode 7

## 1. Project Overview

Unlocking insights into human personality is a critical endeavor across diverse fields, from personalized recommendations to tailored human resource strategies. This project addresses the Kaggle Playground Series Season 5, Episode 7: "Introvert/Extrovert Prediction" challenge, aiming to accurately classify individuals based on their social behaviors.

## 2. Dataset

The dataset for this competition (`train.csv` and `test.csv`) contains anonymized behavioral data related to individuals' social habits and preferences. Key features include:

* `Time_spent_Alone`: Time (in hours) spent alone.
* `Stage_fear`: Binary (Yes/No) indicator of stage fear.
* `Social_event_attendance`: Number of social events attended.
* `Going_outside`: Frequency of going outside (e.g., 0-7 days a week).
* `Drained_after_socializing`: Binary (Yes/No) indicator of feeling drained after social interaction.
* `Friends_circle_size`: Number of close friends.
* `Post_frequency`: Frequency of social media posts.
* `Personality`: Target variable (Extrovert/Introvert).

## 3. Data Preprocessing

To prepare the data for machine learning models, the following preprocessing steps were performed:

1.  **Handling Missing Values**:
    * For numerical columns (`Time_spent_Alone`, `Social_event_attendance`, `Going_outside`, `Friends_circle_size`, `Post_frequency`), whose mean and median values, based on the parameters, were used to impute missing values.
    * For categorical columns (`Stage_fear`, `Drained_after_socializing`), missing values were imputed with the mode of the respective columns.

2.  **Outlier Treatment & Skewness Adjustment**:
    * Exploratory Data Analysis (EDA) revealed the presence of outliers in numerical features. For `Time_spent_Alone`, which had a skewness of 1.149, a `log1p` transformation was applied to reduce its skewness and mitigate the impact of outliers. Other numerical features did not require extensive outlier or skewness treatment.

3.  **Feature Engineering & Dimensionality Reduction**:
    * **PCA (Principal Component Analysis)**: After handling missing values and addressing skewness, PCA was applied to the numerical features. An `n_components=0.90` was chosen, retaining principal components that explained 90% of the variance in the data. This resulted in a reduction to 3 principal components, effectively reducing dimensionality while preserving most of the information.

4.  **Encoding Categorical Variables**:
    * Binary categorical features were converted into numerical representations:
        * `Stage_fear`: 'No' was mapped to 0, 'Yes' to 1.
        * `Drained_after_socializing`: 'No' was mapped to 0, 'Yes' to 1.
        * `Personality`: 'Introvert' was mapped to 0, 'Extrovert' to 1.

## 4. Model Building & Evaluation

The classification task involved addressing the class imbalance and achieving high accuracy. The following models were explored:

### 4.1. Logistic Regression

* **SMOTE**: `SMOTE (Synthetic Minority Over-sampling Technique)` was applied to the training data to balance the classes. After SMOTE, both 'Extrovert' and 'Introvert' classes were balanced with 9589 samples each in the training data.
* **Results**: The Logistic Regression model showed strong performance. On the training set, it achieved an accuracy of 96% (precision: 0.95, recall: 0.95, F1-score: 0.95). On the test set, it maintained an accuracy of 96% (precision: 0.94, recall: 0.95, F1-score: 0.95).

### 4.2. XGBoost

* **Hyperparameter Tuning**: `RandomizedSearchCV` was employed to find optimal hyperparameters for the `XGBClassifier`. The parameter grid included `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
* **Results**: The tuned XGBoost model demonstrated excellent performance. On the training set (after SMOTE), it achieved an accuracy of 96% (precision: 0.96, recall: 0.96, F1-score: 0.96). On the test set, it maintained an accuracy of 96% (precision: 0.95, recall: 0.96, F1-score: 0.95).

### 4.3. Random Forest

* **Hyperparameter Tuning**: Both `RandomizedSearchCV` and `GridSearchCV` were used to fine-tune `RandomForestClassifier` hyperparameters.
* **Results**: The Random Forest model, especially after hyperparameter tuning with `GridSearchCV` (best score of 97.84% accuracy), provided very high accuracy and F1-scores, making it a strong contender. On the training set, it achieved an accuracy of 97% (precision: 0.96, recall: 0.96, F1-score: 0.96). On the test set, it maintained an accuracy of 97% (precision: 0.96, recall: 0.96, F1-score: 0.96).

## 5. Final Model Selection

Based on the evaluation metrics, the **Random Forest Classifier** with the following hyperparameters (derived from GridSearch) was selected as the best performing model for the final submission:

* `max_depth`: 9
* `n_estimators`: 50
* `class_weight`: 'balanced'
* `random_state`: 10

This model consistently delivered high precision, recall, and F1-scores on unseen data, indicating robust generalization capabilities.

## 6. Submission

The final predictions were generated using the best-performing Random Forest model on the preprocessed test data. The predictions were then mapped back to 'Extrovert' (1) and 'Introvert' (0) labels and saved in the `submission.csv` format as required by the Kaggle competition.

**Accuracy achieved on Kaggle**: 97.32%

## 7. Usage

To replicate this project:

1.  Clone this repository.
2.  Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s5e7).
3.  Run the Jupyter notebook (`intro_extro.ipynb`) step-by-step to execute the data preprocessing, EDA, model training, and prediction pipeline.    
