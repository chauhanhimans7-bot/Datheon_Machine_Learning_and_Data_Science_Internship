# California Housing Price Prediction

A machine learning regression project that predicts the estimated median house value of California districts using geographical, demographic, and housing-related information. The project includes data analysis, preprocessing, model experimentation, hyperparameter tuning, model evaluation, and an interactive Streamlit application.

---

## Project Overview

Housing prices are influenced by several factors, including location, household income, population, housing availability, and proximity to the ocean.

The objective of this project is to build a supervised machine learning regression system that learns relationships between these features and historical California housing values to estimate the median house value for a given district.

The project follows an end-to-end machine learning workflow, starting with raw dataset exploration and ending with an interactive application powered by a trained machine learning model.

**Selected Model:** HistGradientBoostingRegressor

**Application:** Streamlit

**Problem Type:** Supervised Machine Learning — Regression

**Target Variable:** `median_house_value`

---

## Project Objectives

- Understand and explore the California Housing dataset.
- Perform exploratory data analysis (EDA) to discover patterns, relationships, and distributions.
- Identify and handle missing values.
- Prepare numerical and categorical features for machine learning.
- Experiment with different regression algorithms.
- Compare model performance using appropriate evaluation metrics.
- Perform hyperparameter tuning to improve model performance.
- Build a reusable machine learning pipeline.
- Save the trained model using Joblib.
- Develop an interactive Streamlit application for house value prediction.

---

## Dataset Information

**Dataset:** California Housing Dataset

**Source:** [Hands-On Machine Learning — California Housing Dataset](https://github.com/ageron/handson-ml2/blob/master/datasets/housing/housing.csv)

The dataset contains information collected from the 1990 California census. Each record represents a geographical district (census block group), rather than an individual house.

### Dataset Characteristics

| Property | Value |
|---|---|
| Dataset | California Housing |
| Total Records | 20,640 |
| Total Columns | 10 |
| Input Features | 9 |
| Target Variable | `median_house_value` |
| Problem Type | Regression |
| Missing Values | Present in `total_bedrooms` |
| Data Format | CSV |

### Dataset Features

The dataset contains nine input features and one target variable.

| Feature | Description |
|---|---|
| `longitude` | Geographical longitude of the district |
| `latitude` | Geographical latitude of the district |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total number of rooms in the district |
| `total_bedrooms` | Total number of bedrooms in the district |
| `population` | Total population of the district |
| `households` | Number of households in the district |
| `median_income` | Median household income, expressed in tens of thousands of US dollars |
| `ocean_proximity` | Categorical feature describing proximity to the ocean |
| `median_house_value` | Median house value of the district (prediction target) |

### Target Variable

The target variable is:

```python
median_house_value
```

It represents the historical median house value for a geographical district, measured in US dollars.

**Important:** This model estimates district-level median housing values. It does not predict the exact market price of an individual property.

---

## Machine Learning Workflow

The project follows a structured machine learning pipeline.

### 1. Data Loading and Inspection

- Load the California Housing dataset using pandas.
- Inspect dataset dimensions, column names, and data types.
- Examine statistical summaries.
- Identify missing values and duplicate records where applicable.

### 2. Exploratory Data Analysis (EDA)

Exploratory data analysis is performed to understand the dataset before building models.

Key activities include:

- Understanding numerical feature distributions.
- Analyzing the target variable distribution.
- Identifying missing values.
- Studying relationships between numerical features.
- Examining correlations.
- Exploring categorical feature distributions.
- Identifying potential outliers and unusual patterns.

### 3. Data Preprocessing

The dataset contains both numerical and categorical features, so separate preprocessing operations are applied.

**Numerical feature preprocessing:**

- Handle missing values using imputation.
- Apply numerical feature scaling where appropriate.

**Categorical feature preprocessing:**

- Encode `ocean_proximity` using One-Hot Encoding.
- Handle previously unseen categories using `handle_unknown="ignore"`.

The preprocessing steps are incorporated into a scikit-learn `ColumnTransformer` and Pipeline to maintain consistent transformations during training and prediction.

### 4. Train-Test Split

The dataset is divided into training and testing subsets.

| Dataset | Proportion |
|---|---:|
| Training Set | 80% |
| Testing Set | 20% |

The split uses `random_state=42` to make the experiment reproducible.

The training set is used for model training and selection, while the held-out test set is used for final performance evaluation.

### 5. Model Training and Comparison

Multiple regression algorithms are experimented with to identify a suitable model.

Models explored include:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- HistGradientBoostingRegressor

The models are evaluated using regression performance metrics, allowing their predictive performance to be compared.

### 6. Hyperparameter Tuning

GridSearchCV is used to identify suitable hyperparameters for the HistGradientBoostingRegressor.

The tuning process explores:

| Hyperparameter | Selected Value |
|---|---|
| `learning_rate` | 0.1 |
| `max_depth` | None |
| `max_leaf_nodes` | 63 |
| `min_samples_leaf` | 20 |
| `l2_regularization` | 0.1 |

Cross-validation is used during model selection to obtain a more reliable estimate of model performance.

### 7. Final Model Selection

Based on the experimentation and evaluation performed in the notebook, **HistGradientBoostingRegressor** is selected as the final model.

The selected estimator is integrated with the fitted preprocessing pipeline so that incoming raw input features receive the same transformations used during training.

---

## Model Performance

The final model is evaluated on the held-out test dataset.

### Evaluation Metrics

| Metric | Score |
|---|---:|
| Mean Absolute Error (MAE) | $31,174.17 |
| Root Mean Squared Error (RMSE) | $47,400.66 |
| R² Score | 0.829 |

### Understanding the Results

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between actual and predicted values.

A test MAE of approximately $31,174 means that the average absolute prediction error on the test set is about $31,174.

**Root Mean Squared Error (RMSE)**

RMSE measures prediction error while penalizing larger errors more heavily than MAE.

The test RMSE is approximately $47,401.

**R² Score**

R² measures how much of the variation in the target variable is explained by the model on the test dataset.

An R² score of 0.829 indicates that the model explains approximately 82.9% of the variation in the test-set target values.

These are the recorded results from the notebook's final held-out test evaluation. They do not guarantee the accuracy of every individual prediction.

---

## Interactive Web Application

The project includes an interactive web application built using **Streamlit**.

The application allows users to enter housing-related information and receive a predicted median house value without manually executing the notebook.

### Application Features

- Interactive input fields for all nine model features.
- Numerical inputs for geographical, demographic, and housing information.
- Categorical selection for ocean proximity.
- Loading of the saved machine learning model.
- Prediction using the fitted preprocessing and model pipeline.
- Display of estimated median house value in US dollars.

### Application Inputs

| Input | Expected Information |
|---|---|
| Longitude | Geographical longitude |
| Latitude | Geographical latitude |
| Housing Median Age | Median housing age |
| Total Rooms | Number of rooms |
| Total Bedrooms | Number of bedrooms |
| Population | District population |
| Households | Number of households |
| Median Income | Income in tens of thousands of USD |
| Ocean Proximity | Categorical ocean-proximity value |

The application uses the same nine original input features expected by the trained model.

---

## Project Structure

```text
Task_2_Housing_Price_Prediction/
│
├── Housing_Price_Prediction.ipynb
│
├── app.py
│
├── housing_model_pred.joblib
│
├── requirements.txt
│
└── README.md
```

| File | Purpose |
|---|---|
| `Housing_Price_Prediction.ipynb` | Complete data analysis, preprocessing, model experimentation, tuning, and evaluation |
| `app.py` | Streamlit application for interactive predictions |
| `housing_model_pred.joblib` | Saved machine learning model artifact |
| `requirements.txt` | Python dependencies required to run the application |
| `README.md` | Project documentation |

---

## Installation and Setup

Follow the instructions below to run the application locally.

### Prerequisites

- Python installed on your system.
- Git installed on your system.
- Access to the project files.
- Internet connection for downloading the dataset when the local CSV fallback is not used.

### 1. Clone the Repository

```bash
git clone https://github.com/chauhanhimans7-bot/Datheon_Machine_Learning_and_Data_Science_Internship.git
```

### 2. Navigate to the Project Directory

```bash
cd Datheon_Machine_Learning_and_Data_Science_Internship/Task_2_Housing_Price_Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

After execution, Streamlit will provide a local URL in the terminal. Open that URL in your browser to access the application.

---

## Dependencies

The project uses the following primary libraries:

| Library | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Dataset manipulation and analysis |
| NumPy | Numerical operations |
| scikit-learn | Data preprocessing, model training, and evaluation |
| Joblib | Model serialization and loading |
| Streamlit | Interactive web application |

The exact dependency versions are specified in `requirements.txt`.

### Model Compatibility Note

The saved model artifact was created using scikit-learn 1.7.2. The included requirements file pins this version.

Scikit-learn does not guarantee compatibility when loading serialized models across different library versions. Use the matching training version when loading the artifact, or retrain and save the model using the target environment's version.

---

## Limitations and Considerations

- The dataset represents California housing conditions from the 1990 census and may not reflect current housing-market conditions.
- The observations represent geographical districts, not individual properties.
- The original target variable has an upper cap of $500,001, which limits the model's ability to distinguish higher-valued districts beyond that cap.
- Predictions depend on the quality and distribution of the training data.
- Model performance on unseen data may differ from the reported test results.
- This application is an educational machine learning project, not a professional property valuation service.

---

## Future Improvements

Potential directions for further development include:

- Experimenting with additional ensemble and boosting algorithms.
- Conducting more extensive feature engineering.
- Performing deeper model diagnostics and error analysis.
- Improving prediction validation on geographically distinct data.
- Enhancing the Streamlit interface and input validation.
- Deploying the application to a public hosting platform.
- Monitoring model performance and evaluating newer housing datasets.

---

## Key Learnings

This project provides practical experience with:

- End-to-end regression problem solving.
- Exploratory data analysis and data visualization.
- Numerical and categorical feature preprocessing.
- Scikit-learn pipelines and ColumnTransformer.
- Regression model comparison and evaluation.
- Cross-validation and hyperparameter tuning.
- Model persistence using Joblib.
- Building an machine learning application using Streamlit.

---

## Author

**Dev Chauhan**

Machine Learning & Data Science Intern  
Datheon — Batch 01

---

## Disclaimer

This project was developed for educational and internship purposes as part of the Datheon Machine Learning & Data Science Internship.

The predictions are estimates based on historical district-level census data. They should not be treated as current property valuations, financial advice, or guaranteed market prices.
