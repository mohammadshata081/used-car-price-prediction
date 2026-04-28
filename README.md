# Used Car Price Predictor 🚗💰

An end-to-end Machine Learning pipeline that predicts used car prices based on a robust dataset of car features. The project implements a scalable and modular software architecture, transitioning from a monolithic Jupyter Notebook to a clean, production-ready Python codebase.

## 🌟 Overview

The objective of this project is to build a highly accurate regression model using XGBoost to predict the price of a used car. The pipeline consists of:
- **Data Preprocessing**: 
  - **Feature Extraction**: Cleaning text units from numeric columns like `Engine` ("1198 cc" -> 1198), `Max Power` ("87 bhp @ 6000 rpm" -> 87), and `Max Torque` ("109 Nm @ 4500 rpm" -> 109).
  - **Handling Missing Values**: Using mean imputation for numerical columns and mode imputation for categorical ones.
  - **Transformation**: Applying a logarithmic transformation to the `Price` target variable to correct right-skewness and improve model stability. Standardizing numerical features and one-hot encoding categorical variables.
- **Model Training & Hyperparameter Tuning**: Utilizing a high-performance **XGBoost Regressor** wrapped in a `TransformedTargetRegressor`. The pipeline leverages **5-Fold Cross Validation** via `RandomizedSearchCV` to automatically find the most robust, efficient hyperparameter combination without overfitting.
- **Evaluation**: Generating robust evaluation metrics (R2, MSE, RMSE, MAE) and saving visual parity plots to intuitively understand the model's predictive accuracy across different price ranges.

## 🗂️ Dataset Details

The dataset `car details v4.csv` contains roughly 2,000 records of used cars. Key features include:
- **Numerical Features**: `Year`, `Kilometer`, `Length`, `Width`, `Height`, `Seating Capacity`, `Fuel Tank Capacity`.
- **Categorical Features**: `Make`, `Fuel Type`, `Transmission`, `Location`, `Color`, `Owner`, `Seller Type`, `Drivetrain`.
- **Mixed/String Features (Cleaned)**: `Engine`, `Max Power`, `Max Torque`.
- **Target Variable**: `Price` (Modeled in original Rupee scale, but logarithmically transformed internally by the algorithm for stability).
- **Dropped Features**: `Model` (removed due to extremely high cardinality).

## 🗺️ Project Steps (Pipeline)

The machine learning pipeline is highly modularized and structured into the following sequential steps to ensure reproducibility and clean code separation:

### 1. Data Ingestion (`src/preprocessing.py`)
- The pipeline starts by reading the raw CSV file (`car details v4.csv`) using `pandas.read_csv`. This acts as the single source of truth for the dataset.

### 2. Data Cleaning (`src/preprocessing.py`)
- **String Parsing:** Several numerical columns were initially stored as strings because they contained units (e.g., "1198 cc", "87 bhp"). A regular expression `r'(\d+\.?\d*)'` is applied to extract the floating-point numerical values from `Engine`, `Max Power`, and `Max Torque`.
- **Dimensionality Reduction:** The `Model` column contains too many unique specific car models, which introduces extreme cardinality and the risk of overfitting. It is proactively dropped from the feature set.
- **Null Handling:** Any rows that lack the target variable (`Price`) are immediately dropped to avoid training on non-existent labels.

### 3. Feature Engineering & Transformation (`src/preprocessing.py` & `src/train.py`)
- **Log Transformation of Target:** The `Price` column in used cars is heavily right-skewed (few very expensive luxury cars). The target is transformed using `np.log()` to stabilize the variance and make the distribution closer to normal, improving the model's ability to learn across different price brackets.
- **Data Splitting:** The data is split into an 80% training set and 20% holdout test set using a fixed `random_state` for perfect reproducibility.
- **Scikit-Learn ColumnTransformer:** A robust preprocessing pipeline is established to avoid data leakage:
  - **Numerical Pipeline:** Missing numerical values are filled using `SimpleImputer(strategy='mean')`, and then scaled using `StandardScaler` to ensure all numerical inputs are on the same magnitude.
  - **Categorical Pipeline:** Missing categorical values are filled using `SimpleImputer(strategy='most_frequent')`, followed by `OneHotEncoder(sparse_output=False)` to convert text categories into a machine-readable binary matrix.

### 4. Model Training, CV & Serialization (`src/train.py`)
- **Algorithm Selection:** An `XGBRegressor` is chosen for its superior ability to capture complex, non-linear relationships in tabular data without requiring massive amounts of deep learning infrastructure.
- **5-Fold Cross Validation:** Instead of using hardcoded parameters, a `RandomizedSearchCV` explores a grid of hyperparameters (`learning_rate`, `n_estimators`, `max_depth`, etc.). The training data is split into 5 folds, testing on one while training on four. This guarantees the selected parameters are universally efficient and not memorizing the dataset.
- **Serialization:** Once the best performing model is found during cross validation, both the fitted `ColumnTransformer` (preprocessor) and the `XGBRegressor` (model) are serialized into binary formats and saved to the `models/` directory using `joblib`.

### 5. Evaluation & Outputs (`src/evaluate.py`)
- The holdout test dataset is loaded and transformed using the *already fitted* `ColumnTransformer` to guarantee there is no data leakage from the test set into the training parameters.
- Predictions are generated. The performance is evaluated using **R2** (overall fit), **MSE / RMSE** (penalty for large errors), and **MAE** (average absolute error) directly on the original Rupee scale.
- **Visual Outputs (The Graphs):** The evaluation script generates and saves three beautiful, highly representative charts to the `outputs/` folder:
  1. **`feature_importance.png`**: A bar chart demonstrating exactly which features the XGBoost algorithm relies on most (e.g., Year, Engine, Max Power). This proves the model is learning logical real-world patterns.
  2. **`parity_plot.png`**: This plots the *Actual Price* vs *Predicted Price* using Logarithmic Axes. Because car prices span from ₹100k to ₹50M+, a linear graph clumps everything together, hiding the true performance. The Log-Log scale cleanly visualizes how well the predictions hug the true identity line across all price brackets.
  3. **`residual_plot.png`**: This plots the *Predicted Price* vs the *Residual Error*. It perfectly highlights how the variance in error scales as the price of the car increases (showing natural heteroscedasticity for ultra-luxury models).


## 📂 Project Structure

```text
used-car-price-prediction project/
│
├── car details v4.csv               # Raw dataset
├── main.py                          # Master orchestration script
│
├── src/                             # Source code
│   ├── preprocessing.py             # Data loading, cleaning, and Scikit-Learn pipelines
│   ├── train.py                     # Data splitting, model training, and serialization
│   └── evaluate.py                  # Evaluation metrics and parity plot generation
│
├── models/                          # Generated directory for saving joblib artifacts
│   ├── preprocessor.joblib          # Saved ColumnTransformer pipeline
│   └── xgb_model.joblib             # Saved XGBoost regression model (Wrapped with TransformedTargetRegressor)
│
└── outputs/                         # Generated directory for saving evaluation charts
    ├── feature_importance.png       # Top 15 Feature Importances
    ├── parity_plot.png              # Actual vs Predicted (Log-Log Scale)
    └── residual_plot.png            # Predicted vs Residual Errors
```

## 🛠️ Installation

1. Ensure you have Python 3.8+ installed.
2. Clone or download this repository.
3. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```

## 🚀 Usage

To execute the full machine learning pipeline (Data Cleaning -> Training -> Evaluation), simply run the `main.py` script from the project root:

```bash
python main.py
```

### What happens when you run `main.py`?
1. The raw dataset (`car details v4.csv`) is loaded and cleaned.
2. The data is split into an 80/20 train-test ratio.
3. A preprocessing pipeline (`ColumnTransformer`) fits and transforms the data.
4. An `XGBRegressor` trains on the processed data.
5. The preprocessor and model are serialized to the `models/` folder.
6. The model is evaluated against the test set, outputting performance metrics to the console.
7. Scatter plots comparing actual vs. predicted prices are saved to the `outputs/` folder.

## 📊 Results & Performance

The XGBoost model demonstrates excellent, highly robust performance on the holdout test set with the following metrics (evaluated directly on the original Rupee scale):

- **R² Score:** `~0.881` (Explains ~88.1% of the variance in raw used car prices, generalized robustly via CV)
- **RMSE:** `~816,300` (Rupees)
- **MAE:** `~241,361` (Rupees)

## 🧪 Test Cases (Inference)

You can easily use the saved model and preprocessor to perform inference on new or existing car data. Here is an example of what the model predicts for a few sample cars compared to their actual price:

| Car Make & Year | Actual Price (₹) | Predicted Price (₹) |
| --- | --- | --- |
| **Hyundai (2011)** | 220,000 | ~ 217,741 |
| **Honda (2017)** | 505,000 | ~ 561,434 |
| **Maruti Suzuki (2014)** | 450,000 | ~ 389,209 |


## 💻 Technology Stack

- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Visualization**: `matplotlib`
- **Model Serialization**: `joblib`
