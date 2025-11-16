# 💎 Diamond Price Prediction using XGBoost

This project builds a **machine learning regression model** to predict diamond prices using the popular **Diamonds dataset** from Seaborn.
The workflow includes feature engineering, preprocessing pipelines, model training with XGBoost, and performance evaluation.

---

## Project Overview

The goal of this project is to create a highly accurate model for predicting diamond prices based on attributes such as:

* Carat
* Cut
* Color
* Clarity
* Depth
* Table
* Dimensions (x, y, z)

We use:

* **Feature Engineering**
* **Preprocessing Pipelines (Scaling + OneHotEncoding)**
* **XGBoost Regressor**
* **Model Performance Metrics**
* **Visual Comparison: Actual vs Predicted Prices**

---

##  Machine Learning Workflow

### **1. Load Dataset**

We load the **Seaborn Diamonds** dataset:

```python
df = sns.load_dataset('diamonds')
```

---

### **2. Feature Engineering**

We create new useful features:

* **volume** = x × y × z
* **carat_per_vol** = carat / volume
* Remove invalid rows where dimensions ≤ 0

---

### **3. Preprocessing**

Categorical columns:

```
cut, color, clarity
```

Numerical columns:

```
carat, depth, table, x, y, z, volume, carat_per_vol
```

Techniques used:

* **StandardScaler** for numerics
* **OneHotEncoder** (drop='first') for categories
* Combined using **ColumnTransformer**

---

### **4. Model – XGBoost Regressor**

We use tuned parameters:

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist'
)
```

---

### **5. Model Evaluation**

We compute:

* **R² Score**
* **Mean Absolute Error (MAE)**
* **Root Mean Square Error (RMSE)**

A scatterplot compares **actual vs predicted prices**.

---

##  Results

You will see printed metrics like:

```
R² Score: 0.98+
Mean Absolute Error: ~300
RMSE: ~500
```

(Your exact results may vary slightly depending on random state and environment.)

---

##  Visualization

A scatter plot is generated:

* X-axis → Actual Prices
* Y-axis → Predicted Prices

This helps visualize model accuracy and residual spread.

---

## Running the Project

### **Install dependencies**

```
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

### **Run the script**

```
python diamond_price_prediction.py
```

---

##  Folder Structure

```
📦 diamond-price-xgboost
 ┣ 📜 README.md
 ┣ 📜 diamond_price_prediction.py
 ┗ 📊 (plots generated automatically)
```

---

## 🛠 Technologies Used

* Python
* Pandas & NumPy
* Seaborn & Matplotlib
* Scikit-Learn
* XGBoost
