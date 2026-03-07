# Customer Churn Prediction

End-to-end machine learning workflow for predicting telecom customer churn using structured preprocessing, supervised classification, cross-validation, threshold optimization, and interpretable feature analysis.

---

## Business Problem

Customer churn directly impacts revenue and long-term growth.

The objective of this project is to build a predictive system that identifies customers likely to churn so proactive retention strategies can be applied.

Rather than relying on the default 0.50 classification threshold, this project optimizes the decision boundary based on F1 score to balance precision and recall according to business tradeoffs.

---

## Dataset

- Telco Customer Churn dataset  
- ~7,000 customer records  
- Mix of categorical and numerical features  
- Target variable: `Churn` (Yes/No)

---

## Project Structure

```
customer-churn-prediction/
│
├── data/                      # Raw dataset
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory data analysis
├── src/
│   └── modeling/
│       └── train_model.py     # Training and evaluation pipeline
├── artifacts/
│   └── roc_curve.png          # Saved ROC curve
├── requirements.txt
└── README.md
```

## Exploratory Data Analysis

Key steps performed:

- Missing value inspection
- Data type corrections
- Distribution analysis
- Churn segmentation by contract type
- Correlation analysis

EDA notebook:


notebooks/01_eda.ipynb


---

## Modeling Pipeline

The modeling workflow includes:

- Categorical encoding (OneHotEncoder)
- Numerical scaling (StandardScaler)
- Class imbalance handling (`class_weight="balanced"`)
- Stratified train/test split
- Logistic Regression classifier
- Probability-based predictions
- Threshold optimization
- 5-fold cross-validation

Training script:


src/modeling/train_model.py


---

## Model Performance

### Test Set Metrics

- Accuracy: 0.732  
- Precision: 0.497  
- Recall: 0.791  
- F1 Score: 0.611  
- AUC: 0.832  

### Cross-Validation

- Mean 5-fold AUC: 0.845  

### Threshold Optimization

- Default threshold: 0.50  
- Optimized threshold: 0.55  
- Best F1 Score: 0.624  

This demonstrates the importance of business-aware decision boundary tuning instead of relying on arbitrary defaults.

---

## Key Drivers of Churn

Based on logistic regression coefficients:

- Short tenure strongly increases churn risk
- Month-to-month contracts significantly increase churn probability
- Two-year contracts significantly reduce churn risk
- Fiber optic customers show higher churn likelihood
- Pricing structure influences churn behavior

This provides actionable business insight beyond raw prediction.

---

## How to Run

Clone the repository:


git clone https://github.com/dylanmartin1130/customer-churn-prediction.git

cd customer-churn-prediction


Create a virtual environment (recommended):


python -m venv .venv
source .venv/bin/activate


Install dependencies:


pip install -r requirements.txt


Run the training pipeline:


python src/modeling/train_model.py


ROC curve will be saved to:


artifacts/roc_curve.png


---

## Future Improvements

- Tree-based model comparison (Random Forest / XGBoost)
- SHAP-based explainability
- Hyperparameter tuning
- Cost-sensitive optimization
- Model serialization
- REST API deployment

---

## Status

Complete end-to-end supervised ML pipeline with evaluation rigor, threshold tuning, cross-validation, and interpretable feature analysis.
