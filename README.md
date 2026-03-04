# Customer Churn Prediction

End-to-end machine learning workflow for predicting customer churn using structured feature engineering and supervised classification models.

---

## Business Problem

Customer churn directly impacts revenue and long-term growth.  
The objective of this project is to build a predictive model that identifies customers likely to churn so retention strategies can be applied proactively.

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
├── data/                  # Raw dataset
├── notebooks/             # Exploratory data analysis
├── src/
│   └── modeling/          # Model training scripts
├── requirements.txt       # Dependencies
└── README.md
```

---

## Exploratory Data Analysis

Key steps performed:

- Missing value inspection
- Data type corrections
- Distribution analysis
- Churn rate breakdown
- Correlation analysis

EDA notebook:

```
notebooks/01_eda.ipynb
```

---

## Modeling Pipeline

The modeling workflow includes:

- Data preprocessing
- Categorical encoding
- Train/test split
- Baseline model training
- Performance evaluation

Training script:

```
src/modeling/train_model.py
```

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Performance metrics will be updated as models are iterated.

---

## How to Run

Clone the repository:

```
git clone https://github.com/dylanmartin1130/customer-churn-prediction.git
cd customer-churn-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the training script:

```
python src/modeling/train_model.py
```

---

## Future Improvements

- Cross-validation
- Hyperparameter tuning
- Feature importance analysis
- Model serialization
- Pipeline automation
- Deployment-ready structure

---

## Status

Project in active development with structured iteration and version control.
