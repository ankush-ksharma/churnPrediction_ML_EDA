# Telco Customer Churn Prediction

End-to-end machine learning project to predict customer churn using a Decision Tree Classifier, served via a FastAPI REST API.

## Project Structure

```
app.py                                     # FastAPI app exposing the /predict endpoint
requirements.txt                           # Python dependencies
sample_request.json                        # Example request payload for /predict
data/
  TelcoCustomerChurn.csv                   # Raw dataset
  TelcoCustomerChurn - Data Dictionary.csv # Column descriptions
model/
  churn_pipeline.pkl                       # Trained scikit-learn pipeline (preprocessing + model)
notebook/
  Exploratory_Data_Analysis.ipynb          # EDA, feature engineering, and model training/tuning
  Analysis.md                              # Written walkthrough of the notebook
```

## Setup Instructions

1. **Clone and set up a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server**

   ```bash
   python app.py
   ```

   The server starts at `http://localhost:8000`.

## Model

The model is a Decision Tree Classifier trained on the [Telco Customer Churn dataset](data/TelcoCustomerChurn.csv). Details on data cleaning, EDA, feature engineering, and model selection are documented in [notebook/Analysis.md](notebook/Analysis.md).

Evaluation is done using f1 score as there are class imbalance so accuracy will not be correct metrics.


## Making Predictions

Send a POST request to `/predict` with customer details as JSON (see [sample_request.json](sample_request.json) for the expected fields):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample_request.json            
```

Example response:

```json
{
  "prediction":"No",
  "churn_probability":0.0108
  }
```
### Sample Data

# Sample Curl Request
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}'
```

# Response
```json
{
  "prediction": "Yes",
  "churn_probability": 0.543
}
```


### Bonus
I have done Hyperparamerter tuning, handled class imbalance in the decision tress, copared various model (each including their own hyberparameter traning)
