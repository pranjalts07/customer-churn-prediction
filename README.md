# Customer Churn Prediction System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This is a production ready customer churn prediction system for telecom retention teams. It predicts churn probability, ranks customers by revenue exposure, recommends retention actions, and estimates campaign return on investment.

## Dashboard Preview

### Landing Page

<p align="center">
  <img src="Images/Landing%20page.png" alt="Customer churn prediction landing page" width="900">
</p>

### Campaign Overview

<p align="center">
  <img src="Images/Overview.png" alt="Customer churn prediction campaign overview dashboard" width="900">
</p>

### Risk Analysis

<p align="center">
  <img src="Images/Risk%20Analysis.png" alt="Customer churn prediction risk analysis dashboard" width="900">
</p>

### Revenue Impact

<p align="center">
  <img src="Images/Revenue%20Impact.png" alt="Customer churn prediction revenue impact dashboard" width="900">
</p>

### Customer Lookup

<p align="center">
  <img src="Images/Customer%20Lookup.png" alt="Customer churn prediction customer lookup dashboard" width="900">
</p>

## What This Project Does

This project helps answer three business questions:

1. Which customers are most likely to churn?
2. Which customers should the retention team contact first?
3. How much revenue can a retention campaign protect?

The system combines a trained machine learning model, a FastAPI backend, an interactive dashboard, and business focused reporting.

## Key Results

| Metric | Value |
| --- | --- |
| Customers analyzed | 7,043 |
| High risk customers identified | 54 |
| Revenue at risk | $81,197 |
| Potential saves | $24,359 |
| AUC ROC | 0.79 |
| AUC PR | 0.23 |
| Cost optimized threshold | 0.23 |

## Main Features

| Feature | Description |
| --- | --- |
| Churn prediction API | Scores individual customers using a trained Random Forest model |
| Portfolio dashboard | Shows customer risk, revenue exposure, and campaign metrics |
| Risk segmentation | Groups customers into critical, high, medium, and low risk tiers |
| Revenue impact analysis | Estimates exposed revenue and expected retention value |
| ROI calculator | Compares outreach cost against expected saved revenue |
| Customer lookup | Lets users test churn risk for a single customer profile |
| Model reporting | Includes model metadata, plots, and priority contact list |

## Quick Start

### Clone And Install

```bash
git clone https://github.com/pranjalts07/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### Run Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open the dashboard:

```bash
http://localhost:8000
```

### Test The API

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
    "SeniorCitizen": 0
  }'
```

```bash
curl http://localhost:8000/insights/global
```

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Interactive dashboard |
| GET | `/health` | Service and model health check |
| POST | `/predict` | Predict churn for one customer |
| GET | `/insights/global` | Portfolio level statistics |
| GET | `/insights/all-customers` | Ranked customer risk list |
| GET | `/insights/roi` | Retention campaign ROI calculation |

## Model Approach

The model is built to support retention prioritization rather than only raw accuracy. It uses engineered customer features, a Random Forest classifier, probability calibration analysis, and a cost optimized decision threshold.

Core workflow:

```text
Customer data
Feature engineering
Random Forest classifier
Calibrated churn probability
Cost optimized threshold
Risk tier and retention recommendation
```

Important features include:

| Feature | Business meaning |
| --- | --- |
| `is_month_to_month` | Month to month customers have lower commitment |
| `monthly_charge_per_tenure` | High cost relative to tenure can signal price pressure |
| `tenure` | Longer tenure usually means higher stability |
| `is_honeymoon_period` | Early lifecycle customers churn more often |
| `service_bundle_depth` | More bundled services increase switching friction |

## Project Structure

```text
customer-churn-prediction/
  app/
    main.py
    templates/
      index.html
  data/
    raw/
      telco_churn.csv
    processed/
      train.parquet
      test.parquet
  Images/
    Landing page.png
    Overview.png
    Risk Analysis.png
    Revenue Impact.png
    Customer Lookup.png
  models/
    trained model artifact
    model_card.json
    plots/
  reports/
    priority_contact_list.csv
  src/
    calibration.py
    config.py
    data_pipeline.py
    evaluate.py
    fairness.py
    train.py
  tests/
    test_api.py
  requirements.txt
  startup.sh
  README.md
```

## Training Workflow

Run the data pipeline:

```bash
python src/data_pipeline.py
```

Train the model:

```bash
python src/train.py
```

Evaluate model performance:

```bash
python src/evaluate.py
```

Run fairness checks:

```bash
python src/fairness.py
```

## Testing

```bash
python -m pytest
```

The test suite covers the FastAPI health check, dashboard route, global insights endpoint, and prediction endpoint.

## Run The Application

### Local Startup Script

```bash
./startup.sh
```

## Known Limitations

| Limitation | Note |
| --- | --- |
| Probability calibration | Use high probability scores for ranking and review rather than as exact probabilities |
| Dataset size | The project uses the IBM Telco churn dataset |
| CLV estimate | Revenue impact uses a simplified 24 month value window |
| Model drift | Retrain periodically when customer behavior or product pricing changes |

## Tech Stack

| Area | Tools |
| --- | --- |
| Backend | FastAPI, Uvicorn |
| Machine learning | scikit-learn, imbalanced-learn, Random Forest |
| Data | pandas, numpy, pyarrow |
| Analysis | SHAP, matplotlib, seaborn |
| Testing | pytest, FastAPI TestClient |

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Pranjal Tariga Suresh

[GitHub](https://github.com/pranjalts07)
