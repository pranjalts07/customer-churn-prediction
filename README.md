# 🎯 ChurnSense: Production Churn Prediction System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Azure](https://img.shields.io/badge/Azure-App%20Service-0078D4?logo=microsoft-azure)](https://azure.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Identify which telecom customers are about to leave.** Machine learning model predicts churn, prioritizes outreach, and calculates ROI for retention campaigns.

**Live Demo:** [https://churn-prediction.azurewebsites.net/](https://churn-prediction.azurewebsites.net/) 🚀

---

## 📊 Dashboard Preview

### Campaign Overview
![Dashboard Overview](https://via.placeholder.com/900x500?text=ChurnSense+Dashboard+Overview)

### Risk Analysis & Insights
![Risk Analysis](https://via.placeholder.com/900x500?text=Risk+Distribution+%26+Customer+Insights)

---

## ⚡ Quick Start

### 1️⃣ Clone & Install
```bash
git clone https://github.com/pranjalts07/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### 2️⃣ Run the API
```bash
# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Visit dashboard
open http://localhost:8000
```

### 3️⃣ Test Predictions
```bash
# Health check
curl http://localhost:8000/health

# Score a customer
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0,
    "SeniorCitizen": 0,
    "Contract": 0
  }'

# Get portfolio insights
curl http://localhost:8000/insights/global
```

---

## 📈 Key Metrics

| Metric | Value |
|--------|-------|
| **Customers Analyzed** | 7,043 |
| **High-Risk Identified** | 54 (3.1%) |
| **Revenue at Risk** | **$81,197** |
| **Potential Annual Saves** | **$24,359+** |
| **Model Accuracy (AUC-ROC)** | 0.79 |
| **Model Precision (AUC-PR)** | 0.23 |

---

## 🤖 How It Works

### The Problem
Telecom companies lose customers daily. How to identify who's leaving **before** they churn?

### The Solution
1. **Predict** — Random Forest classifier scores each customer (0-1 churn probability)
2. **Prioritize** — Cost-weighted threshold (0.23) identifies 54 high-risk customers
3. **Act** — Dashboard ranks customers by revenue exposure + risk tier
4. **Measure** — ROI calculator estimates campaign value

### Model Architecture
```
Input (23 Features)
  ↓
Random Forest (100-500 trees)
  ↓
Probability Calibration
  ↓
Cost-Weighted Threshold (0.23)
  ↓
Risk Tier: CRITICAL | AT-RISK | WATCH | SAFE
```

---

## 🎨 Dashboard Features

### Risk Tiers
- 🔴 **CRITICAL** (15 customers) — Immediate action required
- 🟠 **AT-RISK** (39 customers) — High priority outreach
- 🔵 **WATCH** (477 customers) — Monitor closely
- 🟢 **SAFE** (1,230 customers) — Low churn risk

### Tools
- **Risk Analysis** — Drill into customer segments by risk tier
- **Revenue Impact** — See total exposure and opportunity
- **ROI Calculator** — Estimate campaign savings before investing
- **Customer Lookup** — Score any customer in real-time
- **Retention Playbook** — Recommended actions by churn reason

---

## 📋 API Endpoints

```
GET  /health                    Health check
GET  /                          Interactive dashboard
POST /predict                   Score a customer
GET  /insights/global           Portfolio statistics
GET  /insights/all-customers    List at-risk customers (with filters)
GET  /insights/roi              Calculate campaign ROI
```

**[📖 Full API Docs](docs/API.md)**

---

## 📁 Project Structure

```
churnsense/
├── app/
│   ├── main.py                 # FastAPI server + all endpoints
│   ├── templates/
│   │   └── index_fixed.html    # Interactive dashboard UI
│   └── ...
│
├── src/
│   ├── config.py               # Constants (costs, rates, thresholds)
│   ├── data_pipeline.py        # Data loading & preprocessing
│   ├── train.py                # Model training with SMOTE
│   ├── evaluate.py             # Performance metrics & threshold tuning
│   ├── calibration.py          # Probability calibration analysis
│   └── fairness.py             # Demographic parity audit
│
├── models/
│   ├── churnsense_v1.pkl       # Trained Random Forest + scaler
│   ├── model_card.json         # Model metadata & performance
│   └── plots/                  # 14 SHAP, ROC, calibration plots
│
├── data/
│   ├── raw/telco_churn.csv     # Original IBM dataset (7,043 rows)
│   └── processed/
│       ├── train.parquet       # Training set (5,282 rows, 32.8% churn)
│       └── test.parquet        # Test set (1,761 rows, 7.8% churn)
│
├── reports/
│   ├── priority_contact_list.csv    # Ranked customer list
│   └── churnsense_business_report.md
│
├── requirements.txt            # Python dependencies
├── startup.sh                  # Run the server
└── README.md                   # This file
```

---

## 🚀 Deployment

### Local Development
```bash
./startup.sh
# API running at http://localhost:8000
```

### Azure App Service
```bash
# Package
zip -r churn-deploy.zip app/ models/ data/ src/ requirements.txt

# Deploy
az webapp deployment source config-zip \
  --resource-group <group> \
  --name <app> \
  --src churn-deploy.zip

# Verify
curl https://your-app.azurewebsites.net/health
```

### Docker
```bash
docker build -t churnsense:latest .
docker run -p 8000:8000 churnsense:latest
```

---

## 📊 Model Details

### Why Random Forest?
- ✅ **Better Calibration** — Probabilities reflect real churn rates
- ✅ **Threshold Robust** — Handles cost-weighted optimization better than LR
- ✅ **Feature Interactions** — Captures contract × price dynamics
- ✅ **Imbalance Friendly** — Works with SMOTE (32.8% churn in training)

### Key Features (23 Total)
| Feature | Importance | Signal |
|---------|-----------|--------|
| `is_month_to_month` | ⭐⭐⭐ | 3-10x churn multiplier |
| `monthly_charge_per_tenure` | ⭐⭐⭐ | Price acceptance proxy |
| `tenure` | ⭐⭐⭐ | Stability signal (longer = safer) |
| `is_honeymoon_period` | ⭐⭐ | First 6 months = 3x churn risk |
| `service_bundle_depth` | ⭐⭐ | Switching cost (more = stickier) |

**[See all 23 features →](models/model_card.json)**

### Performance

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **AUC-ROC** | 0.79 | Ranks customers by churn risk with 79% accuracy |
| **AUC-PR** | 0.23 | Honest metric for imbalanced data (7.8% churn) |
| **Threshold** | 0.23 | Cost-weighted optimization: catches 65% of churners |
| **Precision** | 46% | 46% of flagged customers actually churn |
| **Recall** | 65% | Catches 2/3 of future churners |

**Calibration Warning:** Model is overconfident at high probabilities (ECE=0.327). Use with caution for individual scores; better for ranking.

---

## 💰 Business Impact

### By the Numbers
- **54 customers** at high risk of churning
- **$81,197** in annual revenue exposure (11.6% of portfolio)
- **$24,359** potential saves @ 30% retention success rate
- **5.2x ROI** on $25/customer outreach cost

### Use Cases
1. **Proactive Outreach** — Contact top 54 customers with retention offers
2. **Upsell Campaigns** — Target "WATCH" tier with bundle upgrades
3. **Budget Planning** — ROI calculator estimates campaign value
4. **A/B Testing** — Compare retention rates vs. control group
5. **Churn Reason Analysis** — Focus on contract/price/support gaps

---

## 🔄 Training Your Own Model

### 1. Load & Explore Data
```bash
python src/data_pipeline.py
```

### 2. Train Model (with SMOTE + Optuna)
```bash
python src/train.py
```

### 3. Evaluate & Optimize
```bash
python src/evaluate.py
```

### 4. Audit for Fairness
```bash
python src/fairness.py
```

---

## 📚 Documentation

- **[AUDIT_SUMMARY.md](AUDIT_SUMMARY.md)** — Fairness audit results
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** — Technical deep-dive
- **[model_card.json](models/model_card.json)** — Full model metadata
- **[API.md](docs/API.md)** — Endpoint reference

---

## 🛡️ Known Limitations

| Issue | Impact | Mitigation |
|-------|--------|-----------|
| **Model Calibration** | Overconfident at high probabilities | Use for ranking, not absolute scores |
| **Distribution Shift** | Train: 32.8% churn, Test: 7.8% churn | Expected; reflects real-world scenario |
| **CLV Model** | Simple 24-month window | Plan for beyond 2-year horizon |
| **Feature Drift** | Contract mix may change | Retrain quarterly with new data |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_api.py::test_predict -v
```

**Coverage:**
- ✅ API endpoints (health, predict, insights)
- ✅ Model loading & inference
- ✅ Data pipeline
- ✅ Feature engineering

---

## 🗺️ Roadmap

- [ ] **Phase 1 (Month 1):** Shadow deployment — log predictions, analyze accuracy
- [ ] **Phase 2 (Months 2-4):** A/B testing — measure actual retention uplift
- [ ] **Phase 3 (Month 5+):** Full production — scale to all at-risk customers
- [ ] **Phase 4:** Real-time dashboards & automated alerts
- [ ] **Phase 5:** Multi-segment models (business vs. residential)

---

## 📞 Support

- **Found a bug?** [Open an issue](https://github.com/pranjalts07/churn-prediction/issues)
- **Have a question?** [Start a discussion](https://github.com/pranjalts07/churn-prediction/discussions)
- **Want to contribute?** [Submit a PR](https://github.com/pranjalts07/churn-prediction/pulls)

---

## 🛠️ Tech Stack

```
Backend:     FastAPI, Uvicorn, Gunicorn
ML:          scikit-learn, Random Forest, SMOTE, Optuna
Data:        pandas, numpy, pyarrow (Parquet)
Analysis:    SHAP, matplotlib, seaborn
Infrastructure: Azure App Service, Docker
Testing:     pytest, TestClient
```

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Built by

**Pranjal Tariga Suresh**  
[GitHub](https://github.com/pranjalts07) • [LinkedIn](https://linkedin.com/in/pranjalts) • [Email](mailto:pranjalts07@gmail.com)

---

<div align="center">

### Ready to predict churn? 🎯

[![Open Dashboard](https://img.shields.io/badge/Open%20Dashboard-%E2%86%92-black?style=for-the-badge)](https://churn-prediction.azurewebsites.net/)
[![Clone Repo](https://img.shields.io/badge/Clone%20Repo-%E2%86%92-blue?style=for-the-badge)](https://github.com/pranjalts07/customer-churn-prediction)

</div>

---

**Status:** ✅ Production Ready | **Last Updated:** 2026-05-02
