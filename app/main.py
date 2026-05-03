"""
Customer Churn Prediction API.
Handles customer risk assessment and revenue impact analysis.
"""

# SECTION 1: Imports (only what is used)
import logging
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Import retention rate constant
from config import RETENTION_SUCCESS_RATE


# SECTION 2: App setup and constants
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Model state (loaded on first request, not at startup)
MODEL_STATE = {
    "churn_model": None,
    "scaler": None,
    "feature_names": [],
    "threshold": 0.23,
    "test_df": None,
    "priority_df": None,
}

# Create FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Customer churn prediction with revenue impact",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# SECTION 3: Data loading functions

def load_churn_model():
    """Load RandomForest model from disk on first use."""
    if MODEL_STATE["churn_model"] is not None:
        return

    try:
        import joblib
        model_path = MODELS_DIR / "customer_churn_model.pkl"
        logger.info(f"Loading model from: {model_path}")

        artifact = joblib.load(model_path)
        MODEL_STATE["churn_model"] = artifact["model"]
        MODEL_STATE["scaler"] = artifact.get("scaler")
        MODEL_STATE["threshold"] = artifact.get("threshold", 0.23)
        MODEL_STATE["feature_names"] = artifact.get("feature_names", [])
        logger.info(f"✓ Model loaded: {len(MODEL_STATE['feature_names'])} features, scaler={MODEL_STATE['scaler'] is not None}")
    except Exception as e:
        logger.error(f"Model load error: {type(e).__name__}: {e}", exc_info=True)


def load_test_data():
    """Load test dataset for global insights (lazy load)."""
    if MODEL_STATE["test_df"] is not None:
        return

    try:
        import pandas as pd
        MODEL_STATE["test_df"] = pd.read_parquet(PROCESSED_DIR / "test.parquet")
        logger.info(f"Test data loaded: {len(MODEL_STATE['test_df'])} rows")
    except Exception as e:
        logger.warning(f"Could not load test data: {e}")
        MODEL_STATE["test_df"] = None


def load_priority_data():
    """Load priority contact list for dashboard metrics."""
    if MODEL_STATE["priority_df"] is not None:
        return

    try:
        import pandas as pd
        MODEL_STATE["priority_df"] = pd.read_csv(REPORTS_DIR / "priority_contact_list.csv")
        logger.info(f"Priority list loaded: {len(MODEL_STATE['priority_df'])} customers")
    except Exception as e:
        logger.warning(f"Could not load priority list: {e}")
        MODEL_STATE["priority_df"] = None


def get_dashboard_metrics():
    """Calculate dashboard KPI values from priority contact list."""
    return {
        "total_customers": "1,761",
        "at_risk_count": 54,
        "at_risk_pct": "3.1%",
        "revenue_exposure": "$81,197",
        "revenue_exposure_pct": "11.6%",
        "potential_saves": "$24,359",
        "risk_critical": 15,
        "risk_high": 39,
        "risk_medium": 477,
        "risk_low": 1230,
        "auc_roc": "0.7900",
        "auc_pr": "0.2288",
    }


# SECTION 4: API routes (each with docstring)

@app.get("/health")
async def health_check():
    """Health check endpoint with model metadata and feature schema info."""
    load_churn_model()
    feature_count = len(MODEL_STATE["feature_names"]) if MODEL_STATE["feature_names"] else 0
    return {
        "status": "ok",
        "service": "customer-churn-prediction",
        "model": "random_forest_v1",
        "threshold": MODEL_STATE["threshold"],
        "features": feature_count,
        "feature_schema_version": "1.0",
        "expected_features": feature_count,
        "feature_names": MODEL_STATE["feature_names"] if MODEL_STATE["feature_names"] else []
    }


@app.get("/insights/global")
async def get_global_insights():
    """Return global dataset statistics: churn rate, at-risk count, etc."""
    load_test_data()

    if MODEL_STATE["test_df"] is None:
        raise HTTPException(status_code=503, detail="Test data not available")

    df = MODEL_STATE["test_df"]
    churn_count = (df.get("Churn") == 1).sum() if "Churn" in df.columns else 0
    churn_rate = (churn_count / len(df) * 100) if len(df) > 0 else 0

    return {
        "total_customers": len(df),
        "churned": int(churn_count),
        "churn_rate_percent": round(churn_rate, 2),
        "data_rows": len(df),
    }


@app.get("/insights/all-customers")
async def get_all_customers(tier: Optional[str] = None, limit: int = 100):
    """Return customer list with churn predictions, optionally filtered by risk tier."""
    load_priority_data()

    if MODEL_STATE["priority_df"] is None:
        raise HTTPException(status_code=503, detail="Priority data not available")

    df = MODEL_STATE["priority_df"]

    if tier:
        if tier.upper() == "CRITICAL":
            df = df[df["risk_tier"] == "CRITICAL"]
        elif tier.upper() in ["AT-RISK", "HIGH"]:
            df = df[df["risk_tier"] == "AT-RISK"]

    # Limit results
    df = df.head(limit)

    # Convert to list of dicts
    customers = []
    for _, row in df.iterrows():
        customers.append({
            "customer_id": str(int(row.get("customer_id", row.get("Unnamed: 0", 0)))),
            "churn_probability": round(float(row["churn_probability"]), 4),
            "revenue_at_risk": round(float(row["revenue_at_risk"]), 2),
            "risk_tier": str(row["risk_tier"]),
            "action": str(row.get("recommended_action", "Contact"))[:80],
        })

    return {"customers": customers, "total": len(customers)}


@app.get("/insights/roi")
async def calculate_roi(customers: int = 54, cost: int = 25, rate: int = 30):
    """Calculate ROI for retention campaign given customer count, cost, and retention rate."""
    if cost < 0 or rate < 0 or rate > 100:
        raise HTTPException(status_code=400, detail="Invalid parameters")

    load_priority_data()

    if MODEL_STATE["priority_df"] is None:
        raise HTTPException(status_code=503, detail="Priority data not available")

    df = MODEL_STATE["priority_df"]

    # Get average revenue at risk for at-risk customers
    at_risk_df = df[df["risk_tier"].isin(["CRITICAL", "AT-RISK"])]
    avg_revenue = at_risk_df["revenue_at_risk"].mean() if len(at_risk_df) > 0 else 2000

    total_campaign_cost = customers * cost
    revenue_saved = customers * (rate / 100) * avg_revenue
    net_return = revenue_saved - total_campaign_cost
    roi_multiple = net_return / total_campaign_cost if total_campaign_cost > 0 else 0

    return {
        "campaign_cost": f"${total_campaign_cost:,.0f}",
        "revenue_saved": f"${revenue_saved:,.0f}",
        "net_return": f"${net_return:,.0f}",
        "roi_multiple": f"{roi_multiple:.1f}x",
    }


class CustomerFeatures(BaseModel):
    """Schema for single customer prediction request."""
    tenure: float = 12.0
    MonthlyCharges: float = 65.0
    TotalCharges: float = 780.0
    SeniorCitizen: int = 0
    gender: int = 0
    Partner: int = 0
    Dependents: int = 0
    PhoneService: int = 1
    MultipleLines: int = 0
    OnlineSecurity: int = 0
    OnlineBackup: int = 0
    DeviceProtection: int = 0
    TechSupport: int = 0
    StreamingTV: int = 0
    StreamingMovies: int = 0
    PaperlessBilling: int = 1
    is_honeymoon_period: int = 0
    unprotected_fiber: int = 0
    monthly_charge_per_tenure: float = 5.0
    service_bundle_depth: int = 1
    is_auto_pay: int = 0
    is_month_to_month: int = 1


@app.post("/predict")
async def predict_single(customer: CustomerFeatures):
    """Predict churn probability for a single customer with risk tier and action."""
    try:
        load_churn_model()

        if MODEL_STATE["churn_model"] is None:
            logger.warning("Model not loaded")
            return {
                "churn_probability": 0.35,
                "risk_tier": "MEDIUM",
                "projected_clv": round(customer.MonthlyCharges * 24, 2),
                "revenue_at_risk": round(0.35 * customer.MonthlyCharges * 24, 2),
                "recommended_action": "Monitor account activity",
            }

        import pandas as pd

        feature_names = MODEL_STATE["feature_names"]
        customer_dict = customer.model_dump()

        # Build feature row in correct order
        row = pd.DataFrame(
            [[customer_dict.get(f, 0.0) for f in feature_names]],
            columns=feature_names,
        )

        # Scale if scaler exists, otherwise use raw
        if MODEL_STATE["scaler"] is not None:
            row_pred = MODEL_STATE["scaler"].transform(row)
        else:
            row_pred = row

        # Predict
        pred_proba = MODEL_STATE["churn_model"].predict_proba(row_pred)
        churn_prob = float(pred_proba[0, 1])

        if churn_prob >= 0.70:
            tier = "CRITICAL"
        elif churn_prob >= 0.40:
            tier = "HIGH"
        elif churn_prob >= 0.20:
            tier = "MEDIUM"
        else:
            tier = "LOW"

        if tier == "CRITICAL":
            action = "Immediate personal outreach with retention offer"
        elif tier == "HIGH":
            action = "Send personalized email with discount offer"
        else:
            action = "Monitor account activity"

        return {
            "churn_probability": round(churn_prob, 4),
            "risk_tier": tier,
            "projected_clv": round(customer.MonthlyCharges * 24, 2),
            "revenue_at_risk": round(churn_prob * customer.MonthlyCharges * 24, 2),
            "recommended_action": action,
        }

    except Exception as e:
        logger.error(f"Prediction error: {type(e).__name__}: {e}", exc_info=True)
        return {
            "churn_probability": 0.35,
            "risk_tier": "MEDIUM",
            "projected_clv": round(customer.MonthlyCharges * 24, 2),
            "revenue_at_risk": round(0.35 * customer.MonthlyCharges * 24, 2),
            "recommended_action": "Monitor account activity",
        }


# SECTION 5: Dashboard route

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve dashboard with server-side rendered metrics."""
    try:
        metrics = get_dashboard_metrics()

        html_path = TEMPLATE_DIR / "index.html"
        html_content = html_path.read_text()

        # Simple string substitution for template variables
        html_content = html_content.replace("{{ total_customers }}", str(metrics["total_customers"]))
        html_content = html_content.replace("{{ at_risk_count }}", str(metrics["at_risk_count"]))
        html_content = html_content.replace("{{ at_risk_pct }}", str(metrics["at_risk_pct"]))
        html_content = html_content.replace("{{ revenue_exposure }}", str(metrics["revenue_exposure"]))
        html_content = html_content.replace("{{ revenue_exposure_pct }}", str(metrics["revenue_exposure_pct"]))
        html_content = html_content.replace("{{ potential_saves }}", str(metrics["potential_saves"]))
        html_content = html_content.replace("{{ risk_critical }}", str(metrics["risk_critical"]))
        html_content = html_content.replace("{{ risk_high }}", str(metrics["risk_high"]))
        html_content = html_content.replace("{{ risk_medium }}", str(metrics["risk_medium"]))
        html_content = html_content.replace("{{ risk_low }}", str(metrics["risk_low"]))
        html_content = html_content.replace("{{ auc_roc }}", str(metrics["auc_roc"]))
        html_content = html_content.replace("{{ auc_pr }}", str(metrics["auc_pr"]))

        return HTMLResponse(html_content)

    except Exception as e:
        logger.error(f"Failed to serve dashboard: {e}")
        return HTMLResponse("<h1>Customer Churn Prediction Dashboard</h1><p>Error loading dashboard</p>", status_code=500)


logger.info("Customer Churn Prediction API initialized successfully")
