# app/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings

# Page config
st.set_page_config(page_title="EMIPredict AI", page_icon="💳", layout="centered")

# Helpers: find and load models
def find_models_dir(max_up=4):
    p = Path(__file__).resolve()
    for _ in range(max_up + 1):
        candidate = p.parent / "models"
        if candidate.exists():
            return candidate
        p = p.parent
    return None

@st.cache_resource
def load_models():
    models_dir = find_models_dir()
    if models_dir is None:
        raise FileNotFoundError("Could not find a models/ directory (looked up parent folders).")
    clf_path = models_dir / "pipeline_classification_xgb_tuned_fast.joblib"
    reg_path = models_dir / "pipeline_regression_xgb_focused.joblib"
    if not clf_path.exists() or not reg_path.exists():
        raise FileNotFoundError(f"Model files not found.\n - {clf_path}\n - {reg_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = joblib.load(clf_path)
        reg = joblib.load(reg_path)
    return clf, reg

try:
    clf_model, reg_model = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

st.title("💳 EMIPredict AI — Eligibility & EMI Predictor")
st.write("Enter customer details and get classification + predicted maximum EMI.")

def fmt_currency(x):
    try:
        return f"₹{int(round(x)):,}"
    except Exception:
        return str(x)

# Preset data
def sample_1():
    return {
        "age": 30, "gender":"Male", "marital_status":"Married", "education":"Graduate",
        "employment_type":"Private", "company_type":"Private", "years_of_employment":5, "monthly_salary":60000,
        "family_size":4, "dependents":2, "house_type":"Rented", "monthly_rent":12000,
        "school_fees":4000, "college_fees":0, "travel_expenses":3000, "groceries_utilities":8000,
        "other_monthly_expenses":4000, "existing_loans":1, "current_emi_amount":5000,
        "credit_score":720, "bank_balance":100000, "emergency_fund":50000,
        "emi_scenario":"Standard", "requested_amount":300000, "requested_tenure":36,
        "emi_eligibility":"Eligible"
    }

def sample_2():
    return {
        "age": 45, "gender":"Female", "marital_status":"Married", "education":"Postgraduate",
        "employment_type":"Government", "company_type":"Government", "years_of_employment":15, "monthly_salary":90000,
        "family_size":5, "dependents":3, "house_type":"Own", "monthly_rent":0,
        "school_fees":10000, "college_fees":5000, "travel_expenses":4000, "groceries_utilities":12000,
        "other_monthly_expenses":5000, "existing_loans":2, "current_emi_amount":15000,
        "credit_score":780, "bank_balance":300000, "emergency_fund":150000,
        "emi_scenario":"Standard", "requested_amount":800000, "requested_tenure":60,
        "emi_eligibility":"Eligible"
    }

def sample_3():
    return {
        "age": 28, "gender":"Male", "marital_status":"Single", "education":"Graduate",
        "employment_type":"Self-employed", "company_type":"Private", "years_of_employment":3, "monthly_salary":25000,
        "family_size":2, "dependents":0, "house_type":"Rented", "monthly_rent":7000,
        "school_fees":0, "college_fees":0, "travel_expenses":3000, "groceries_utilities":5000,
        "other_monthly_expenses":3000, "existing_loans":1, "current_emi_amount":4000,
        "credit_score":650, "bank_balance":25000, "emergency_fund":20000,
        "emi_scenario":"Standard", "requested_amount":150000, "requested_tenure":24,
        "emi_eligibility":"Not_Eligible"
    }

# Input form
with st.form(key="customer_form"):
    cols = st.columns(2)
    left = cols[0]; right = cols[1]

    with left:
        age = left.number_input("Age", 18, 80, 30)
        gender = left.selectbox("Gender", ["Male", "Female"])
        marital_status = left.selectbox("Marital Status", ["Single", "Married"])
        education = left.selectbox("Education", ["High School", "Graduate", "Postgraduate", "Doctorate"])
        employment_type = left.selectbox("Employment Type", ["Private", "Self-employed", "Government"])
        company_type = left.selectbox("Company Type", ["Private", "Public", "Government", "Startup", "Other"])
        years_of_employment = left.number_input("Years of Employment", 0, 50, 5)
        monthly_salary = left.number_input("Monthly Salary (₹)", 0, 5_000_000, 50000, step=1000)
        family_size = left.number_input("Family Size", 1, 20, 4)
        dependents = left.number_input("Dependents", 0, 10, 1)

    with right:
        house_type = right.selectbox("House Type", ["Rented", "Family", "Own"])
        monthly_rent = right.number_input("Monthly Rent (₹)", 0, 200_000, 10000, step=500)
        school_fees = right.number_input("School Fees (₹)", 0, 200_000, 5000, step=100)
        college_fees = right.number_input("College Fees (₹)", 0, 500_000, 10000, step=100)
        travel_expenses = right.number_input("Travel Expenses (₹)", 0, 100_000, 3000)
        groceries_utilities = right.number_input("Groceries + Utilities (₹)", 0, 200_000, 8000)
        other_monthly_expenses = right.number_input("Other Monthly Expenses (₹)", 0, 200_000, 5000)
        existing_loans = right.number_input("Existing Loans (count)", 0, 20, 1)
        current_emi_amount = right.number_input("Current EMI (₹)", 0, 200_000, 0, step=100)
        credit_score = right.number_input("Credit Score", 300, 900, 700)
        bank_balance = right.number_input("Bank Balance (₹)", 0, 50_000_000, 100000)
        emergency_fund = right.number_input("Emergency Fund (₹)", 0, 50_000_000, 50000)
        emi_scenario = right.selectbox("EMI Scenario", ["Standard", "Stress", "Custom"])
        requested_amount = right.number_input("Requested Amount (₹)", 0, 10_000_000, 200000)
        requested_tenure = right.number_input("Requested Tenure (months)", 6, 240, 36)
        emi_eligibility = right.selectbox("EMI Eligibility (pre-classified)", ["Eligible", "Not_Eligible", "High_Risk"])

    submit = st.form_submit_button("Predict")

# Build input DataFrame
def build_input_df():
    dti = (current_emi_amount / monthly_salary) if monthly_salary else 0
    etoi = ((monthly_rent + groceries_utilities + other_monthly_expenses) / monthly_salary) if monthly_salary else 0
    sav = (bank_balance / monthly_salary) if monthly_salary else 0
    row = {
        "age": age, "gender": gender, "marital_status": marital_status, "education": education,
        "monthly_salary": monthly_salary, "employment_type": employment_type,
        "years_of_employment": years_of_employment, "company_type": company_type, "house_type": house_type,
        "monthly_rent": monthly_rent, "family_size": family_size, "dependents": dependents,
        "school_fees": school_fees, "college_fees": college_fees, "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities, "other_monthly_expenses": other_monthly_expenses,
        "existing_loans": existing_loans, "current_emi_amount": current_emi_amount,
        "credit_score": credit_score, "bank_balance": bank_balance, "emergency_fund": emergency_fund,
        "emi_scenario": emi_scenario, "requested_amount": requested_amount, "requested_tenure": requested_tenure,
        "emi_eligibility": emi_eligibility, "debt_to_income_ratio": dti,
        "expense_to_income_ratio": etoi, "savings_ratio": sav
    }
    return pd.DataFrame([row])

# --- Perform prediction and show results ---
if submit:
    input_df = build_input_df()

    # --- Classification ---
    try:
        pred_num = clf_model.predict(input_df)
        pred_class = pred_num[0] if len(pred_num) else None
        class_proba = clf_model.predict_proba(input_df)[0] if hasattr(clf_model, "predict_proba") else None
    except Exception as e:
        pred_class = "Error"
        class_proba = None
        st.error(f"Classification error: {e}")

    # --- Decode numeric class → text label ---
    decoded_labels = None
    try:
        encoder_path = Path(find_models_dir()) / "label_encoder_classes.joblib"
        if encoder_path.exists():
            label_encoder = joblib.load(encoder_path)
            decoded_labels = label_encoder.classes_
        else:
            decoded_labels = ["Not_Eligible", "Moderate_Risk", "Eligible"]
    except Exception:
        decoded_labels = ["Not_Eligible", "Moderate_Risk", "Eligible"]

    if isinstance(pred_class, (int, np.integer)) and decoded_labels is not None:
        if 0 <= pred_class < len(decoded_labels):
            pred_label = decoded_labels[pred_class]
        else:
            pred_label = str(pred_class)
    else:
        pred_label = str(pred_class)

    # --- Regression ---
    try:
        pred_emi = reg_model.predict(input_df)[0]
    except Exception as e:
        pred_emi = None
        st.error(f"Regression error: {e}")

    # --- Display results ---
    st.markdown("### Results")
    if pred_emi is not None:
        st.metric("Predicted EMI (₹)", fmt_currency(pred_emi))
    else:
        st.write("No regression prediction available.")

    st.write(f"**Predicted Eligibility Class:** {pred_label}")

    # --- Probability table (decoded) ---
    if class_proba is not None:
        try:
            classes = list(decoded_labels) if decoded_labels is not None else list(clf_model.classes_)
            proba_df = pd.DataFrame({
                "Class (Label)": classes,
                "Probability": [float(x) for x in class_proba]
            }).sort_values("Probability", ascending=False)
            st.table(proba_df)
        except Exception:
            st.write("Probability table could not be displayed.")

    # --- History ---
    if "pred_history" not in st.session_state:
        st.session_state.pred_history = []
    rec = input_df.iloc[0].to_dict()
    rec.update({
        "predicted_emi": float(pred_emi) if pred_emi is not None else None,
        "predicted_class": str(pred_label),
        "timestamp": datetime.utcnow().isoformat()
    })
    st.session_state.pred_history.insert(0, rec)

# --- Show history table and download option ---
if "pred_history" in st.session_state and len(st.session_state.pred_history) > 0:
    st.subheader("Recent Predictions")
    hist_df = pd.DataFrame(st.session_state.pred_history)
    st.dataframe(hist_df.head(10))
    csv = hist_df.to_csv(index=False)
    st.download_button("Download predictions (CSV)", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.info("Models loaded from the `models/` directory. Make sure the joblib files are present.")
