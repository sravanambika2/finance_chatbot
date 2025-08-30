import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import requests
import os

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "ibm-granite/granite 3.3-8b-instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

EXCHANGE_RATE = 83.5  # USD -> INR

def convert_currency(amount: float, to_currency: str) -> float:
    return amount * EXCHANGE_RATE if "INR" in to_currency else amount

def query_granite(prompt: str) -> str:
    try:
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 250}}
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
        if response.status_code == 401:
            return "(Unauthorized: Check your Hugging Face API key and model access.)"
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            resp_text = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            resp_text = data["generated_text"]
        else:
            resp_text = str(data)
        return resp_text
    except Exception as e:
        return f"(Error calling Hugging Face: {e})"

SAMPLE_CSV = """date,description,amount,category
2025-06-01,Monthly Subscription - Streaming,-12.99,Subscriptions
2025-06-02,Grocery Store - Supermart,-76.45,Groceries
2025-06-05,Salary Payment,2000,Income
2025-06-07,Electricity Bill,-54.30,Utilities
2025-06-10,Coffee Shop - Morning Cup,-8.50,Food
2025-06-12,Taxi - Ride,-15.20,Transport
2025-06-15,Online Course - Student Discount,-29.00,Education
"""

def infer_category(desc: str) -> str:
    s = (desc or "").lower()
    if any(k in s for k in ["grocery","supermarket","grocer"]): return "Groceries"
    if any(k in s for k in ["uber","lyft","taxi","bus","train","transport"]): return "Transport"
    if any(k in s for k in ["coffee","cafe","starbucks"]): return "Food"
    if any(k in s for k in ["rent","mortgage"]): return "Housing"
    if any(k in s for k in ["netflix","spotify","subscription","prime","hulu"]): return "Subscriptions"
    if any(k in s for k in ["salary","payroll","paycheck"]): return "Income"
    if any(k in s for k in ["electric","water","gas","bill"]): return "Utilities"
    return "Other"

def load_transactions(file_or_buffer):
    df = pd.read_csv(file_or_buffer)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    if "date" in df.columns:
        try: df["date"] = pd.to_datetime(df["date"])
        except: pass
    if "description" not in df.columns: df["description"] = ""
    if "category" not in df.columns:
        df["category"] = df["description"].fillna("").apply(infer_category)
    else:
        df["category"] = df["category"].fillna("").replace("", np.nan)
        missing = df["category"].isna()
        if missing.any():
            df.loc[missing, "category"] = df.loc[missing, "description"].apply(infer_category)
    return df

def budget_summary(df: pd.DataFrame, currency="USD ($)"):
    df2 = df.copy()
    df2["spend"] = df2["amount"].apply(lambda x: -x if x < 0 else 0)
    total_spend = df2["spend"].sum()
    total_income = df2[df2["amount"] > 0]["amount"].sum()
    by_cat = df2.groupby("category")["spend"].sum().sort_values(ascending=False)
    summary = {
        "Total Income": convert_currency(total_income, currency),
        "Total Spending": convert_currency(total_spend, currency),
        "Top Categories": by_cat
    }
    return summary

def spending_insights(df: pd.DataFrame, currency="USD ($)"):
    df2 = df.copy()
    df2["spend"] = df2["amount"].apply(lambda x: -x if x < 0 else 0)
    total_spend = df2["spend"].sum()
    insights = []
    if total_spend > 0:
        by_cat = df2.groupby("category")["spend"].sum().sort_values(ascending=False)
        top_cat = by_cat.index[0]
        top_amt = by_cat.iloc[0]
        top_pct = top_amt / total_spend * 100
        insights.append(f"Largest spending category: {top_cat} — {currency} {convert_currency(top_amt, currency):,.2f} ({top_pct:.1f}%)")
    return insights

st.set_page_config(page_title="Personal Finance Dashboard", layout="wide")
if "profile_submitted" not in st.session_state:
    st.session_state.profile_submitted = False

if not st.session_state.profile_submitted:
    st.title("👋 Welcome to Personal Finance Dashboard")
    st.markdown("Please enter your details to continue.")
    with st.form("profile_form"):
        name = st.text_input("Name", "")
        age = st.number_input('Age', 1, 120, 25)
        email = st.text_input("Email", "")
        country = st.selectbox("Country", ["United States", "India", "United Kingdom", "Germany", "Japan", "Canada"])
        currency = st.selectbox("Currency", ["USD ($)", "INR (₹)", "EUR (€)", "GBP (£)", "JPY (¥)"])
        language = st.selectbox("Language", ["English", "Hindi", "Spanish", "French", "German", "Chinese"])
        submitted = st.form_submit_button("Continue to Dashboard")
    if submitted:
        st.session_state.profile_submitted = True
        st.session_state.profile = {
            "name": name,
            "age": age,
            "email": email,
            "country": country,
            "currency": currency,
            "language": language
        }
        st.rerun()
else:
    profile = st.session_state.profile
    st.sidebar.header("⚙️ Settings")

    # --- File uploader (unique key) ---
    uploaded = st.sidebar.file_uploader("transactions.csv", type=["csv"], key="file_upload")
    if st.sidebar.button("Load Sample Data", key="sample_btn"):
        uploaded = StringIO(SAMPLE_CSV)

    # --- Load data ---
    if uploaded:
        try:
            df = load_transactions(uploaded)
        except:
            st.sidebar.error("Failed to load CSV.")
            df = load_transactions(StringIO(SAMPLE_CSV))
    else:
        df = load_transactions(StringIO(SAMPLE_CSV))

    summary = budget_summary(df, currency=profile["currency"])
    insights = spending_insights(df, currency=profile["currency"])

    # --- Settings Form ---
    with st.sidebar.form("settings_form"):
        monthly_income = st.number_input("Monthly Income", min_value=0.0, value=3000.0, format="%.2f", key="monthly_income")
        savings_goal = st.number_input("Monthly Savings Goal", min_value=0.0, value=500.0, format="%.2f", key="savings_goal")
        user_type = st.selectbox("User Type", ["student", "professional"], key="user_type")
        goal = st.selectbox("Main Goal", ["Save money", "Understand taxes", "Start investing"], key="goal")
        risk = st.selectbox("Risk Tolerance", ["low", "medium", "high"], key="risk")
        knowledge = st.selectbox("Financial Knowledge", ["beginner", "intermediate", "advanced"], key="knowledge")
        country = st.selectbox("Country", ["United States", "India", "United Kingdom", "Germany", "Japan", "Canada"], key="country")
        currency = st.selectbox("Currency", ["USD ($)", "INR (₹)", "EUR (€)", "GBP (£)", "JPY (¥)"], key="currency")
        language = st.selectbox("Language", ["English", "Hindi", "Spanish", "French", "German", "Chinese"], key="language")
        save_settings = st.form_submit_button("Save Settings")
        if save_settings:
            st.session_state.profile.update({
                "monthly_income": monthly_income,
                "savings_goal": savings_goal,
                "user_type": user_type,
                "goal": goal,
                "risk": risk,
                "knowledge": knowledge,
                "country": country,
                "currency": currency,
                "language": language
            })
            st.success("Settings saved!")

    st.sidebar.markdown("### Add a Transaction")
    with st.sidebar.form("add_transaction"):
        t_date = st.date_input("Date", key="txn_date")
        t_desc = st.text_input("Description", key="txn_desc")
        t_amount = st.number_input("Amount", value=0.0, format="%.2f", key="txn_amount")
        t_cat = st.selectbox("Category", ["Groceries", "Transport", "Food", "Housing", "Subscriptions", "Income", "Utilities", "Education", "Other"], key="txn_cat")
        add_txn = st.form_submit_button("Add")
        if add_txn:
            new_row = {
                "date": pd.to_datetime(t_date),
                "description": t_desc,
                "amount": t_amount,
                "category": t_cat
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Transaction added!")

    # --- Dashboard ---
    st.title(f"💼 Personal Finance Dashboard - {profile['name']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Income", f"{currency} {monthly_income:,.2f}")
    col2.metric("Total Spending", f"{currency} {summary['Total Spending']:,.2f}")
    col3.metric("Balance", f"{currency} {monthly_income - summary['Total Spending']:,.2f}")

    st.subheader("Spending by Category")
    cat_chart = px.pie(names=summary["Top Categories"].index,
                       values=summary["Top Categories"].values,
                       color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(cat_chart, use_container_width=True)

    st.subheader("Recent Transactions")
    st.dataframe(df.sort_values("date", ascending=False))

    st.subheader("Insights")
    for i in insights:
        st.info(i)

    # --- Chatbot ---
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown("---")
    st.header("💬 Finance Chatbot")
    query = st.text_input("Ask the Chatbot a finance question:", key="query_input")
    col1, col2 = st.columns([4,1])
    with col2:
        if st.button("Ask"):
            if query.strip():
                prompt = (
                    f"User profile: {profile}\n"
                    f"Country: {country}\n"
                    f"Currency: {currency}\n"
                    f"Language: {language}\n"
                    f"User Type: {user_type}\n"
                    f"Goal: {goal}\n"
                    f"Risk Tolerance: {risk}\n"
                    f"Financial Knowledge: {knowledge}\n"
                    f"Question: {query}\n"
                    "Give practical finance tips. Keep it short. Not financial advice."
                )
                resp = query_granite(prompt)
                st.session_state.history.append({"user": query, "bot": resp})
            else:
                st.warning("Please type a question first.")

    st.subheader("Chat History")
    for msg in reversed(st.session_state.history):
        st.markdown(f"**You:** {msg['user']}")
        st.markdown(f"**Assistant:** {msg['bot']}")
        st.markdown("---")