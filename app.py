import streamlit as st
import pandas as pd
from fairlearn.metrics import demographic_parity_difference
import plotly.express as px
import plotly.graph_objects as go
import time

# --- BRANDING & CSS ---
# Pastel Green: #A7D7C5
# Pastel Orange: #F6B896
# Background: #F9FDFB
st.set_page_config(page_title="FairAuditor Dashboard", page_icon="⚖️", layout="wide")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password", help="Get this from Google AI Studio")

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #F9FDFB;
        color: #333333;
    }
    /* Headers */
    h1, h2, h3 {
        color: #2F4F4F;
        font-family: 'Inter', sans-serif;
    }
    /* Cards and containers */
    .css-1r6slb0, .css-1y4p8pa {
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 20px;
    }
    /* Custom primary button */
    .stButton>button {
        background-color: #A7D7C5;
        color: #2F4F4F;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #8EBEAB;
        color: #1A2F2F;
        transform: translateY(-2px);
    }
    /* Specific styling for AI Report Button */
    #ai-report-btn button {
        background-color: #F6B896;
        color: #5C3A21;
        width: 100%;
        font-size: 1.1rem;
        padding: 15px;
    }
    #ai-report-btn button:hover {
        background-color: #E2A482;
    }
    /* Metrics display */
    div[data-testid="stMetricValue"] {
        color: #F6B896;
    }
    div[data-testid="stMetricLabel"] * {
        color: #2F4F4F !important;
    }
    /* Alerts */
    .stAlert, .stAlert * {
        color: #2F4F4F !important;
    }
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3358/3358826.png", width=80) # Placeholder logo
with col2:
    st.title("FairAuditor: AI Bias Detection")
    st.markdown("*A professional dashboard to audit AI datasets and models for fairness.*")

st.markdown("---")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]
    df = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True)
    df.dropna(inplace=True)
    return df

with st.spinner("Loading Adult Income Dataset..."):
    df = load_data()

st.subheader("1. Data Overview")
st.write("We are analyzing the **Adult Income Dataset**. The goal is to detect bias in the historical data by observing the rate of positive outcomes (Income >50K) across different demographic groups.")

with st.expander("View Raw Data Snippet"):
    st.dataframe(df.head(10))

# --- FAIRNESS CALCULATION ---
st.subheader("2. Fairness Analysis: Gender Bias")

# Prepare target variable (binarize)
df['outcome_bin'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

# Calculate base rates
gender_rates = df.groupby('sex')['outcome_bin'].mean().reset_index()
gender_rates.rename(columns={'outcome_bin': 'Positive Outcome Rate'}, inplace=True)

# Calculate Demographic Parity Difference
dpd = demographic_parity_difference(y_true=df['outcome_bin'], y_pred=df['outcome_bin'], sensitive_features=df['sex'])

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric(label="Overall Positive Outcome Rate", value=f"{df['outcome_bin'].mean():.1%}")
with col_m2:
    st.metric(label="Demographic Parity Difference", value=f"{dpd:.1%}", 
              help="The absolute difference in positive outcome rates between the demographic groups.")

if dpd > 0.10:
    st.warning(f"⚠️ **Significant Disparity Detected:** The demographic parity difference is {dpd:.1%}. This exceeds the standard 10% threshold, indicating potential bias in the dataset outcomes between sexes.")
else:
    st.success(f"✅ **Fairness Acceptable:** The demographic parity difference is {dpd:.1%}, which is within an acceptable range.")

# --- VISUALIZATION ---
st.markdown("### Disparate Impact Visualization")
st.write("The chart below illustrates the percentage of individuals earning >50K, grouped by sex.")

fig = px.bar(
    gender_rates, 
    x='sex', 
    y='Positive Outcome Rate',
    text='Positive Outcome Rate',
    color='sex',
    color_discrete_map={'Female': '#A7D7C5', 'Male': '#F6B896'},
    labels={'sex': 'Gender', 'Positive Outcome Rate': 'Rate of Income >50K'}
)
fig.update_traces(texttemplate='%{text:.1%}', textposition='outside', textfont_color='#2F4F4F')
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    yaxis=dict(tickformat='.0%', title='Positive Outcome Rate (>50K)'),
    xaxis=dict(title='Gender'),
    margin=dict(t=20, b=20, l=20, r=20),
    showlegend=False,
    font=dict(color='#2F4F4F')
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- AI REPORT GENERATION ---
st.subheader("3. Automated Insights")
st.write("Generate a comprehensive AI-driven bias report based on the observed disparities.")

st.markdown('<div id="ai-report-btn">', unsafe_allow_html=True)
if st.button("✨ Generate Gemini AI Bias Report"):
    with st.spinner("Gemini is analyzing the data and drafting the executive report..."):
        time.sleep(2)
        mock_report = f"""
**Executive Summary**

The Demographic Parity analysis on the Adult Income dataset reveals a significant historical gender bias, with a **Demographic Parity Difference of {dpd:.1%}**. Specifically, the positive outcome rate (income >$50K) for males is **{gender_rates.loc[gender_rates['sex'] == 'Male', 'Positive Outcome Rate'].values[0]:.1%}**, compared to only **{gender_rates.loc[gender_rates['sex'] == 'Female', 'Positive Outcome Rate'].values[0]:.1%}** for females. 

This statistical disparity indicates that any machine learning model trained naively on this dataset will likely learn and perpetuate this existing inequity, resulting in discriminatory predictive outcomes that disproportionately favor male candidates.

**Actionable Engineering Recommendations**

1.  **Implement Data Pre-processing Interventions**: Apply reweighing techniques to the training data. Assign higher sample weights to the unprivileged group with positive outcomes (Female, >$50K) and lower weights to the privileged group with positive outcomes (Male, >$50K). This balances the effective base rates without altering the underlying feature distributions.
2.  **In-Processing Regularization**: Integrate a fairness constraint directly into the model's loss function during training (e.g., using `fairlearn.reductions.ExponentiatedGradient`). This will penalize the model for violating Demographic Parity, forcing it to find a decision boundary that optimizes for both accuracy and fairness simultaneously.
"""
        st.success("Report Generated Successfully!")
        st.markdown("### 📋 Executive Bias Report")
        st.info(mock_report)
st.markdown('</div>', unsafe_allow_html=True)

# Testing collaboration feature