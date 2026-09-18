import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---- Page config (must be the first Streamlit command) ----
st.set_page_config(
    page_title="BhuNirvighna AI",
    page_icon="🏗️",
    layout="wide"
)

# ---- Load saved model, explainer, and feature list ----
model = joblib.load("models/risk_model.pkl")
explainer = joblib.load("models/shap_explainer.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ---- Load your cleaned dataset ----
df = pd.read_csv("data/cleaned/cleaned_data.csv")

# ---- Basic title ----
st.title("BhuNirvighna AI")
st.caption("Predictive Analytics for Early Detection of Land Acquisition Delays")

st.write("Data loaded successfully:", df.shape)

# data sets loading and rating risks


# ---- Prepare features exactly like we did in the notebook ----
features = df.drop(columns=['Delay_Label', 'Risk_Score_0_100', 'Project_ID', 'Data_Basis'], errors='ignore')

date_cols = ['Sanction_Date','3A_Notification_Date','3D_Notification_Date',
             'Award_Date','Payment_Date','Possession_Date']
for c in date_cols:
    features[c] = pd.to_datetime(features[c])

features['Days_Sanction_to_3A'] = (features['3A_Notification_Date'] - features['Sanction_Date']).dt.days
features['Days_3A_to_3D'] = (features['3D_Notification_Date'] - features['3A_Notification_Date']).dt.days
features['Days_3D_to_Award'] = (features['Award_Date'] - features['3D_Notification_Date']).dt.days
features['Days_Award_to_Payment'] = (features['Payment_Date'] - features['Award_Date']).dt.days
features['Days_Payment_to_Possession'] = (features['Possession_Date'] - features['Payment_Date']).dt.days

features = features.drop(columns=date_cols + ['date_issue'], errors='ignore')

# ---- One-hot encode, then align to the exact training columns ----
features_encoded = pd.get_dummies(features, columns=[
    'State', 'District', 'Project_Type', 'Compensation_Status', 'Possession_Status'
], drop_first=True)

# Reindex to match training columns exactly — fills any missing dummy columns with 0
features_encoded = features_encoded.reindex(columns=feature_columns, fill_value=0)

# ---- Predict risk probability for every project ----
df['Predicted_Risk_Probability'] = model.predict_proba(features_encoded)[:, 1]
df['Predicted_Risk_Percent'] = (df['Predicted_Risk_Probability'] * 100).round(1)

st.write("Risk score distribution check:", df['Predicted_Risk_Percent'].describe())

def risk_category(p):
    if p >= 70:
        return "🔴 High"
    elif p >= 40:
        return "🟡 Medium"
    else:
        return "🟢 Low"

df['Risk_Category'] = df['Predicted_Risk_Percent'].apply(risk_category)

# ---- Display the risk-ranked table ----
st.subheader("Project Risk Rankings")

display_cols = ['Project_ID', 'State', 'District', 'Project_Type',
                 'Predicted_Risk_Percent', 'Risk_Category', 'Compensation_Status', 'Possession_Status']

sorted_df = df[display_cols].sort_values('Predicted_Risk_Percent', ascending=False)

st.dataframe(sorted_df, use_container_width=True, height=400)


# chats

# ---- Summary metrics row ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Projects", len(df))
col2.metric("High Risk", (df['Risk_Category'] == "🔴 High").sum())
col3.metric("Medium Risk", (df['Risk_Category'] == "🟡 Medium").sum())
col4.metric("Low Risk", (df['Risk_Category'] == "🟢 Low").sum())

st.divider()

# ---- Charts row ----
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Delay Rate by Compensation Status")
    comp_chart = df.groupby('Compensation_Status')['Delay_Label'].mean().reset_index()
    comp_chart['Delay_Label'] = comp_chart['Delay_Label'] * 100
    fig1 = px.bar(comp_chart, x='Compensation_Status', y='Delay_Label',
                  labels={'Delay_Label': 'Delay Rate (%)'}, color='Delay_Label',
                  color_continuous_scale='Reds')
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("Risk Distribution Across Projects")
    fig2 = px.histogram(df, x='Predicted_Risk_Percent', nbins=30,
                         labels={'Predicted_Risk_Percent': 'Predicted Risk (%)'})
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---- Statewise average risk (sets up for GIS map next) ----
st.subheader("Average Risk by State")
state_risk = df.groupby('State')['Predicted_Risk_Percent'].mean().reset_index().sort_values('Predicted_Risk_Percent', ascending=False)
fig3 = px.bar(state_risk, x='State', y='Predicted_Risk_Percent',
              labels={'Predicted_Risk_Percent': 'Avg Predicted Risk (%)'},
              color='Predicted_Risk_Percent', color_continuous_scale='OrRd')
st.plotly_chart(fig3, use_container_width=True)



## importing map

import folium
from streamlit_folium import st_folium

# ---- Approximate state-level centroid coordinates (India) ----
state_coords = {
    'Rajasthan': (27.0238, 74.2179), 'Madhya Pradesh': (23.4733, 77.9479),
    'Odisha': (20.9517, 85.0985), 'West Bengal': (22.9868, 87.8550),
    'Bihar': (25.0961, 85.3131), 'Gujarat': (22.2587, 71.1924),
    'Tamil Nadu': (11.1271, 78.6569), 'Uttar Pradesh': (26.8467, 80.9462),
    'Maharashtra': (19.7515, 75.7139), 'Karnataka': (15.3173, 75.7139),
    'Punjab': (31.1471, 75.3412), 'Haryana': (29.0588, 76.0856),
    'Kerala': (10.8505, 76.2711), 'Andhra Pradesh': (15.9129, 79.7400),
    'Telangana': (18.1124, 79.0193), 'Assam': (26.2006, 92.9376),
}

st.subheader("GIS Risk Heatmap — State-wise")

state_summary = df.groupby('State').agg(
    Avg_Risk=('Predicted_Risk_Percent', 'mean'),
    Project_Count=('Project_ID', 'count')
).reset_index()

m = folium.Map(location=[22.5, 80], zoom_start=5, tiles='CartoDB positron')

for _, row in state_summary.iterrows():
    coords = state_coords.get(row['State'])
    if coords:
        risk = row['Avg_Risk']
        color = '#a43b3b' if risk >= 50 else ('#b5871a' if risk >= 25 else '#4a7a1e')
        folium.CircleMarker(
            location=coords,
            radius=8 + (risk / 5),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>{row['State']}</b><br>Avg Risk: {risk:.1f}%<br>Projects: {row['Project_Count']}"
        ).add_to(m)

st_folium(m, use_container_width=True, height=500)


## Semi-Circular Risk Gauge

import plotly.graph_objects as go

st.subheader("Overall Portfolio Risk Gauge")

overall_risk = df['Predicted_Risk_Percent'].mean()

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=overall_risk,
    number={'suffix': "%", 'font': {'size': 40}},
    title={'text': "Average Delay Risk Across All Projects", 'font': {'size': 18}},
    gauge={
        'axis': {'range': [0, 100], 'tickwidth': 1},
        'bar': {'color': "#2b2d76", 'thickness': 0.3},
        'steps': [
            {'range': [0, 40], 'color': "#d9ecd0"},   # low risk - light green
            {'range': [40, 70], 'color': "#fdf1d6"},  # medium risk - light gold
            {'range': [70, 100], 'color': "#f6dede"}, # high risk - light red
        ],
        'threshold': {
            'line': {'color': "#a43b3b", 'width': 4},
            'thickness': 0.85,
            'value': overall_risk
        }
    }
))

fig_gauge.update_layout(height=350, margin=dict(t=50, b=10, l=30, r=30))
st.plotly_chart(fig_gauge, use_container_width=True)


## Input Form + Live Prediction

def show_risk_gauge(risk_value, label="Predicted Risk Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value,
        number={'suffix': "%", 'font': {'size': 36}},
        title={'text': label, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2b2d76", 'thickness': 0.3},
            'steps': [
                {'range': [0, 40], 'color': "#d9ecd0"},
                {'range': [40, 70], 'color': "#fdf1d6"},
                {'range': [70, 100], 'color': "#f6dede"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
    return fig


st.divider()
st.header("🔍 Predict Risk for a New Project")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        input_state = st.selectbox("State", sorted(df['State'].unique()))
        input_district = st.selectbox("District", sorted(df['District'].unique()))
        input_project_type = st.selectbox("Project Type", sorted(df['Project_Type'].unique()))
        input_land_required = st.number_input("Land Required (Hectares)", min_value=1.0, value=100.0)
        input_land_acquired = st.number_input("Land Acquired So Far (Hectares)", min_value=0.0, value=50.0)

    with col2:
        input_affected_families = st.number_input("Affected Families", min_value=0, value=200)
        input_displaced_families = st.number_input("Displaced Families", min_value=0, value=50)
        input_compensation_amount = st.number_input("Compensation Amount (Crore)", min_value=0.0, value=10.0)
        input_rr_amount = st.number_input("R&R Amount (Crore)", min_value=0.0, value=2.0)
        input_rr_progress = st.slider("R&R Progress (%)", 0, 100, 50)

    with col3:
        input_compensation_status = st.selectbox("Compensation Status", sorted(df['Compensation_Status'].unique()))
        input_possession_status = st.selectbox("Possession Status", sorted(df['Possession_Status'].unique()))
        input_legal_disputes = st.number_input("Legal Disputes Count", min_value=0, value=1)
        input_days_3a_to_3d = st.number_input("Days: 3A to 3D Notification", min_value=0, value=90)
        input_days_3d_to_award = st.number_input("Days: 3D Notification to Award", min_value=0, value=90)

    submitted = st.form_submit_button("Predict Risk", use_container_width=True)

if submitted:
    # Build a single-row dataframe matching your training feature structure
    new_project = pd.DataFrame([{
        'State': input_state,
        'District': input_district,
        'Project_Type': input_project_type,
        'Land_Required_Hectare': input_land_required,
        'Land_Acquired_Hectare': input_land_acquired,
        'Affected_Families': input_affected_families,
        'Displaced_Families': input_displaced_families,
        'Compensation_Amount_Crore': input_compensation_amount,
        'RR_Amount_Crore': input_rr_amount,
        'Compensation_Status': input_compensation_status,
        'R&R_Progress_Percent': input_rr_progress,
        'Legal_Disputes_Count': input_legal_disputes,
        'Possession_Status': input_possession_status,
        'Days_Sanction_to_3A': 60,  # reasonable default — not collected in form
        'Days_3A_to_3D': input_days_3a_to_3d,
        'Days_3D_to_Award': input_days_3d_to_award,
        'Days_Award_to_Payment': 60,  # reasonable default
        'Days_Payment_to_Possession': 60,  # reasonable default
    }])

    # One-hot encode exactly like training, then align to the same columns
    new_encoded = pd.get_dummies(new_project, columns=[
        'State', 'District', 'Project_Type', 'Compensation_Status', 'Possession_Status'
    ])
    new_encoded = new_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict
    risk_prob = model.predict_proba(new_encoded)[:, 1][0]
    risk_percent = round(risk_prob * 100, 1)

    st.success("Prediction complete")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.plotly_chart(show_risk_gauge(risk_percent, "Predicted Delay Risk"), use_container_width=True)

    with col_b:
        st.metric("Predicted Risk Category",
                   "🔴 High" if risk_percent >= 70 else ("🟡 Medium" if risk_percent >= 40 else "🟢 Low"))
        st.metric("Predicted Risk Score", f"{risk_percent}%")

        # Recommendation enginee

            # ---- Recommendation Engine (Module C) ----
    st.divider()
    st.subheader("💡 Recommended Actions")

    # Get SHAP values for this single new project
    new_shap_values = explainer.shap_values(new_encoded)

    # Pair each feature with its SHAP impact, sort by importance
    shap_impact = pd.DataFrame({
        'Feature': new_encoded.columns,
        'Impact': new_shap_values[0]
    }).sort_values('Impact', ascending=False)

    # Only features pushing risk UP matter for recommendations
    top_risk_drivers = shap_impact[shap_impact['Impact'] > 0].head(3)

    # Map feature names to plain-language recommendations
    recommendation_map = {
        'Compensation_Status_Pending': "🔴 Expedite compensation disbursement approval — pending payments are a major delay driver.",
        'Compensation_Status_Partially paid': "🟡 Accelerate remaining compensation disbursement to reduce risk further.",
        'Possession_Status_Pending': "🔴 Prioritize possession handover — escalate coordination with district administration.",
        'Possession_Status_Partial possession': "🟡 Follow up on pending possession handovers to close the gap.",
        'Legal_Disputes_Count': "⚖️ Fast-track pending legal dispute resolution through designated fast-track courts.",
        'R&R_Progress_Percent': "🏘️ Increase rehabilitation & resettlement progress — engage affected families proactively.",
        'Days_3A_to_3D': "📄 Reduce administrative delay between 3A and 3D notification stages.",
        'Days_3D_to_Award': "📋 Expedite award declaration process to avoid stage-wise bottleneck.",
    }

    if len(top_risk_drivers) == 0:
        st.success("✅ No significant risk drivers detected — this project shows a healthy risk profile.")
    else:
        for _, row in top_risk_drivers.iterrows():
            feature_name = row['Feature']
            # Try exact match first, then partial match for one-hot encoded columns
            rec = recommendation_map.get(feature_name)
            if rec is None:
                for key in recommendation_map:
                    if key.split('_')[0] in feature_name:
                        rec = recommendation_map[key]
                        break
            if rec:
                st.warning(rec)
            else:
                st.info(f"⚠️ Monitor **{feature_name}** — contributing to elevated risk.")
