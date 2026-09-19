import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import base64

# PAGE CONFIG


st.set_page_config(
    page_title="BhuNirvighna AI",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM UI 


st.markdown("""
<style>
/* ---------- Global ---------- */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.main-header {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 850;
    letter-spacing: -1px;
    text-align: center;
    margin: 0;
}

.project-name {
    text-align: center;
    font-size: clamp(2.35rem, 4.8vw, 3.8rem);
    font-weight: 850;
    letter-spacing: -1.5px;
    line-height: 1.05;
    margin: 0;
}
.project-bhu { color: #2e8b57; }
.project-nirvighna { color: #2457c5; }
.project-ai { color: var(--text-color); }

.sub-header {
    text-align: center;
    opacity: 0.70;
    font-size: 1.05rem;
    margin: 0.2rem 0 1.3rem 0;
}

/* ---------- Cards ---------- */
.metric-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 16px;
    padding: 18px 20px;
    min-height: 125px;
    transition: transform .18s ease, border-color .18s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(217,140,43,.65);
}

.metric-label {
    font-size: .88rem;
    opacity: .70;
    font-weight: 650;
}

.metric-value {
    font-size: 2rem;
    font-weight: 850;
    margin-top: 6px;
}

.metric-help {
    font-size: .78rem;
    opacity: .55;
    margin-top: 4px;
}

.info-card, .alert-card, .action-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 12px;
}

.alert-card {
    border-left: 5px solid #c74747;
}

.action-card {
    border-left: 5px solid #d98c2b;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 800;
    margin-top: .4rem;
    margin-bottom: .6rem;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 750;
}

.badge-high { background: rgba(199,71,71,.15); color: #c74747; }
.badge-medium { background: rgba(181,135,26,.15); color: #b5871a; }
.badge-low { background: rgba(74,122,30,.15); color: #4a7a1e; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128,128,128,.22);
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 7px;
    border-bottom: 1px solid rgba(128,128,128,.20);
}

.stTabs [data-baseweb="tab"] {
    font-weight: 700;
    border-radius: 10px 10px 0 0;
    padding: 10px 14px;
}

/* ---------- Dataframe ---------- */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
}

/* ---------- Form ---------- */
div[data-testid="stForm"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 16px;
    padding: 20px;
}

/* ---------- Progress ---------- */
.progress-label {
    font-size: .85rem;
    font-weight: 700;
    margin-bottom: 5px;
}

/* ---------- Mobile ---------- */
@media (max-width: 800px) {
    .block-container {
        padding-left: .8rem;
        padding-right: .8rem;
    }
}
</style>
""", unsafe_allow_html=True)

# LOAD MODELS & DATA


@st.cache_resource
def load_models():
    model = joblib.load("models/risk_model.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    cost_model = joblib.load("models/cost_model.pkl")
    cost_feature_columns = joblib.load("models/cost_feature_columns.pkl")
    return model, explainer, feature_columns, cost_model, cost_feature_columns

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned/cleaned_data.csv")

model, explainer, feature_columns, cost_model, cost_feature_columns = load_models()
df = load_data().copy()

# HEADER


logo_path = "assets/BhNv-logo.png"
try:
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<div style="display:flex;justify-content:center;margin-bottom:6px;">'
        f'<img src="data:image/png;base64,{logo_b64}" width="145"></div>',
        unsafe_allow_html=True
    )
except FileNotFoundError:
    pass

st.markdown(
    '<div class="project-name"><span class="project-bhu">Bhu</span>'
    '<span class="project-nirvighna">Nirvighna</span>'
    '<span class="project-ai">-Ai</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">AI-Powered Early Warning System for Land Acquisition Delays</p>',
    unsafe_allow_html=True
)

# BATCH RISK SCORING - ORIGINAL LOGIC PRESERVED

features = df.drop(
    columns=['Delay_Label', 'Risk_Score_0_100', 'Project_ID', 'Data_Basis'],
    errors='ignore'
)

date_cols = [
    'Sanction_Date', '3A_Notification_Date', '3D_Notification_Date',
    'Award_Date', 'Payment_Date', 'Possession_Date'
]

for c in date_cols:
    features[c] = pd.to_datetime(features[c])

features['Days_Sanction_to_3A'] = (
    features['3A_Notification_Date'] - features['Sanction_Date']
).dt.days
features['Days_3A_to_3D'] = (
    features['3D_Notification_Date'] - features['3A_Notification_Date']
).dt.days
features['Days_3D_to_Award'] = (
    features['Award_Date'] - features['3D_Notification_Date']
).dt.days
features['Days_Award_to_Payment'] = (
    features['Payment_Date'] - features['Award_Date']
).dt.days
features['Days_Payment_to_Possession'] = (
    features['Possession_Date'] - features['Payment_Date']
).dt.days

features = features.drop(columns=date_cols + ['date_issue'], errors='ignore')

features_encoded = pd.get_dummies(
    features,
    columns=[
        'State', 'District', 'Project_Type',
        'Compensation_Status', 'Possession_Status'
    ],
    drop_first=True
)

features_encoded = features_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

df['Predicted_Risk_Probability'] = model.predict_proba(features_encoded)[:, 1]
df['Predicted_Risk_Percent'] = (
    df['Predicted_Risk_Probability'] * 100
).round(1)


def risk_category(p):
    if p >= 70:
        return "🔴 High"
    elif p >= 40:
        return "🟡 Medium"
    else:
        return "🟢 Low"


df['Risk_Category'] = df['Predicted_Risk_Percent'].apply(risk_category)

# ORIGINAL HELPER FUNCTIONS PRESERVED


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
            ]
        }
    ))
    fig.update_layout(
        height=290,
        margin=dict(t=45, b=5, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#555")
    )
    return fig


def calculate_land_feasibility(
    land_type,
    land_required,
    land_acquired_so_far,
    affected_families,
    legal_disputes,
    rr_progress
):
    score = 100
    land_type_penalty = {
        'Forest': 25,
        'Tribal': 20,
        'Government': 5,
        'Agricultural': 10,
        'Private': 15
    }
    score -= land_type_penalty.get(land_type, 10)

    progress_pct = (
        land_acquired_so_far / land_required * 100
        if land_required > 0 else 0
    )

    if progress_pct < 30:
        score -= 15
    elif progress_pct < 60:
        score -= 7

    if affected_families > 500:
        score -= 15
    elif affected_families > 200:
        score -= 8

    score -= min(legal_disputes * 4, 20)
    score += (rr_progress / 100) * 10

    return max(0, min(100, round(score, 1)))


recommendation_map = {
    'Compensation_Status_Pending':
        "🔴 Expedite compensation disbursement approval - pending payments are a major delay driver.",
    'Compensation_Status_Partially paid':
        "🟡 Accelerate remaining compensation disbursement to reduce risk further.",
    'Possession_Status_Pending':
        "🔴 Prioritize possession handover - escalate coordination with district administration.",
    'Possession_Status_Partial possession':
        "🟡 Follow up on pending possession handovers to close the gap.",
    'Legal_Disputes_Count':
        "⚖️ Fast-track pending legal dispute resolution through designated fast-track courts.",
    'R&R_Progress_Percent':
        "🏘️ Increase rehabilitation & resettlement progress - engage affected families proactively.",
    'Days_3A_to_3D':
        "📄 Reduce administrative delay between 3A and 3D notification stages.",
    'Days_3D_to_Award':
        "📋 Expedite award declaration process to avoid stage-wise bottleneck.",
}

# SIDEBAR - GLOBAL CONTROLS


with st.sidebar:
    st.markdown("## 🎛️ Dashboard Controls")
    st.caption("Use these controls to explore the project portfolio.")

    global_search = st.text_input(
        "🔎 Search projects",
        placeholder="Project ID, district, state..."
    )

    st.markdown("### Risk threshold")
    risk_threshold = st.slider(
        "Show projects above",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
        format="%d%%"
    )

    st.markdown("### Quick filters")
    selected_states = st.multiselect(
        "States",
        options=sorted(df["State"].dropna().unique()),
        default=[]
    )

    selected_types = st.multiselect(
        "Project types",
        options=sorted(df["Project_Type"].dropna().unique()),
        default=[]
    )

    st.divider()

    if st.button("↻ Reset dashboard", use_container_width=True):
        st.rerun()

    st.caption(
        f"Dataset: {len(df):,} projects · "
        f"{df['State'].nunique()} states · "
        f"{df['District'].nunique()} districts"
    )

# GLOBAL FILTERED DATA


filtered_global = df.copy()

if global_search.strip():
    q = global_search.strip().lower()
    mask = (
        filtered_global["Project_ID"].astype(str).str.lower().str.contains(q, na=False)
        | filtered_global["State"].astype(str).str.lower().str.contains(q, na=False)
        | filtered_global["District"].astype(str).str.lower().str.contains(q, na=False)
    )
    filtered_global = filtered_global[mask]

if selected_states:
    filtered_global = filtered_global[
        filtered_global["State"].isin(selected_states)
    ]

if selected_types:
    filtered_global = filtered_global[
        filtered_global["Project_Type"].isin(selected_types)
    ]

if risk_threshold > 0:
    filtered_global = filtered_global[
        filtered_global["Predicted_Risk_Percent"] >= risk_threshold
    ]

# STATUS STRIP


c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(label, value, help_text):
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-help">{help_text}</div>'
        f'</div>'
    )

with c1:
    st.markdown(metric_card(
        "TOTAL PROJECTS",
        f"{len(filtered_global):,}",
        "Current filtered view"
    ), unsafe_allow_html=True)

with c2:
    high_count = (filtered_global["Risk_Category"] == "🔴 High").sum()
    st.markdown(metric_card(
        "🔴 HIGH RISK",
        f"{high_count:,}",
        "Risk ≥ 70%"
    ), unsafe_allow_html=True)

with c3:
    medium_count = (filtered_global["Risk_Category"] == "🟡 Medium").sum()
    st.markdown(metric_card(
        "🟡 MEDIUM RISK",
        f"{medium_count:,}",
        "Risk 40–69.9%"
    ), unsafe_allow_html=True)

with c4:
    low_count = (filtered_global["Risk_Category"] == "🟢 Low").sum()
    st.markdown(metric_card(
        "🟢 LOW RISK",
        f"{low_count:,}",
        "Risk < 40%"
    ), unsafe_allow_html=True)

with c5:
    avg_risk = (
        filtered_global["Predicted_Risk_Percent"].mean()
        if len(filtered_global) else 0
    )
    st.markdown(metric_card(
        "AVERAGE RISK",
        f"{avg_risk:.1f}%",
        "Filtered portfolio average"
    ), unsafe_allow_html=True)

st.write("")

# TABS

(
    tab_overview,
    tab_rankings,
    tab_analytics,
    tab_map,
    tab_predict
) = st.tabs([
    "📊 Overview",
    "📋 Risk Rankings",
    "📈 Analytics",
    "🗺️ GIS Map",
    "🔍 Predict New Project"
])

# OVERVIEW


with tab_overview:
    if len(filtered_global) == 0:
        st.warning("No projects match the current filters.")
    else:
        overall_risk = filtered_global["Predicted_Risk_Percent"].mean()

        left, right = st.columns([1, 1.45])

        with left:
            st.markdown('<div class="section-title">Portfolio Risk</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                show_risk_gauge(
                    overall_risk,
                    "Average Delay Risk"
                ),
                use_container_width=True
            )

        with right:
            st.markdown('<div class="section-title">🚨 Immediate Attention</div>',
                        unsafe_allow_html=True)

            urgent = (
                filtered_global
                .sort_values("Predicted_Risk_Percent", ascending=False)
                .head(5)
            )

            for _, row in urgent.iterrows():
                risk = row["Predicted_Risk_Percent"]
                badge = (
                    "badge-high" if risk >= 70
                    else "badge-medium" if risk >= 40
                    else "badge-low"
                )

                st.markdown(
                    f"""
                    <div class="alert-card">
                        <b>{row['Project_ID']}</b>
                        &nbsp; <span class="badge {badge}">{risk:.1f}% risk</span>
                        <br>
                        <span style="opacity:.7;">
                        {row['District']}, {row['State']} · {row['Project_Type']}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.divider()

        about1, about2 = st.columns(2)

        with about1:
            st.markdown('<div class="section-title">What is BhuNirvighna-Ai?</div>',
                        unsafe_allow_html=True)
            st.write(
                "Land acquisition for infrastructure projects - highways, railways, "
                "power lines, irrigation canals - can be delayed by compensation, "
                "legal disputes, rehabilitation, possession, or administrative backlog. "
                "BhuNirvighna-Ai turns these signals into an early warning system."
            )

        with about2:
            st.markdown('<div class="section-title">How it works</div>',
                        unsafe_allow_html=True)
            st.write(
                "A machine learning model evaluates historical project records and "
                "produces a delay-risk score. SHAP explains the main risk drivers, "
                "while the recommendation engine converts those drivers into "
                "plain-language actions."
            )

        st.divider()

        st.markdown('<div class="section-title">📦 Dataset Snapshot</div>',
                    unsafe_allow_html=True)

        d1, d2, d3 = st.columns(3)
        d1.metric("States", df["State"].nunique())
        d2.metric("Districts", df["District"].nunique())
        d3.metric("Project Types", df["Project_Type"].nunique())

        st.caption(
            "The dataset is a synthetic prototype with field structures and value "
            "ranges modeled on publicly documented patterns of land-acquisition "
            "administration in India."
        )

# RISK RANKINGS

with tab_rankings:
    st.subheader("📋 Interactive Project Risk Rankings")

    f1, f2, f3 = st.columns([1, 1, 1])

    with f1:
        local_risk = st.multiselect(
            "Risk category",
            ["🔴 High", "🟡 Medium", "🟢 Low"],
            default=["🔴 High", "🟡 Medium", "🟢 Low"],
            key="ranking_risk"
        )

    with f2:
        ranking_sort = st.selectbox(
            "Sort by",
            ["Risk: High → Low", "Risk: Low → High", "Project ID"],
            key="ranking_sort"
        )

    with f3:
        rows_to_show = st.selectbox(
            "Rows",
            [25, 50, 100, 250],
            index=1
        )

    ranking_df = filtered_global[
        filtered_global["Risk_Category"].isin(local_risk)
    ].copy()

    if ranking_sort == "Risk: High → Low":
        ranking_df = ranking_df.sort_values(
            "Predicted_Risk_Percent", ascending=False
        )
    elif ranking_sort == "Risk: Low → High":
        ranking_df = ranking_df.sort_values(
            "Predicted_Risk_Percent", ascending=True
        )
    else:
        ranking_df = ranking_df.sort_values("Project_ID")

    display_cols = [
        "Project_ID", "State", "District", "Project_Type",
        "Predicted_Risk_Percent", "Risk_Category",
        "Compensation_Status", "Possession_Status"
    ]

    st.dataframe(
        ranking_df[display_cols].head(rows_to_show),
        use_container_width=True,
        height=470,
        column_config={
            "Predicted_Risk_Percent": st.column_config.ProgressColumn(
                "Risk %",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "Project_ID": "Project",
            "Project_Type": "Type",
            "Compensation_Status": "Compensation",
            "Possession_Status": "Possession"
        },
        hide_index=True
    )

    st.caption(
        f"Showing {min(rows_to_show, len(ranking_df)):,} of "
        f"{len(ranking_df):,} filtered projects."
    )

    csv_data = ranking_df[display_cols].to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download filtered rankings",
        data=csv_data,
        file_name="bhuNirvighna_filtered_risk_rankings.csv",
        mime="text/csv"
    )

    st.divider()

    st.markdown("### 🔎 Project Detail Explorer")

    if len(ranking_df):
        selected_project = st.selectbox(
            "Select a project to inspect",
            ranking_df["Project_ID"].tolist(),
            key="selected_project"
        )

        project_row = ranking_df[
            ranking_df["Project_ID"] == selected_project
        ].iloc[0]

        a, b, c, d = st.columns(4)
        a.metric("Risk", f"{project_row['Predicted_Risk_Percent']:.1f}%")
        b.metric("State", project_row["State"])
        c.metric("Compensation", project_row["Compensation_Status"])
        d.metric("Possession", project_row["Possession_Status"])

# ANALYTICS

with tab_analytics:
    st.subheader("📈 Interactive Analytics")

    if len(filtered_global) == 0:
        st.warning("No data available for analytics under the current filters.")
    else:
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("#### Delay Rate by Compensation Status")

            comp_chart = (
                filtered_global
                .groupby("Compensation_Status")["Delay_Label"]
                .mean()
                .reset_index()
            )
            comp_chart["Delay_Label"] *= 100

            fig1 = px.bar(
                comp_chart,
                x="Compensation_Status",
                y="Delay_Label",
                labels={"Delay_Label": "Delay Rate (%)"},
                text_auto=".1f"
            )
            fig1.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)

        with chart2:
            st.markdown("#### Risk Distribution")

            fig2 = px.histogram(
                filtered_global,
                x="Predicted_Risk_Percent",
                nbins=30,
                labels={"Predicted_Risk_Percent": "Predicted Risk (%)"}
            )
            fig2.update_layout(
                height=390,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        st.markdown("#### Average Risk by State")

        state_risk = (
            filtered_global
            .groupby("State")["Predicted_Risk_Percent"]
            .mean()
            .reset_index()
            .sort_values("Predicted_Risk_Percent", ascending=False)
        )

        fig3 = px.bar(
            state_risk,
            x="State",
            y="Predicted_Risk_Percent",
            labels={"Predicted_Risk_Percent": "Avg Predicted Risk (%)"},
            text_auto=".1f"
        )
        fig3.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=20, b=80)
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.divider()

        left, right = st.columns(2)

        with left:
            st.markdown("#### Risk by Project Type")

            type_risk = (
                filtered_global
                .groupby("Project_Type")["Predicted_Risk_Percent"]
                .mean()
                .reset_index()
                .sort_values("Predicted_Risk_Percent", ascending=False)
            )

            fig4 = px.bar(
                type_risk,
                x="Predicted_Risk_Percent",
                y="Project_Type",
                orientation="h",
                labels={"Predicted_Risk_Percent": "Average Risk (%)"}
            )
            fig4.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig4, use_container_width=True)

        with right:
            st.markdown("#### Risk Category Composition")

            category_counts = (
                filtered_global["Risk_Category"]
                .value_counts()
                .rename_axis("Risk Category")
                .reset_index(name="Projects")
            )

            fig5 = px.pie(
                category_counts,
                names="Risk Category",
                values="Projects",
                hole=.52
            )
            fig5.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=20, b=20)
            )
            st.plotly_chart(fig5, use_container_width=True)

# GIS MAP


with tab_map:
    st.subheader("🗺️ GIS Risk Heatmap - State-wise")

    state_coords = {
        'Rajasthan': (27.0238, 74.2179),
        'Madhya Pradesh': (23.4733, 77.9479),
        'Odisha': (20.9517, 85.0985),
        'West Bengal': (22.9868, 87.8550),
        'Bihar': (25.0961, 85.3131),
        'Gujarat': (22.2587, 71.1924),
        'Tamil Nadu': (11.1271, 78.6569),
        'Uttar Pradesh': (26.8467, 80.9462),
        'Maharashtra': (19.7515, 75.7139),
        'Karnataka': (15.3173, 75.7139),
        'Punjab': (31.1471, 75.3412),
        'Haryana': (29.0588, 76.0856),
        'Kerala': (10.8505, 76.2711),
        'Andhra Pradesh': (15.9129, 79.7400),
        'Telangana': (18.1124, 79.0193),
        'Assam': (26.2006, 92.9376),
    }

    state_summary = (
        filtered_global
        .groupby("State")
        .agg(
            Avg_Risk=("Predicted_Risk_Percent", "mean"),
            Project_Count=("Project_ID", "count")
        )
        .reset_index()
    )

    m = folium.Map(
        location=[22.5, 80],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    for _, row in state_summary.iterrows():
        coords = state_coords.get(row["State"])

        if coords:
            risk = row["Avg_Risk"]

            color = (
                "#a43b3b" if risk >= 50
                else "#b5871a" if risk >= 25
                else "#4a7a1e"
            )

            folium.CircleMarker(
                location=coords,
                radius=8 + (risk / 5),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.72,
                popup=folium.Popup(
                    f"""
                    <b>{row['State']}</b><br>
                    Average Risk: {risk:.1f}%<br>
                    Projects: {row['Project_Count']}
                    """,
                    max_width=300
                ),
                tooltip=f"{row['State']} - {risk:.1f}% risk"
            ).add_to(m)

    st_folium(
        m,
        use_container_width=True,
        height=560,
        returned_objects=[]
    )

    st.caption(
        "State-level centroid markers are used for prototype visualization. "
        "A production version can replace them with precise district boundary GeoJSON."
    )

    st.markdown("### State Risk Summary")

    map_table = state_summary.sort_values(
        "Avg_Risk", ascending=False
    ).copy()

    st.dataframe(
        map_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg_Risk": st.column_config.ProgressColumn(
                "Average Risk %",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "Project_Count": "Projects"
        }
    )

# TAB 5 - PREDICT NEW PROJECT
# ORIGINAL PREDICTION LOGIC PRESERVED


with tab_predict:
    st.header("🔍 Predict Risk, Cost & Feasibility")
    st.caption(
        "Enter project information below. The existing ML models, cost model, "
        "SHAP explainer and feasibility calculation are used unchanged."
    )

    with st.form("prediction_form"):
        st.markdown("### 🏗️ Project Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            input_state = st.selectbox(
                "State",
                sorted(df["State"].unique())
            )
            input_district = st.selectbox(
                "District",
                sorted(df["District"].unique())
            )
            input_project_type = st.selectbox(
                "Project Type",
                sorted(df["Project_Type"].unique())
            )
            input_land_type = st.selectbox(
                "Land Type",
                ["Agricultural", "Forest", "Tribal", "Government", "Private"]
            )

            input_land_required = st.number_input(
                "Land Required (Hectares)",
                min_value=1.0,
                value=100.0,
                step=10.0
            )

            input_land_acquired = st.number_input(
                "Land Acquired So Far (Hectares)",
                min_value=0.0,
                value=50.0,
                step=5.0
            )

        with col2:
            input_affected_families = st.number_input(
                "Affected Families",
                min_value=0,
                value=200,
                step=10
            )

            input_displaced_families = st.number_input(
                "Displaced Families",
                min_value=0,
                value=50,
                step=5
            )

            input_compensation_amount = st.number_input(
                "Compensation Amount (Crore)",
                min_value=0.0,
                value=10.0,
                step=1.0
            )

            input_rr_amount = st.number_input(
                "R&R Amount (Crore)",
                min_value=0.0,
                value=2.0,
                step=.5
            )

            input_rr_progress = st.slider(
                "R&R Progress (%)",
                0,
                100,
                50
            )

        with col3:
            input_compensation_status = st.selectbox(
                "Compensation Status",
                sorted(df["Compensation_Status"].unique())
            )

            input_possession_status = st.selectbox(
                "Possession Status",
                sorted(df["Possession_Status"].unique())
            )

            input_legal_disputes = st.number_input(
                "Legal Disputes Count",
                min_value=0,
                value=1,
                step=1
            )

            input_days_3a_to_3d = st.number_input(
                "Days: 3A → 3D Notification",
                min_value=0,
                value=90,
                step=10
            )

            input_days_3d_to_award = st.number_input(
                "Days: 3D → Award",
                min_value=0,
                value=90,
                step=10
            )

        submitted = st.form_submit_button(
            "🔮 Predict Project",
            use_container_width=True
        )

    # Live helper information before prediction
    acquired_pct = (
        min(100, input_land_acquired / input_land_required * 100)
        if input_land_required > 0 else 0
    )

    st.markdown("### 📍 Input Snapshot")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.metric("Land Acquired", f"{acquired_pct:.1f}%")
        st.progress(acquired_pct / 100)

    with s2:
        st.metric("R&R Progress", f"{input_rr_progress}%")
        st.progress(input_rr_progress / 100)

    with s3:
        dispute_state = (
            "High attention" if input_legal_disputes >= 5
            else "Monitor" if input_legal_disputes > 0
            else "No disputes entered"
        )
        st.metric("Legal Disputes", input_legal_disputes)
        st.caption(dispute_state)

    if submitted:
        # ----------------------------------------------------
        # BUILD FEATURE ROW - ORIGINAL LOGIC
        # ----------------------------------------------------
        new_project = pd.DataFrame([{
            "State": input_state,
            "District": input_district,
            "Project_Type": input_project_type,
            "Land_Required_Hectare": input_land_required,
            "Land_Acquired_Hectare": input_land_acquired,
            "Affected_Families": input_affected_families,
            "Displaced_Families": input_displaced_families,
            "Compensation_Amount_Crore": input_compensation_amount,
            "RR_Amount_Crore": input_rr_amount,
            "Compensation_Status": input_compensation_status,
            "R&R_Progress_Percent": input_rr_progress,
            "Legal_Disputes_Count": input_legal_disputes,
            "Possession_Status": input_possession_status,
            "Days_Sanction_to_3A": 60,
            "Days_3A_to_3D": input_days_3a_to_3d,
            "Days_3D_to_Award": input_days_3d_to_award,
            "Days_Award_to_Payment": 60,
            "Days_Payment_to_Possession": 60,
        }])

        new_encoded = pd.get_dummies(
            new_project,
            columns=[
                "State", "District", "Project_Type",
                "Compensation_Status", "Possession_Status"
            ]
        )

        new_encoded = new_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        risk_prob = model.predict_proba(new_encoded)[:, 1][0]
        risk_percent = round(risk_prob * 100, 1)

        # ----------------------------------------------------
        # COST ESTIMATION - ORIGINAL LOGIC
        # ----------------------------------------------------
        cost_input = pd.DataFrame([{
            "State": input_state,
            "Project_Type": input_project_type,
            "Land_Required_Hectare": input_land_required,
            "Affected_Families": input_affected_families,
            "Displaced_Families": input_displaced_families,
        }])

        cost_input_encoded = pd.get_dummies(
            cost_input,
            columns=["State", "Project_Type"]
        )

        cost_input_encoded = cost_input_encoded.reindex(
            columns=cost_feature_columns,
            fill_value=0
        )

        predicted_cost = cost_model.predict(cost_input_encoded)[0]

        # ----------------------------------------------------
        # LAND FEASIBILITY - ORIGINAL LOGIC
        # ----------------------------------------------------
        feasibility_score = calculate_land_feasibility(
            land_type=input_land_type,
            land_required=input_land_required,
            land_acquired_so_far=input_land_acquired,
            affected_families=input_affected_families,
            legal_disputes=input_legal_disputes,
            rr_progress=input_rr_progress
        )

        st.success("✅ Prediction complete")

        st.markdown("## 🎯 Prediction Results")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "💰 Estimated Cost",
                f"₹{predicted_cost:.1f} Crore"
            )

        with m2:
            st.metric(
                "🏞️ Land Feasibility",
                f"{feasibility_score}/100"
            )

        with m3:
            risk_label = (
                "🔴 High" if risk_percent >= 70
                else "🟡 Medium" if risk_percent >= 40
                else "🟢 Low"
            )
            st.metric(
                "📊 Risk Category",
                risk_label,
                delta=f"{risk_percent:.1f}% risk"
            )

        st.divider()

        # ----------------------------------------------------
        # GAUGE + SHAP
        # ----------------------------------------------------
        col_gauge, col_shap = st.columns([1, 1.35])

        with col_gauge:
            st.plotly_chart(
                show_risk_gauge(
                    risk_percent,
                    "Predicted Delay Risk"
                ),
                use_container_width=True
            )

            st.markdown(
                f"""
                <div class="info-card">
                    <b>Risk interpretation</b><br>
                    <span style="opacity:.75;">
                    {risk_label} · Model probability: {risk_percent:.1f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_shap:
            st.markdown("### 🧠 Why did the model predict this?")

            new_shap_values = explainer.shap_values(new_encoded)

            shap_impact = pd.DataFrame({
                "Feature": new_encoded.columns,
                "Impact": new_shap_values[0]
            }).sort_values(
                "Impact",
                ascending=False
            )

            top_positive = shap_impact[
                shap_impact["Impact"] > 0
            ].head(8).copy()

            if len(top_positive):
                shap_fig = px.bar(
                    top_positive.sort_values("Impact"),
                    x="Impact",
                    y="Feature",
                    orientation="h",
                    labels={"Impact": "SHAP impact"}
                )
                shap_fig.update_layout(
                    height=390,
                    margin=dict(l=10, r=10, t=20, b=20)
                )
                st.plotly_chart(
                    shap_fig,
                    use_container_width=True
                )
            else:
                st.success(
                    "✅ No positive SHAP risk drivers detected."
                )

        # ----------------------------------------------------
        # RECOMMENDATIONS
        # ----------------------------------------------------
        st.divider()
        st.markdown("### 💡 Recommended Actions")

        top_risk_drivers = shap_impact[
            shap_impact["Impact"] > 0
        ].head(5)

        if len(top_risk_drivers) == 0:
            st.success(
                "✅ No significant risk drivers detected - "
                "this project shows a healthy risk profile."
            )
        else:
            for _, row in top_risk_drivers.iterrows():
                feature_name = row["Feature"]
                rec = recommendation_map.get(feature_name)

                if rec is None:
                    for key in recommendation_map:
                        if key.split("_")[0] in feature_name:
                            rec = recommendation_map[key]
                            break

                if rec:
                    st.markdown(
                        f"""
                        <div class="action-card">
                            <b>{feature_name}</b><br>
                            {rec}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.info(
                        f"⚠️ Monitor **{feature_name}** - "
                        "contributing to elevated risk."
                    )

        # RESULT SUMMARY
        
        st.divider()
        st.markdown("### 📄 Project Summary")

        summary_df = pd.DataFrame({
            "Parameter": [
                "State",
                "District",
                "Project Type",
                "Land Type",
                "Land Required",
                "Land Acquired",
                "Affected Families",
                "Displaced Families",
                "Compensation Status",
                "Possession Status",
                "Legal Disputes",
                "R&R Progress",
                "Predicted Risk",
                "Estimated Cost",
                "Land Feasibility"
            ],
            "Value": [
                input_state,
                input_district,
                input_project_type,
                input_land_type,
                f"{input_land_required:.1f} hectares",
                f"{input_land_acquired:.1f} hectares",
                input_affected_families,
                input_displaced_families,
                input_compensation_status,
                input_possession_status,
                input_legal_disputes,
                f"{input_rr_progress}%",
                f"{risk_percent:.1f}%",
                f"₹{predicted_cost:.1f} Crore",
                f"{feasibility_score}/100"
            ]
        })

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

# FOOTER


st.divider()
st.markdown(
    """<div style="text-align:center;padding:14px 10px 6px;opacity:.78;font-size:.88rem;">
    Developed by <b>KiranPy</b> &nbsp;·&nbsp;
    <a href="https://github.com/Butkii025/BhuNirvighna-Ai" target="_blank" style="text-decoration:none;font-weight:700;">GitHub Repository ↗</a><br>
    <span style="font-size:.78rem;opacity:.65;">BhuNirvighna-Ai · AI-powered land acquisition delay early-warning prototype</span>
    </div>""", unsafe_allow_html=True
)
