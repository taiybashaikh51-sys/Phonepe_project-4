import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PhonePe Transaction Insights",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0B1220; color: #ffffff; }
    .main-header {
        background: linear-gradient(135deg, #1A2333, #0B1220);
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .section-title {
        color: #8B5CF6;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(139,92,246,0.3);
    }
    div[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(139,92,246,0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('phonepe_transaction.csv')
    df.columns = [col.strip() for col in df.columns]
    df['State'] = df['State'].str.replace('-', ' ').str.title()
    df['Amount_Cr'] = df['Amount'] / 1e7
    df['Period'] = df['Year'].astype(str) + ' Q' + df['Quarter'].astype(str)
    return df

df = load_data()

with st.sidebar:
    st.markdown("### 💰 PhonePe Analytics")
    st.markdown("---")
    page = st.radio("📊 Select Page", [
        "🏠 Overview",
        "📊 State Analysis",
        "📈 Trend Analysis",
        "🔮 ML Prediction",
        "📋 Data Explorer"
    ])
    st.markdown("---")
    year_filter = st.multiselect("📅 Year",
        options=sorted(df['Year'].unique()),
        default=sorted(df['Year'].unique()))
    type_filter = st.multiselect("💳 Transaction Type",
        options=df['Type'].unique().tolist(),
        default=df['Type'].unique().tolist())

filtered = df[df['Year'].isin(year_filter) & df['Type'].isin(type_filter)]

st.markdown("""
<div class="main-header">
    <h1 style="color:#8B5CF6; margin:0; font-size:2rem;">💰 PhonePe Transaction Insights</h1>
    <p style="color:#94A3B8; margin:0.5rem 0 0 0;">Business Intelligence Dashboard | SQL + Python + ETL Analytics</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📊 Total Records", f"{len(filtered):,}")
with col2:
    st.metric("💰 Total Amount", f"₹{filtered['Amount_Cr'].sum():.1f}Cr")
with col3:
    st.metric("🔢 Total Count", f"{filtered['Count'].sum()/1e6:.1f}M")
with col4:
    st.metric("🏛️ States", f"{filtered['State'].nunique()}")
with col5:
    st.metric("📅 Years", f"{filtered['Year'].nunique()}")

st.markdown("---")

if page == "🏠 Overview":
    st.markdown('<p class="section-title">📊 Transaction Overview</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        type_amount = filtered.groupby('Type')['Amount_Cr'].sum().reset_index()
        fig = px.pie(type_amount, values='Amount_Cr', names='Type',
            title='Transaction Amount by Type',
            color_discrete_sequence=['#8B5CF6','#3B82F6','#06B6D4','#22C55E','#F97316'])
        fig.update_layout(paper_bgcolor='#1A2333', font=dict(color='#94A3B8'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        yearly = filtered.groupby('Year').agg({'Amount_Cr':'sum','Count':'sum'}).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=yearly['Year'], y=yearly['Amount_Cr'],
            name='Amount (Cr)', marker_color='#8B5CF6'))
        fig2.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'), title='Yearly Transaction Amount',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top_states = filtered.groupby('State')['Amount_Cr'].sum().sort_values(ascending=False).head(10).reset_index()
        fig3 = px.bar(top_states, x='Amount_Cr', y='State',
            orientation='h', color='Amount_Cr',
            color_continuous_scale='Purples',
            title='Top 10 States by Amount')
        fig3.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        quarterly = filtered.groupby(['Year','Quarter'])['Amount_Cr'].sum().reset_index()
        quarterly['Period'] = quarterly['Year'].astype(str) + ' Q' + quarterly['Quarter'].astype(str)
        fig4 = px.line(quarterly, x='Period', y='Amount_Cr',
            markers=True, title='Quarterly Transaction Trend',
            color_discrete_sequence=['#8B5CF6'])
        fig4.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickangle=45),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig4, use_container_width=True)

elif page == "📊 State Analysis":
    st.markdown('<p class="section-title">🏛️ State-wise Analysis</p>', unsafe_allow_html=True)

    state_data = filtered.groupby('State').agg({
        'Amount_Cr': 'sum',
        'Count': 'sum'
    }).reset_index().sort_values('Amount_Cr', ascending=False)

    fig = px.bar(state_data.head(15), x='State', y='Amount_Cr',
        color='Amount_Cr', color_continuous_scale='Purples',
        title='Top 15 States by Transaction Amount')
    fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickangle=45),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.scatter(state_data, x='Count', y='Amount_Cr',
            hover_name='State', size='Amount_Cr',
            color='Amount_Cr', color_continuous_scale='Purples',
            title='Transaction Count vs Amount by State')
        fig2.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        selected_state = st.selectbox("🏛️ Select State",
            sorted(filtered['State'].unique()))
        state_df = filtered[filtered['State'] == selected_state]
        state_type = state_df.groupby('Type')['Amount_Cr'].sum().reset_index()
        fig3 = px.pie(state_type, values='Amount_Cr', names='Type',
            title=f'{selected_state} - Transaction by Type',
            color_discrete_sequence=['#8B5CF6','#3B82F6','#06B6D4'])
        fig3.update_layout(paper_bgcolor='#1A2333', font=dict(color='#94A3B8'))
        st.plotly_chart(fig3, use_container_width=True)

elif page == "📈 Trend Analysis":
    st.markdown('<p class="section-title">📈 Transaction Trends</p>', unsafe_allow_html=True)

    quarterly = filtered.groupby(['Year','Quarter','Type']).agg({
        'Amount_Cr':'sum','Count':'sum'}).reset_index()
    quarterly['Period'] = quarterly['Year'].astype(str) + ' Q' + quarterly['Quarter'].astype(str)

    fig = px.line(quarterly, x='Period', y='Amount_Cr', color='Type',
        markers=True, title='Quarterly Amount Trend by Type')
    fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickangle=45),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        yoy = filtered.groupby(['Year','Type'])['Amount_Cr'].sum().reset_index()
        fig2 = px.bar(yoy, x='Year', y='Amount_Cr', color='Type',
            barmode='group', title='Year-over-Year Growth by Type',
            color_discrete_sequence=['#8B5CF6','#3B82F6','#06B6D4','#22C55E'])
        fig2.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        quarter_avg = filtered.groupby('Quarter')['Amount_Cr'].mean().reset_index()
        quarter_avg['Quarter_Name'] = quarter_avg['Quarter'].map(
            {1:'Q1 (Jan-Mar)',2:'Q2 (Apr-Jun)',3:'Q3 (Jul-Sep)',4:'Q4 (Oct-Dec)'})
        fig3 = px.bar(quarter_avg, x='Quarter_Name', y='Amount_Cr',
            color='Amount_Cr', color_continuous_scale='Purples',
            title='Average Amount by Quarter')
        fig3.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
            font=dict(color='#94A3B8'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
        st.plotly_chart(fig3, use_container_width=True)

elif page == "🔮 ML Prediction":
    st.markdown('<p class="section-title">🔮 Transaction Amount Prediction</p>', unsafe_allow_html=True)

    ml_df = df.copy()
    le_state = LabelEncoder()
    le_type = LabelEncoder()
    ml_df['State_enc'] = le_state.fit_transform(ml_df['State'])
    ml_df['Type_enc'] = le_type.fit_transform(ml_df['Type'])

    features = ['Year', 'Quarter', 'State_enc', 'Type_enc', 'Count']
    X = ml_df[features]
    y = ml_df['Amount_Cr']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    with st.spinner("Training ML Model..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📊 R² Score", f"{r2:.4f}")
    with col2: st.metric("📉 MAE", f"₹{mae:.2f}Cr")
    with col3: st.metric("✅ Accuracy", f"{r2*100:.1f}%")

    fig = go.Figure()
    sample = min(200, len(y_test))
    fig.add_trace(go.Scatter(x=list(range(sample)), y=y_test.values[:sample],
        name='Actual', line=dict(color='#8B5CF6', width=2)))
    fig.add_trace(go.Scatter(x=list(range(sample)), y=y_pred[:sample],
        name='Predicted', line=dict(color='#F97316', width=2, dash='dot')))
    fig.update_layout(paper_bgcolor='#1A2333', plot_bgcolor='#1A2333',
        font=dict(color='#94A3B8'), title='Actual vs Predicted Amount',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-title">🎯 Predict Transaction Amount</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        pred_year = st.slider("📅 Year", 2018, 2030, 2024)
        pred_quarter = st.selectbox("📅 Quarter", [1, 2, 3, 4])
    with col2:
        pred_state = st.selectbox("🏛️ State", sorted(df['State'].unique()))
        pred_type = st.selectbox("💳 Type", df['Type'].unique())
    with col3:
        pred_count = st.number_input("🔢 Transaction Count", 1000, 10000000, 100000)

    pred_state_enc = le_state.transform([pred_state])[0] if pred_state in le_state.classes_ else 0
    pred_type_enc = le_type.transform([pred_type])[0] if pred_type in le_type.classes_ else 0

    pred_amount = model.predict([[pred_year, pred_quarter, pred_state_enc, pred_type_enc, pred_count]])[0]

    st.markdown(f"""
    <div style="background:#1A2333; border:2px solid #8B5CF6; border-radius:16px;
    padding:2rem; text-align:center; margin-top:1rem;">
        <h2 style="color:#8B5CF6; margin:0;">💰 Predicted Amount: ₹{pred_amount:.2f} Cr</h2>
        <p style="color:#94A3B8; margin:0.5rem 0 0 0;">Based on Random Forest Regression</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "📋 Data Explorer":
    st.markdown('<p class="section-title">📋 Data Explorer</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        state_sel = st.selectbox("🏛️ State", ['All'] + sorted(df['State'].unique()))
    with col2:
        type_sel = st.selectbox("💳 Type", ['All'] + df['Type'].unique().tolist())
    with col3:
        year_sel = st.selectbox("📅 Year", ['All'] + sorted(df['Year'].unique().tolist()))

    show_df = df.copy()
    if state_sel != 'All': show_df = show_df[show_df['State'] == state_sel]
    if type_sel != 'All': show_df = show_df[show_df['Type'] == type_sel]
    if year_sel != 'All': show_df = show_df[show_df['Year'] == int(year_sel)]

    st.dataframe(show_df, use_container_width=True, height=400)
    csv = show_df.to_csv(index=False)
    st.download_button("📥 Download Data", csv, "phonepe_data.csv", "text/csv")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.8rem; padding:1rem;">
    💰 PhonePe Transaction Insights | Built by <strong style="color:#8B5CF6">Taiyba Shaikh</strong> | Data Analytics Project
</div>
""", unsafe_allow_html=True)