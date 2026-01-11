import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.database import get_engine, get_session
from src.models import Run, Signal, TargetWeight, Order, PortfolioValue, Position
import os
from sqlalchemy import desc

st.set_page_config(page_title="Investment Bot Dashboard", layout="wide")

@st.cache_resource
def get_db_connection():
    return get_engine()

engine = get_db_connection()

def load_portfolio_history():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM portfolio_values ORDER BY date", conn)
    return df

def load_latest_positions():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM positions WHERE date = (SELECT max(date) FROM positions)", conn)
    return df

def load_recent_runs():
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM runs ORDER BY started_at DESC LIMIT 10", conn)
    return df

st.title("📈 Investment Bot Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Decisions", "Holdings", "System Health"])

if page == "Overview":
    st.header("Portfolio Overview")
    
    history = load_portfolio_history()
    if not history.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history['date'], y=history['equity'], name='Portfolio Equity'))
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        current_equity = history['equity'].iloc[-1]
        start_equity = history['equity'].iloc[0]
        total_return = (current_equity / start_equity - 1) * 100
        
        col1.metric("Current Equity", f"${current_equity:,.2f}")
        col2.metric("Total Return", f"{total_return:.2f}%")
        col3.metric("Cash Balance", f"${history['cash'].iloc[-1]:,.2f}")
    else:
        st.info("No portfolio history data available yet.")

elif page == "Decisions":
    st.header("Recent Strategy Decisions")
    
    runs = load_recent_runs()
    rebalance_runs = runs[runs['run_type'] == 'rebalance']
    
    if not rebalance_runs.empty:
        selected_run_id = st.selectbox("Select Rebalance Run", rebalance_runs['id'])
        
        # Load signals for this run
        with engine.connect() as conn:
            signals = pd.read_sql(f"SELECT * FROM signals WHERE run_id = {selected_run_id} ORDER BY rank", conn)
            targets = pd.read_sql(f"SELECT * FROM target_weights WHERE run_id = {selected_run_id}", conn)
        
        st.subheader("Top Ranked Tickers")
        st.dataframe(signals.head(20))
        
        st.subheader("Target Weights")
        st.dataframe(targets)
    else:
        st.info("No rebalance runs found.")

elif page == "Holdings":
    st.header("Current Holdings")
    positions = load_latest_positions()
    if not positions.empty:
        st.dataframe(positions)
        
        # Pie chart of weights
        fig = go.Figure(data=[go.Pie(labels=positions['ticker'], values=positions['market_value'])])
        st.plotly_chart(fig)
    else:
        st.info("No open positions.")

elif page == "System Health":
    st.header("System Health & Logs")
    
    st.subheader("Recent Runs")
    st.dataframe(load_recent_runs())
    
    # Simple alert simulation or real alerts if implemented
    st.subheader("Recent Alerts")
    with engine.connect() as conn:
        alerts = pd.read_sql("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 20", conn)
    st.dataframe(alerts)



