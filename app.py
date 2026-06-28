import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration & Session State Initialization ---
st.set_page_config(page_title="Demand Planning Tool", layout="wide")

# Initialize "database" in session state
if 'historical_data' not in st.session_state:
    st.session_state['historical_data'] = None
if 'system_forecast' not in st.session_state:
    st.session_state['system_forecast'] = None
if 'consensus_forecast' not in st.session_state:
    st.session_state['consensus_forecast'] = None

# --- Sidebar Navigation ---
st.sidebar.title("DILOP Workflow")
page = st.sidebar.radio("Navigate to:", [
    "1. Data Management", 
    "2. System Forecast Generation", 
    "3. Demand Review & Collaboration", 
    "4. Post-Game Analysis"
])

# --- Page 1: Data Management ---
if page == "1. Data Management":
    st.title("Data Management (Collect Input Data)")
    st.markdown("Upload your transactional data (historical sales) here.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Ensure a Date column exists
        if 'Date' in df.columns and 'Sales' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            st.session_state['historical_data'] = df
            st.success("Data successfully uploaded and written to database!")
            st.dataframe(df.head())
        else:
            st.error("CSV must contain 'Date' and 'Sales' columns.")

# --- Page 2: System Forecast Generation ---
elif page == "2. System Forecast Generation":
    st.title("System Forecast Generation")
    
    if st.session_state['historical_data'] is None:
        st.warning("Please upload data in Step 1 first.")
    else:
        df = st.session_state['historical_data'].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            horizon = st.number_input("Forecast Horizon (Periods)", min_value=1, max_value=24, value=6)
        with col2:
            method = st.selectbox("Statistical Method", ["Simple Moving Average (SMA)", "Naive"])
            
        if st.button("Generate Baseline Forecast"):
            last_date = df['Date'].max()
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
            
            if method == "Simple Moving Average (SMA)":
                # Average of last 3 periods as a basic SMA
                sma_val = df['Sales'].tail(3).mean()
                forecast_vals = [sma_val] * horizon
            else:
                # Naive: Last known value
                naive_val = df['Sales'].iloc[-1]
                forecast_vals = [naive_val] * horizon
                
            forecast_df = pd.DataFrame({'Date': future_dates, 'Baseline_Forecast': forecast_vals})
            st.session_state['system_forecast'] = forecast_df
            st.success("System-recommended forecast generated.")
            st.dataframe(forecast_df)

# --- Page 3: Demand Review & Collaboration ---
elif page == "3. Demand Review & Collaboration":
    st.title("Demand Review (Adjust & Finalize)")
    
    if st.session_state['system_forecast'] is None:
        st.warning("Please generate a system forecast in Step 2 first.")
    else:
        st.markdown("Analyze the baseline and manually input overrides (e.g., adding marketing adjustments).")
        df = st.session_state['system_forecast'].copy()
        
        # Add columns for manual adjustments if they don't exist
        if 'Manual_Override' not in df.columns:
            df['Manual_Override'] = 0.0
            
        # Display data editor for user input
        edited_df = st.data_editor(
            df,
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", disabled=True),
                "Baseline_Forecast": st.column_config.NumberColumn("Baseline", disabled=True),
                "Manual_Override": st.column_config.NumberColumn("Override (+/- volume)", default=0.0)
            },
            hide_index=True
        )
        
        # Calculate Final Consensus
        edited_df['Consensus_Forecast'] = edited_df['Baseline_Forecast'] + edited_df['Manual_Override']
        
        if st.button("Finalize Consensus"):
            st.session_state['consensus_forecast'] = edited_df
            st.success("Consensus forecast finalized and published!")
            
            # Visualization using Plotly
            hist = st.session_state['historical_data']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Sales'], mode='lines+markers', name='Historical Sales'))
            fig.add_trace(go.Scatter(x=edited_df['Date'], y=edited_df['Baseline_Forecast'], mode='lines+markers', name='Baseline Forecast', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=edited_df['Date'], y=edited_df['Consensus_Forecast'], mode='lines+markers', name='Consensus Forecast', line=dict(color='green')))
            fig.update_layout(title="Historical vs Consensus Forecast", xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)

# --- Page 4: Post-Game Analysis ---
elif page == "4. Post-Game Analysis":
    st.title("Post-Game Analysis")
    st.markdown("Analyze the accuracy of predictions against actual sales data.")
    
    # For demonstration, we simulate actuals for the forecast periods to calculate errors
    if st.session_state['consensus_forecast'] is None:
        st.warning("Please complete the consensus process to measure KPIs.")
    else:
        results = st.session_state['consensus_forecast'].copy()
        
        # Simulating actuals slightly off from the consensus for demonstration purposes
        np.random.seed(42)
        variance = np.random.uniform(0.8, 1.2, len(results))
        results['Actuals'] = (results['Consensus_Forecast'] * variance).round(2)
        
        # Calculate KPIs
        # Absolute Error
        results['Abs_Error'] = abs(results['Actuals'] - results['Consensus_Forecast'])
        # Bias: (Forecast / Actual - 1) * 100
        results['Bias_%'] = ((results['Consensus_Forecast'] / results['Actuals']) - 1) * 100
        # MAPE
        mape = (results['Abs_Error'] / results['Actuals']).mean() * 100
        
        st.metric(label="Overall System MAPE", value=f"{mape:.2f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Forecast vs Actuals")
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(x=results['Date'], y=results['Actuals'], name='Actuals'))
            fig_acc.add_trace(go.Scatter(x=results['Date'], y=results['Consensus_Forecast'], mode='lines+markers', name='Forecast', line=dict(color='orange', width=3)))
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with col2:
            st.subheader("Forecast Bias Over Time")
            fig_bias = px.bar(results, x='Date', y='Bias_%', color='Bias_%', color_continuous_scale='RdBu', title="Positive = Over-forecast | Negative = Under-forecast")
            st.plotly_chart(fig_bias, use_container_width=True)
            
        st.dataframe(results)