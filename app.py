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
if 'filtered_historical_total' not in st.session_state:
    st.session_state['filtered_historical_total'] = None
if 'best_fit_results' not in st.session_state:
    st.session_state['best_fit_results'] = None

# --- Main Navigation ---
st.sidebar.title("DILOP Workflow")
page = st.sidebar.radio("Navigate to:", [
    "1. Data Management", 
    "2. System Forecast Generation", 
    "3. Demand Review & Collaboration", 
    "4. Post-Game Analysis"
])

st.sidebar.divider()

# --- Page 1: Data Management ---
if page == "1. Data Management":
    st.title("Data Management (Collect Input Data)")
    st.markdown("Upload your transactional data. Ensure it contains the necessary hierarchies and causal drivers.")
    
    required_columns = [
        'Date', 'Region', 'State', 'Brand', 'SKU', 'Location', 'Customer_Name', 
        'Sales_Volume', 'Marketing_Spend', 'Discount_Pct'
    ]
    st.info(f"Expected Columns: `{', '.join(required_columns)}`")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if not missing_cols:
            # Handle NA values
            df['Marketing_Spend'] = df['Marketing_Spend'].fillna(0)
            df['Discount_Pct'] = df['Discount_Pct'].fillna(0)
            
            # Format Date to the 1st of the month
            df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
            
            # Aggregate at the monthly level by all dimensions
            agg_dict = {
                'Sales_Volume': 'sum',
                'Marketing_Spend': 'sum',
                'Discount_Pct': 'mean'
            }
            group_cols = ['Date', 'Region', 'State', 'Brand', 'SKU', 'Location', 'Customer_Name']
            df_monthly = df.groupby(group_cols).agg(agg_dict).reset_index()
            
            df_monthly = df_monthly.sort_values('Date')
            st.session_state['historical_data'] = df_monthly
            
            st.success("Data successfully cleaned, aggregated to monthly level, and written to database!")
            st.dataframe(df_monthly.head(10))
        else:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")

# --- Page 2: System Forecast Generation (Best Fit Simulation) ---
elif page == "2. System Forecast Generation":
    st.title("System Forecast Generation (Best Fit)")
    
    if st.session_state['historical_data'] is None:
        st.warning("Please upload data in Step 1 first.")
    else:
        df_raw = st.session_state['historical_data']
        
        # --- Dynamic Sidebar Filters ---
        st.sidebar.markdown("### Forecast Filters")
        
        all_regions = df_raw['Region'].unique().tolist()
        sel_regions = st.sidebar.multiselect("Region(s)", all_regions, default=all_regions)
        
        all_states = df_raw['State'].unique().tolist()
        sel_states = st.sidebar.multiselect("State(s)", all_states, default=all_states)
        
        all_brands = df_raw['Brand'].unique().tolist()
        sel_brands = st.sidebar.multiselect("Brand(s)", all_brands, default=all_brands)
        
        all_customers = df_raw['Customer_Name'].unique().tolist()
        sel_customers = st.sidebar.multiselect("Customer(s)", all_customers, default=all_customers)
        
        # Apply filters to the raw data
        df_filtered = df_raw[
            (df_raw['Region'].isin(sel_regions)) &
            (df_raw['State'].isin(sel_states)) &
            (df_raw['Brand'].isin(sel_brands)) &
            (df_raw['Customer_Name'].isin(sel_customers))
        ]
        
        if df_filtered.empty:
            st.error("No data available for the selected filters. Please adjust your criteria.")
        else:
            df_total = df_filtered.groupby('Date')['Sales_Volume'].sum().reset_index()
            
            st.markdown(f"**Analyzing {len(df_filtered)} filtered records.**")
            
            horizon = st.number_input("Forecast Horizon (Months)", min_value=1, max_value=24, value=6)
            
            if st.button("Run Best Fit Analysis"):
                methods = ["SMA", "SES", "DES", "Croston", "MLR", "Auto-ARIMA"]
                results_list = []
                
                # Loop for demonstration of the Best Fit architecture
                for method in methods:
                    # 1. Calculate metrics: MAPE, Bias, Accuracy (Simulated)
                    mape = np.random.uniform(5, 20) 
                    bias = np.random.uniform(-5, 5)
                    accuracy = 100 - mape
                    
                    results_list.append({
                        "Method": method,
                        "Accuracy": round(accuracy, 2),
                        "MAPE": round(mape, 2),
                        "Bias": round(bias, 2)
                    })
                    
                    # 2. Simulate Fitted Values (Backdated forecast for the historical period)
                    # We use the generated MAPE to create realistic-looking deviations from the actuals
                    np.random.seed(len(method)) # Seed to keep charts consistent upon re-renders
                    variance = np.random.normal(0, mape / 100, len(df_total))
                    fitted_values = df_total['Sales_Volume'] * (1 + variance)
                    
                    # 3. Graphing in different windows
                    with st.expander(f"View Forecast vs Actuals: {method}"):
                        fig = go.Figure()
                        # Plot Actual Data
                        fig.add_trace(go.Scatter(
                            x=df_total['Date'], 
                            y=df_total['Sales_Volume'], 
                            mode='lines+markers', 
                            name='Actual Sales',
                            line=dict(color='blue')
                        ))
                        # Plot Simulated Fitted Data (Backdated Forecast)
                        fig.add_trace(go.Scatter(
                            x=df_total['Date'], 
                            y=fitted_values, 
                            mode='lines+markers', 
                            name=f'{method} Fit',
                            line=dict(color='orange', dash='dot')
                        ))
                        
                        fig.update_layout(
                            title=f"Historical Model Fit: {method} (MAPE: {mape:.2f}%)", 
                            xaxis_title="Date", 
                            yaxis_title="Volume",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Store results in session state
                results_df = pd.DataFrame(results_list)
                st.session_state['best_fit_results'] = results_df
                
            # Display Accuracy Table and Finalize Selection
            if st.session_state.get('best_fit_results') is not None:
                st.subheader("Model Comparison Matrix")
                st.dataframe(st.session_state['best_fit_results'], use_container_width=True)
                
                selected_method = st.selectbox("Select the Best Fit Method:", 
                                               st.session_state['best_fit_results']['Method'])
                
                if st.button("Finalize and Lock Baseline"):
                    # Generate a baseline for the selected method to pass to Step 3
                    last_date = df_total['Date'].max()
                    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
                    
                    # Dummy math to simulate the selected model's future output
                    baseline_val = df_total['Sales_Volume'].tail(3).mean() if not df_total.empty else 0
                    forecast_vals = [baseline_val] * horizon
                        
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Baseline_Forecast': forecast_vals})
                    
                    st.session_state['system_forecast'] = forecast_df
                    st.session_state['filtered_historical_total'] = df_total
                    
                    st.success(f"System baseline locked using {selected_method}.")
                
                # Download Button (Rendered outside the st.button to prevent Streamlit reset behavior)
                if st.session_state['system_forecast'] is not None:
                    csv = st.session_state['system_forecast'].to_csv(index=False)
                    st.download_button(
                        label="Download Statistical Forecast",
                        data=csv,
                        file_name="statistical_forecast_baseline.csv",
                        mime="text/csv"
                    )

# --- Page 3: Demand Review & Collaboration ---
elif page == "3. Demand Review & Collaboration":
    st.title("Demand Review (Adjust & Finalize)")
    
    if st.session_state['system_forecast'] is None:
        st.warning("Please generate and finalize a system forecast in Step 2 first.")
    else:
        st.markdown("Analyze the baseline and manually input overrides.")
        df = st.session_state['system_forecast'].copy()
        
        if 'Manual_Override' not in df.columns:
            df['Manual_Override'] = 0.0
            
        edited_df = st.data_editor(
            df,
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", disabled=True, format="MMM YYYY"),
                "Baseline_Forecast": st.column_config.NumberColumn("Baseline", disabled=True),
                "Manual_Override": st.column_config.NumberColumn("Override (+/- volume)", default=0.0)
            },
            hide_index=True
        )
        
        edited_df['Consensus_Forecast'] = edited_df['Baseline_Forecast'] + edited_df['Manual_Override']
        
        if st.button("Finalize Consensus"):
            st.session_state['consensus_forecast'] = edited_df
            st.success("Consensus forecast finalized and published!")
            
            # Use the filtered historical data for an accurate visual comparison
            hist_total = st.session_state['filtered_historical_total']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_total['Date'], y=hist_total['Sales_Volume'], mode='lines+markers', name='Historical (Filtered)'))
            fig.add_trace(go.Scatter(x=edited_df['Date'], y=edited_df['Baseline_Forecast'], mode='lines+markers', name='Baseline Forecast', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=edited_df['Date'], y=edited_df['Consensus_Forecast'], mode='lines+markers', name='Consensus Forecast', line=dict(color='green')))
            fig.update_layout(title="Historical vs Consensus Forecast", xaxis_title="Date", yaxis_title="Volume")
            st.plotly_chart(fig, use_container_width=True)

# --- Page 4: Post-Game Analysis ---
elif page == "4. Post-Game Analysis":
    st.title("Post-Game Analysis")
    st.markdown("Analyze the accuracy of predictions against actual sales data.")
    
    if st.session_state['consensus_forecast'] is None:
        st.warning("Please complete the consensus process to measure KPIs.")
    else:
        results = st.session_state['consensus_forecast'].copy()
        
        # Simulating actuals for demonstration
        np.random.seed(42)
        variance = np.random.uniform(0.8, 1.2, len(results))
        results['Actuals'] = (results['Consensus_Forecast'] * variance).round(2)
        
        # Calculate KPIs
        results['Abs_Error'] = abs(results['Actuals'] - results['Consensus_Forecast'])
        results['Bias_%'] = ((results['Consensus_Forecast'] / results['Actuals']) - 1) * 100
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
