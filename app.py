import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import pmdarima as pm

# Suppress statsmodels warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# --- Helper Function for True Error Metrics ---
def calculate_error_metrics(actual, forecast):
    # Create masks to drop NaNs and align arrays
    mask = ~np.isnan(actual) & ~np.isnan(forecast)
    a, f = actual[mask], forecast[mask]
    
    if len(a) == 0: 
        return 0, 0, 0
        
    # Prevent division by zero in MAPE
    a_safe = np.where(a == 0, 1e-5, a)
    
    mape = np.mean(np.abs((a - f) / a_safe)) * 100
    bias = (np.sum(f - a) / np.sum(a)) * 100 if np.sum(a) != 0 else 0
    accuracy = max(0, 100 - mape)
    
    return round(accuracy, 2), round(mape, 2), round(bias, 2)

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
if 'model_forecast_dict' not in st.session_state:
    st.session_state['model_forecast_dict'] = {} 
if 'trained_mlr_model' not in st.session_state:
    st.session_state['trained_mlr_model'] = None
if 'default_mlr_future_X' not in st.session_state:
    st.session_state['default_mlr_future_X'] = None

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
    st.markdown("Hi Nupur Mam!! Please upload your transaction data. Ensure it contains the necessary hierarchies and causal drivers.")
    
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
            df['Marketing_Spend'] = df['Marketing_Spend'].fillna(0)
            df['Discount_Pct'] = df['Discount_Pct'].fillna(0)
            
            # Format Date to the 1st of the month
            df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()
            
            # Aggregate at the monthly level
            agg_dict = {'Sales_Volume': 'sum', 'Marketing_Spend': 'sum', 'Discount_Pct': 'mean'}
            group_cols = ['Date', 'Region', 'State', 'Brand', 'SKU', 'Location', 'Customer_Name']
            df_monthly = df.groupby(group_cols).agg(agg_dict).reset_index().sort_values('Date')
            
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
            # Updated aggregation: We must pull through causal drivers for MLR
            df_total = df_filtered.groupby('Date').agg({
                'Sales_Volume': 'sum',
                'Marketing_Spend': 'sum',
                'Discount_Pct': 'mean'
            }).reset_index()
            
            st.markdown(f"**Analyzing {len(df_filtered)} filtered records.**")
            
            horizon = st.number_input("Forecast Horizon (Months)", min_value=1, max_value=24, value=6)
            
            if st.button("Run Best Fit Analysis"):
                # Updated Methods list 
                methods = ["SMA (3-Month)", "Holt-Winters (Exponential Smoothing)", "MLR", "Auto-ARIMA"]
                results_list = []
                forecast_dict = {}
                actuals = df_total['Sales_Volume'].values
                dates = df_total['Date'].values
                
                for method in methods:
                    try:
                        if method == "SMA (3-Month)":
                            # True 3-Month Simple Moving Average
                            window = 3
                            fitted_values = df_total['Sales_Volume'].rolling(window=window).mean().shift(1)
                            # Fill early NaNs with the first available rolling mean for graphing continuity
                            fitted_values = fitted_values.bfill().values 
                            
                            # Future Forecast
                            baseline_val = actuals[-window:].mean() if len(actuals) >= window else actuals.mean()
                            future_forecast = [baseline_val] * horizon
                            
                        elif method == "Holt-Winters (Exponential Smoothing)":
                            # Updated to Holt-Winters using Exponential Smoothing from statsmodels
                            # trend='add' (linear trend), seasonal param can also be applied based on needs
                            model = ExponentialSmoothing(actuals, trend='add', initialization_method="estimated").fit()
                            fitted_values = model.fittedvalues
                            
                            # Future Forecast
                            future_forecast = model.forecast(horizon).tolist()

                        elif method == "MLR":
                            # Define independent (X) and dependent (y) variables
                            X = df_total[['Marketing_Spend', 'Discount_Pct']]
                            y = df_total['Sales_Volume']
                            
                            # Fit the real MLR model
                            mlr_model = LinearRegression()
                            mlr_model.fit(X, y)
                            st.session_state['trained_mlr_model'] = mlr_model  # Save for custom input later
                            
                            # Extract real fitted historical values
                            fitted_values = mlr_model.predict(X)
                            
                            # Generate future baseline forecast
                            # Default uses recent average, this gets saved to session state for the grid
                            recent_marketing = df_total['Marketing_Spend'].tail(3).mean()
                            recent_discount = df_total['Discount_Pct'].tail(3).mean()
                            
                            future_X = pd.DataFrame({
                                'Marketing_Spend': [recent_marketing] * horizon,
                                'Discount_Pct': [recent_discount] * horizon
                            })
                            st.session_state['default_mlr_future_X'] = future_X
                            future_forecast = mlr_model.predict(future_X).tolist()

                        elif method == "Auto-ARIMA":
                            # Fit the real Auto-ARIMA model with Yearly seasonality applied
                            arima_model = pm.auto_arima(actuals, 
                                                        seasonal=True, 
                                                        m=6,  # Yearly cyclicity 
                                                        suppress_warnings=True, 
                                                        error_action="ignore")
                            
                            # Extract real fitted historical values
                            fitted_values = arima_model.predict_in_sample()
                            
                            # Generate true future baseline forecast for the selected horizon
                            future_forecast = arima_model.predict(n_periods=horizon).tolist()

                        # 1. Calculate TRUE metrics
                        accuracy, mape, bias = calculate_error_metrics(actuals, fitted_values)
                        
                        results_list.append({
                            "Method": method,
                            "Accuracy": accuracy,
                            "MAPE": mape,
                            "Bias": bias
                        })
                        
                        # Store future forecast in dictionary for Step 3 transfer
                        forecast_dict[method] = future_forecast
                        
                        # 2. Graphing the True Fit
                        with st.expander(f"View Forecast vs Actuals: {method}"):
                            fig = go.Figure()
                            # Actuals
                            fig.add_trace(go.Scatter(
                                x=df_total['Date'], y=actuals, mode='lines+markers', 
                                name='Actual Sales', line=dict(color='blue')
                            ))
                            # Fitted (Backdated Forecast)
                            fig.add_trace(go.Scatter(
                                x=df_total['Date'], y=fitted_values, mode='lines+markers', 
                                name=f'{method} Fit', line=dict(color='orange', dash='dot')
                            ))
                            fig.update_layout(
                                title=f"Historical Model Fit: {method} (MAPE: {mape:.2f}%)", 
                                xaxis_title="Date", yaxis_title="Volume", hovermode="x unified"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error calculating {method}: {str(e)}")
                
                # Store results in session state
                st.session_state['best_fit_results'] = pd.DataFrame(results_list)
                st.session_state['model_forecast_dict'] = forecast_dict
                
            # Display Accuracy Table and Finalize Selection
            if st.session_state.get('best_fit_results') is not None:
                st.subheader("Model Comparison Matrix")
                st.dataframe(st.session_state['best_fit_results'], use_container_width=True)
                
                selected_method = st.selectbox("Select the Best Fit Method:", st.session_state['best_fit_results']['Method'])
                
                # Dynamic Logic for MLR Selection Grid Override
                edited_mlr_inputs = None
                if selected_method == "MLR":
                    st.markdown("#### Adjust MLR Causal Drivers")
                    st.info("Because MLR relies on Marketing Spend and Discounts, please input your future plans below to dynamically generate the baseline.")
                    
                    last_date = df_total['Date'].max()
                    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
                    
                    # Construct default dataframe for editor
                    default_mlr_grid = st.session_state['default_mlr_future_X'].copy()
                    default_mlr_grid.insert(0, 'Date', [d.strftime('%b %Y') for d in future_dates])
                    
                    edited_mlr_inputs = st.data_editor(
                        default_mlr_grid,
                        column_config={
                            "Date": st.column_config.Column("Future Horizon Month", disabled=True),
                            "Marketing_Spend": st.column_config.NumberColumn("Planned Marketing Spend"),
                            "Discount_Pct": st.column_config.NumberColumn("Planned Discount %")
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                if st.button("Finalize and Lock Baseline"):
                    if selected_method == "MLR" and edited_mlr_inputs is not None:
                        # Recalculate MLR forecast using the user's custom inputs
                        mlr_model = st.session_state['trained_mlr_model']
                        custom_X = edited_mlr_inputs[['Marketing_Spend', 'Discount_Pct']]
                        winning_forecast = mlr_model.predict(custom_X).tolist()
                    else:
                        # Pull standard calculated future forecast for other winning models
                        winning_forecast = st.session_state['model_forecast_dict'][selected_method]
                    
                    last_date = df_total['Date'].max()
                    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
                        
                    forecast_df = pd.DataFrame({'Date': future_dates, 'Baseline_Forecast': winning_forecast})
                    
                    st.session_state['system_forecast'] = forecast_df
                    st.session_state['filtered_historical_total'] = df_total
                    
                    st.success(f"System baseline locked using true statistical output from {selected_method}.")
                
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
