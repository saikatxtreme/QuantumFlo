import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm, median_abs_deviation
import math # For math.ceil
import graphviz # Import graphviz

import warnings
warnings.filterwarnings('ignore') # Suppress warnings

# For Plotting
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration for Dummy Data Generation (used for templates and sample run) ---
DEFAULT_NUM_SKUS = 5
DEFAULT_NUM_COMPONENTS_PER_SKU = 3
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 31)
DEFAULT_PROMOTION_FREQUENCY_DAYS = 60
DEFAULT_MAX_LEAD_TIME_DAYS = 30
DEFAULT_MAX_SKU_SHELF_LIFE_DAYS = 365
DEFAULT_SALES_CHANNELS = ["Distributor Network", "Amazon", "Own Website"]

# --- Default Cost Parameters (used if global_config.csv is not uploaded) ---
DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY = 0.10
DEFAULT_ORDERING_COST_PER_ORDER = 50.00

# --- Dummy Data Generation ---

def generate_dummy_sales_data(num_skus, start_date, end_date):
    """
    Generates dummy sales data for multiple SKUs, including promotions.
    """
    sales_data = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for i in range(1, num_skus + 1):
        sku_id = f'SKU_{i:03d}'
        base_demand = random.randint(10, 50)
        
        # Add promotions
        promotion_dates = random.sample(date_range.tolist(), len(date_range) // DEFAULT_PROMOTION_FREQUENCY_DAYS)
        
        for single_date in date_range:
            demand = int(np.random.normal(base_demand, base_demand * 0.2))
            demand = max(0, demand) # Demand cannot be negative
            
            is_promo = single_date in promotion_dates
            if is_promo:
                demand = int(demand * random.uniform(1.2, 1.8))
            
            sales_data.append({
                'Date': single_date,
                'SKU_ID': sku_id,
                'Sales_Quantity': demand,
                'Promotion': 1 if is_promo else 0
            })
            
    sales_df = pd.DataFrame(sales_data)
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    
    return sales_df

def generate_dummy_config_data(num_skus):
    """
    Generates dummy configuration data for SKUs.
    """
    config_data = []
    for i in range(1, num_skus + 1):
        sku_id = f'SKU_{i:03d}'
        config_data.append({
            'SKU_ID': sku_id,
            'Price': round(random.uniform(10, 200), 2),
            'Cost': round(random.uniform(5, 150), 2),
            'Lead_Time_Days': random.randint(5, DEFAULT_MAX_LEAD_TIME_DAYS),
            'Shelf_Life_Days': random.randint(90, DEFAULT_MAX_SKU_SHELF_LIFE_DAYS),
            'Safety_Stock_Factor': random.uniform(0.5, 2.0),
            'Optimal_Reorder_Point': random.randint(50, 200),
            'Min_Order_Quantity': random.randint(10, 50),
            'EOQ': random.randint(50, 150)
        })
    return pd.DataFrame(config_data)

def generate_dummy_inventory_data(sales_df, config_df, start_date):
    """
    Generates dummy inventory data based on sales and config.
    """
    inventory_data = []
    unique_skus = sales_df['SKU_ID'].unique()
    
    for sku_id in unique_skus:
        sku_sales = sales_df[sales_df['SKU_ID'] == sku_id].copy()
        
        initial_inventory = random.randint(100, 500)
        current_inventory = initial_inventory
        
        sku_config = config_df[config_df['SKU_ID'] == sku_id]
        if not sku_config.empty:
            optimal_reorder_point = sku_config['Optimal_Reorder_Point'].iloc[0]
            lead_time_days = sku_config['Lead_Time_Days'].iloc[0]
            eoq = sku_config['EOQ'].iloc[0]
        else:
            optimal_reorder_point = 100
            lead_time_days = 10
            eoq = 50

        last_order_date = None
        
        for index, row in sku_sales.sort_values(by='Date').iterrows():
            date = row['Date']
            sales_quantity = row['Sales_Quantity']
            
            current_inventory -= sales_quantity
            
            if current_inventory <= optimal_reorder_point and (last_order_date is None or (date - last_order_date).days > lead_time_days):
                order_quantity = eoq
                order_arrival_date = date + timedelta(days=lead_time_days)
                
                if order_arrival_date <= sales_df['Date'].max():
                    inventory_data.append({
                        'Date': order_arrival_date,
                        'SKU_ID': sku_id,
                        'Replenishment_Quantity': order_quantity,
                        'Inventory_Level': 0
                    })
                last_order_date = date

            inventory_data.append({
                'Date': date,
                'SKU_ID': sku_id,
                'Replenishment_Quantity': 0,
                'Inventory_Level': current_inventory
            })

    inventory_df = pd.DataFrame(inventory_data)
    inventory_df.sort_values(by=['SKU_ID', 'Date'], inplace=True)
    inventory_df.drop_duplicates(subset=['SKU_ID', 'Date'], keep='last', inplace=True)
    
    final_inventory_data = []
    for sku_id in unique_skus:
        sku_df = inventory_df[inventory_df['SKU_ID'] == sku_id].copy()
        sku_sales = sales_df[sales_df['SKU_ID'] == sku_id].copy()
        
        start_inv = random.randint(100, 500)
        current_inv = start_inv
        
        merged_df = pd.merge(sku_sales, sku_df[['Date', 'Replenishment_Quantity']], on='Date', how='left').fillna(0)
        merged_df.sort_values(by='Date', inplace=True)
        
        for _, row in merged_df.iterrows():
            current_inv = current_inv - row['Sales_Quantity'] + row['Replenishment_Quantity']
            final_inventory_data.append({
                'Date': row['Date'],
                'SKU_ID': sku_id,
                'Inventory_Level': max(0, current_inv),
                'Replenishment_Quantity': row['Replenishment_Quantity']
            })
            
    final_inventory_df = pd.DataFrame(final_inventory_data)
    final_inventory_df.sort_values(by=['SKU_ID', 'Date'], inplace=True)
    
    final_inventory_df['Stockout'] = final_inventory_df['Inventory_Level'].apply(lambda x: 1 if x == 0 else 0)
    
    return final_inventory_df

def generate_dummy_demand_signals_data(num_skus, start_date, end_date):
    """
    Generates dummy demand signals data for multiple SKUs.
    """
    demand_signals_data = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for i in range(1, num_skus + 1):
        sku_id = f'SKU_{i:03d}'
        
        for single_date in date_range:
            demand_signals_data.append({
                'Date': single_date,
                'SKU_ID': sku_id,
                'Web_Traffic': random.randint(100, 1000),
                'Social_Media_Mentions': random.randint(10, 100),
                'Competitor_Price_Index': round(random.uniform(0.8, 1.2), 2)
            })
            
    demand_signals_df = pd.DataFrame(demand_signals_data)
    demand_signals_df['Date'] = pd.to_datetime(demand_signals_df['Date'])
    
    return demand_signals_df

# --- Core Logic Functions ---

def preprocess_data(sales_df, config_df, demand_signals_df, global_config_df):
    """
    Merges and prepares all dataframes for forecasting and analysis.
    """
    # Merge sales with config and demand signals
    df = pd.merge(sales_df, config_df, on='SKU_ID', how='left')
    df = pd.merge(df, demand_signals_df, on=['Date', 'SKU_ID'], how='left')
    
    # Merge with global config for costs
    if not global_config_df.empty:
        global_config_df = global_config_df.set_index('Parameter')
        df['Holding_Cost_Per_Unit_Per_Day'] = float(global_config_df.loc['Holding_Cost_Per_Unit_Per_Day', 'Value'])
        df['Ordering_Cost_Per_Order'] = float(global_config_df.loc['Ordering_Cost_Per_Order', 'Value'])
    else:
        df['Holding_Cost_Per_Unit_Per_Day'] = DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY
        df['Ordering_Cost_Per_Order'] = DEFAULT_ORDERING_COST_PER_ORDER
    
    # Create lag features for machine learning models
    df.sort_values(['SKU_ID', 'Date'], inplace=True)
    df['Day_of_Week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)
    
    for lag in range(1, 4):
        df[f'Sales_Lag_{lag}'] = df.groupby('SKU_ID')['Sales_Quantity'].shift(lag)
    
    df.fillna(0, inplace=True)
    
    return df

def aggregate_kpi_for_plot(df, selected_sku, kpi_column, roll_up_choice, agg_func):
    """
    Aggregates KPI data for a specific SKU based on the chosen roll-up period.
    """
    if selected_sku and kpi_column in df.columns:
        filtered_df = df[df['SKU_ID'] == selected_sku].copy()
        if not filtered_df.empty:
            
            freq_map = {
                'Daily': 'D',
                'Weekly': 'W',
                'Monthly': 'M'
            }
            freq = freq_map.get(roll_up_choice, 'D')
            
            if agg_func == 'mean':
                agg_df = filtered_df.groupby(pd.Grouper(key='Date', freq=freq))[kpi_column].mean().reset_index()
            elif agg_func == 'sum':
                agg_df = filtered_df.groupby(pd.Grouper(key='Date', freq=freq))[kpi_column].sum().reset_index()
            else:
                agg_df = pd.DataFrame(columns=['Date', kpi_column])
            
            agg_df.rename(columns={kpi_column: 'Value'}, inplace=True)
            return agg_df
    return pd.DataFrame(columns=['Date', 'Value'])

# --- Forecasting Models ---

def forecast_with_xgboost(train_df, test_df, horizon):
    """
    Forecasts demand using XGBoost.
    """
    features = ['Promotion', 'Web_Traffic', 'Social_Media_Mentions', 'Competitor_Price_Index', 'Day_of_Week', 'Month', 'Year', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3']
    target = 'Sales_Quantity'
    
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(train_df[features], train_df[target])
    
    predictions = model.predict(test_df[features])
    return predictions

def forecast_with_random_forest(train_df, test_df, horizon):
    """
    Forecasts demand using RandomForestRegressor.
    """
    features = ['Promotion', 'Web_Traffic', 'Social_Media_Mentions', 'Competitor_Price_Index', 'Day_of_Week', 'Month', 'Year', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3']
    target = 'Sales_Quantity'
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train_df[features], train_df[target])
    
    predictions = model.predict(test_df[features])
    return predictions

def forecast_with_moving_average(series, window_size, horizon):
    """
    Forecasts future demand using a simple moving average.
    """
    predictions = []
    last_known_values = series.tail(window_size).tolist()
    
    for _ in range(horizon):
        if len(last_known_values) > 0:
            next_prediction = np.mean(last_known_values)
            predictions.append(next_prediction)
            
            last_known_values.pop(0)
            last_known_values.append(next_prediction)
        else:
            predictions.append(0)
            
    return np.array(predictions)
    
def forecast_with_moving_median(series, window_size, horizon):
    """
    Forecasts future demand using a simple moving median.
    """
    predictions = []
    last_known_values = series.tail(window_size).tolist()
    
    for _ in range(horizon):
        if len(last_known_values) > 0:
            next_prediction = np.median(last_known_values)
            predictions.append(next_prediction)
            
            last_known_values.pop(0)
            last_known_values.append(next_prediction)
        else:
            predictions.append(0)
            
    return np.array(predictions)

def evaluate_models(train_df, validation_df):
    """
    Evaluates the specified forecasting models and returns their MAE scores.
    """
    # Ensure validation data is not empty
    if validation_df.empty:
        return {'XGBoost': np.inf, 'Random Forest': np.inf, 'Moving Average': np.inf, 'Moving Median': np.inf}

    mae_scores = {}
    
    # XGBoost
    try:
        xgboost_preds = forecast_with_xgboost(train_df, validation_df, len(validation_df))
        mae_scores['XGBoost'] = mean_absolute_error(validation_df['Sales_Quantity'], xgboost_preds)
    except Exception as e:
        mae_scores['XGBoost'] = np.inf
        
    # Random Forest
    try:
        rf_preds = forecast_with_random_forest(train_df, validation_df, len(validation_df))
        mae_scores['Random Forest'] = mean_absolute_error(validation_df['Sales_Quantity'], rf_preds)
    except Exception as e:
        mae_scores['Random Forest'] = np.inf

    # Moving Average
    try:
        ma_preds = forecast_with_moving_average(train_df['Sales_Quantity'], window_size=7, horizon=len(validation_df))
        mae_scores['Moving Average'] = mean_absolute_error(validation_df['Sales_Quantity'], ma_preds)
    except Exception as e:
        mae_scores['Moving Average'] = np.inf

    # Moving Median
    try:
        mm_preds = forecast_with_moving_median(train_df['Sales_Quantity'], window_size=7, horizon=len(validation_df))
        mae_scores['Moving Median'] = mean_absolute_error(validation_df['Sales_Quantity'], mm_preds)
    except Exception as e:
        mae_scores['Moving Median'] = np.inf
        
    return mae_scores

def auto_select_best_model(sku_data):
    """
    Automatically selects the best model for a given SKU based on MAE.
    """
    # Split data for training and validation
    train_size = int(len(sku_data) * 0.8)
    train_df = sku_data.iloc[:train_size]
    validation_df = sku_data.iloc[train_size:]
    
    # Evaluate all models
    scores = evaluate_models(train_df, validation_df)
    
    # Find the best model with the minimum MAE
    best_model = min(scores, key=scores.get)
    best_mae = scores[best_model]
    
    return best_model, best_mae

# --- KPI Calculation Functions ---

def calculate_stockout_rate(inventory_df):
    """
    Calculates the daily stockout rate for each SKU.
    """
    df = inventory_df.copy()
    stockout_df = df.groupby(['SKU_ID', 'Date'])['Stockout'].first().reset_index()
    stockout_df['Stockout_Rate'] = stockout_df.groupby('SKU_ID')['Stockout'].transform(lambda x: x.rolling(window=30, min_periods=1).mean() * 100)
    return stockout_df.drop(columns=['Stockout'])


# --- Streamlit App ---

st.set_page_config(page_title="Demand & Inventory Intelligence", layout="wide")

st.title("Demand & Inventory Intelligence Platform")
st.markdown("This platform allows you to upload your sales, inventory, and configuration data to generate demand forecasts and optimize inventory KPIs.")

# --- Session State Initialization ---
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = pd.DataFrame()
if 'config_df' not in st.session_state:
    st.session_state.config_df = pd.DataFrame()
if 'demand_signals_df' not in st.session_state:
    st.session_state.demand_signals_df = pd.DataFrame()
if 'global_config_df' not in st.session_state:
    st.session_state.global_config_df = pd.DataFrame()
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = pd.DataFrame()
if 'all_forecasts_df' not in st.session_state:
    st.session_state.all_forecasts_df = pd.DataFrame()
if 'all_stockout_rates_df' not in st.session_state:
    st.session_state.all_stockout_rates_df = pd.DataFrame()
if 'model_selection_summary' not in st.session_state:
    st.session_state.model_selection_summary = pd.DataFrame()

# --- Sidebar for Data Upload and Configuration ---
with st.sidebar:
    st.header("1. Data Upload")
    
    if st.button("Run with Sample Data"):
        st.session_state.sales_df = generate_dummy_sales_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE)
        st.session_state.config_df = generate_dummy_config_data(DEFAULT_NUM_SKUS)
        st.session_state.demand_signals_df = generate_dummy_demand_signals_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE)
        st.success("Sample data generated successfully!")
    
    st.markdown("---")
    st.subheader("Or upload your own CSV files:")
    sales_file = st.file_uploader("Upload Sales Data (sales.csv)", type="csv")
    if sales_file:
        st.session_state.sales_df = pd.read_csv(sales_file, parse_dates=['Date'])
    
    config_file = st.file_uploader("Upload SKU Config (config.csv)", type="csv")
    if config_file:
        st.session_state.config_df = pd.read_csv(config_file)
        
    demand_signals_file = st.file_uploader("Upload Demand Signals (demand_signals.csv)", type="csv")
    if demand_signals_file:
        st.session_state.demand_signals_df = pd.read_csv(demand_signals_file, parse_dates=['Date'])
        
    global_config_file = st.file_uploader("Upload Global Config (global_config.csv)", type="csv")
    if global_config_file:
        st.session_state.global_config_df = pd.read_csv(global_config_file)
        
    st.markdown("---")
    st.header("2. Forecasting Parameters")
    
    model_choice = st.selectbox(
        "Select Forecasting Model",
        ("Auto-Select Best Model", "XGBoost", "Random Forest", "Moving Average", "Moving Median")
    )
    
    forecast_horizon_days = st.number_input("Forecast Horizon (in days)", min_value=1, max_value=365, value=30)
    
    if st.button("Run Forecasting"):
        if not st.session_state.sales_df.empty and not st.session_state.config_df.empty and not st.session_state.demand_signals_df.empty:
            
            with st.spinner("Preprocessing data..."):
                preprocessed_df = preprocess_data(
                    st.session_state.sales_df,
                    st.session_state.config_df,
                    st.session_state.demand_signals_df,
                    st.session_state.global_config_df
                )
                st.session_state.preprocessed_df = preprocessed_df
            
            all_forecasts = []
            model_summary = []
            unique_skus = preprocessed_df['SKU_ID'].unique()
            
            st.info("Running forecasting...")
            progress_bar = st.progress(0)
            
            for i, sku_id in enumerate(unique_skus):
                sku_data = preprocessed_df[preprocessed_df['SKU_ID'] == sku_id].copy()
                
                selected_model = model_choice
                if model_choice == "Auto-Select Best Model":
                    selected_model, best_mae = auto_select_best_model(sku_data)
                    model_summary.append({'SKU_ID': sku_id, 'Best_Model': selected_model, 'MAE': best_mae})
                
                # Split data for final forecast
                last_train_date = sku_data['Date'].max() - timedelta(days=forecast_horizon_days)
                train_df = sku_data[sku_data['Date'] <= last_train_date]
                
                last_date = train_df['Date'].max()
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon_days, freq='D')
                
                future_df = pd.DataFrame({'Date': future_dates, 'SKU_ID': sku_id})
                
                # Create future features
                future_df['Promotion'] = 0
                future_df['Web_Traffic'] = train_df['Web_Traffic'].mean()
                future_df['Social_Media_Mentions'] = train_df['Social_Media_Mentions'].mean()
                future_df['Competitor_Price_Index'] = train_df['Competitor_Price_Index'].mean()
                future_df['Day_of_Week'] = future_df['Date'].dt.dayofweek
                future_df['Month'] = future_df['Date'].dt.month
                future_df['Year'] = future_df['Date'].dt.year
                future_df['Week_of_Year'] = future_df['Date'].dt.isocalendar().week.astype(int)
                
                future_df['Sales_Lag_1'] = train_df['Sales_Quantity'].iloc[-1] if len(train_df) >= 1 else 0
                future_df['Sales_Lag_2'] = train_df['Sales_Quantity'].iloc[-2] if len(train_df) >= 2 else 0
                future_df['Sales_Lag_3'] = train_df['Sales_Quantity'].iloc[-3] if len(train_df) >= 3 else 0
                
                # Determine which model to run for the final forecast
                forecast_predictions = []
                if selected_model == "XGBoost":
                    forecast_predictions = forecast_with_xgboost(train_df, future_df, forecast_horizon_days)
                elif selected_model == "Random Forest":
                    forecast_predictions = forecast_with_random_forest(train_df, future_df, forecast_horizon_days)
                elif selected_model == "Moving Average":
                    forecast_predictions = forecast_with_moving_average(train_df['Sales_Quantity'], window_size=7, horizon=forecast_horizon_days)
                elif selected_model == "Moving Median":
                    forecast_predictions = forecast_with_moving_median(train_df['Sales_Quantity'], window_size=7, horizon=forecast_horizon_days)
                
                forecast_df = future_df.copy()
                forecast_df['Forecasted_Sales'] = np.maximum(0, forecast_predictions).round(0).astype(int)
                forecast_df['Model_Used'] = selected_model
                forecast_df = forecast_df[['Date', 'SKU_ID', 'Forecasted_Sales', 'Model_Used']]
                all_forecasts.append(forecast_df)
                
                progress_bar.progress((i + 1) / len(unique_skus))
                
            st.session_state.all_forecasts_df = pd.concat(all_forecasts)
            st.session_state.model_selection_summary = pd.DataFrame(model_summary)
            st.success("Forecasting complete!")
            
            st.info("Calculating KPIs (Stockout Rate)...")
            dummy_inventory_df = generate_dummy_inventory_data(st.session_state.sales_df, st.session_state.config_df, DEFAULT_START_DATE)
            st.session_state.all_stockout_rates_df = calculate_stockout_rate(dummy_inventory_df)
            st.success("KPI calculations complete!")

# --- Main Content Area ---
if not st.session_state.sales_df.empty:
    st.header("Dashboard")
    
    unique_skus = st.session_state.sales_df['SKU_ID'].unique()
    
    if not st.session_state.all_forecasts_df.empty:
        if st.session_state.model_selection_summary is not None and not st.session_state.model_selection_summary.empty:
            st.subheader("Auto-Model Selection Summary")
            st.dataframe(st.session_state.model_selection_summary.set_index('SKU_ID'), use_container_width=True)
            
        selected_sku_for_plot = st.selectbox(
            "Select SKU to visualize",
            unique_skus
        )
        
        forecast_roll_up_choice = st.selectbox(
            "Select Time Period for Plotting",
            ("Daily", "Weekly", "Monthly"),
            index=2
        )
        
        st.subheader(f"Demand Forecast for {selected_sku_for_plot}")
        
        historical_sales = st.session_state.sales_df[st.session_state.sales_df['SKU_ID'] == selected_sku_for_plot]
        forecasted_sales = st.session_state.all_forecasts_df[st.session_state.all_forecasts_df['SKU_ID'] == selected_sku_for_plot]
        model_used = forecasted_sales['Model_Used'].iloc[0] if not forecasted_sales.empty else "N/A"

        historical_plot_df = aggregate_kpi_for_plot(historical_sales, selected_sku_for_plot, 'Sales_Quantity', forecast_roll_up_choice, 'sum')
        forecast_plot_df = aggregate_kpi_for_plot(forecasted_sales, selected_sku_for_plot, 'Forecasted_Sales', forecast_roll_up_choice, 'sum')

        if not historical_plot_df.empty:
            
            fig_forecast = go.Figure()

            fig_forecast.add_trace(go.Scatter(
                x=historical_plot_df['Date'],
                y=historical_plot_df['Value'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='royalblue')
            ))

            if not forecast_plot_df.empty:
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_plot_df['Date'],
                    y=forecast_plot_df['Value'],
                    mode='lines',
                    name=f'{model_used} Forecast',
                    line=dict(color='red', dash='dot')
                ))

            fig_forecast.update_layout(
                title=f"Sales Forecast for {selected_sku_for_plot} (Roll-up: {forecast_roll_up_choice})",
                xaxis_title="Date",
                yaxis_title="Sales Quantity",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.info("No historical sales data to plot.")

    else:
        st.info("Please run the forecasting models to see the forecast plots.")

    st.subheader("Key Performance Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total SKUs", len(unique_skus))
    
    if not st.session_state.all_stockout_rates_df.empty:
        unique_skus_for_stockout_plot = st.session_state.all_stockout_rates_df['SKU_ID'].unique()
        if len(unique_skus_for_stockout_plot) > 0:
            stockout_plot_df = aggregate_kpi_for_plot(
                st.session_state.all_stockout_rates_df, selected_sku_for_plot, 'Stockout_Rate', forecast_roll_up_choice, 'mean'
            ).tail(12)

            if not stockout_plot_df.empty:
                fig_stockout = px.line(
                    stockout_plot_df,
                    x="Date",
                    y="Value",
                    title=f"Stockout Rate for {selected_sku_for_plot} (Roll-up: {forecast_roll_up_choice})",
                    labels={"Value": "Stockout Rate (%)", "Date": forecast_roll_up_choice},
                    color_discrete_sequence=['orange']
                )
                fig_stockout.update_layout(hovermode="x unified")
                with col2:
                    st.plotly_chart(fig_stockout, use_container_width=True)
            else:
                with col2:
                    st.info(f"No Stockout Rate data to plot for {selected_sku_for_plot}.")
        else:
            with col2:
                st.info("No SKUs available for Stockout Rate plotting.")
    else:
        with col2:
            st.info("No Stockout Rate data available. Please run the forecasting and KPI calculation.")

else:
    st.info("Please upload your data files or run with sample data to get started.")

