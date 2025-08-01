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
import warnings
warnings.filterwarnings('ignore') # Suppress warnings

# For Plotting
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration for Realistic Dummy Data Generation ---
DEFAULT_NUM_SKUS = 8
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 31)
DEFAULT_PROMOTION_FREQUENCY_DAYS = 90
DEFAULT_MAX_LEAD_TIME_DAYS = 60 # More realistic for global supply chain
DEFAULT_SALES_CHANNELS = ["E-commerce", "Major Retailer", "Carrier Store", "Wholesale"]
PRODUCT_TYPES = ["Smartphone", "Smartwatch", "Wireless_Earbuds", "Fitness_Tracker"]
SKU_NAMES = {
    "Smartphone": ["Galaxy_Fold_X", "Pixel_Fusion_Pro", "Quantum_Phone_S1"],
    "Smartwatch": ["Chronos_GT_Pro", "Pulse_Watch_4", "Apex_Sync_SE"],
    "Wireless_Earbuds": ["Aura_Buds_Gen3", "Sonos_Pods_Plus"],
    "Fitness_Tracker": ["Vita_Band_2000", "Zenith_Flex_Pro"]
}

# --- Default Cost Parameters (used if global_config.csv is not uploaded) ---
DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY = 0.50 # Higher cost for tech products
DEFAULT_ORDERING_COST_PER_ORDER = 150.00 # Higher fixed cost for logistics and admin

# --- Realistic Dummy Data Generation for Mobile & Wearables ---

def generate_realistic_sales_data(num_skus, start_date, end_date):
    """
    Generates realistic dummy sales data for mobile & wearables, with seasonality and promotions.
    """
    sales_data = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    sku_list = []
    while len(sku_list) < num_skus:
        product_type = random.choice(PRODUCT_TYPES)
        sku_name = random.choice(SKU_NAMES[product_type])
        sku_id = f"{product_type}_{sku_name}"
        if sku_id not in sku_list:
            sku_list.append(sku_id)

    for sku_id in sku_list:
        base_demand = random.randint(30, 120)
        
        # Add promotions
        promotion_dates = random.sample(date_range.tolist(), len(date_range) // DEFAULT_PROMOTION_FREQUENCY_DAYS)
        
        for single_date in date_range:
            day_of_year = single_date.timetuple().tm_yday
            
            # Seasonality: higher sales in Q4 (holidays)
            seasonal_factor = 1.0
            if single_date.month in [11, 12]:
                seasonal_factor = random.uniform(1.5, 2.5)
            elif single_date.month in [1, 2]:
                seasonal_factor = random.uniform(0.7, 0.9)
            
            # Weekly pattern: higher sales on weekends
            weekly_factor = 1.0
            if single_date.weekday() >= 5: # Saturday or Sunday
                weekly_factor = random.uniform(1.1, 1.3)

            demand = int(np.random.normal(base_demand, base_demand * 0.3))
            demand = max(0, demand) # Demand cannot be negative
            demand = int(demand * seasonal_factor * weekly_factor)
            
            is_promo = single_date in promotion_dates
            if is_promo:
                demand = int(demand * random.uniform(1.3, 2.0))
            
            sales_data.append({
                'Date': single_date,
                'SKU_ID': sku_id,
                'Sales_Quantity': demand,
                'Promotion': 1 if is_promo else 0
            })
            
    sales_df = pd.DataFrame(sales_data)
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    
    return sales_df

def generate_realistic_config_data(sales_df):
    """
    Generates realistic configuration data for SKUs based on mobile/wearable context.
    """
    config_data = []
    unique_skus = sales_df['SKU_ID'].unique()
    
    for sku_id in unique_skus:
        # Assign different cost/price/lead time based on product type
        product_type = sku_id.split('_')[0]
        if product_type == "Smartphone":
            price = random.uniform(600, 1500)
            cost = price * random.uniform(0.6, 0.8)
            lead_time = random.randint(30, DEFAULT_MAX_LEAD_TIME_DAYS)
        elif product_type == "Smartwatch":
            price = random.uniform(250, 600)
            cost = price * random.uniform(0.5, 0.7)
            lead_time = random.randint(20, DEFAULT_MAX_LEAD_TIME_DAYS)
        elif product_type == "Wireless_Earbuds":
            price = random.uniform(80, 250)
            cost = price * random.uniform(0.4, 0.6)
            lead_time = random.randint(15, DEFAULT_MAX_LEAD_TIME_DAYS)
        else: # Fitness_Tracker
            price = random.uniform(50, 150)
            cost = price * random.uniform(0.3, 0.5)
            lead_time = random.randint(10, DEFAULT_MAX_LEAD_TIME_DAYS)

        config_data.append({
            'SKU_ID': sku_id,
            'Price': round(price, 2),
            'Cost': round(cost, 2),
            'Lead_Time_Days': lead_time,
            'Shelf_Life_Days': random.randint(730, 1095), # 2-3 years, tech products don't expire quickly
            'Safety_Stock_Factor': 0, # To be calculated
            'Optimal_Reorder_Point': 0, # To be calculated
            'Min_Order_Quantity': random.choice([500, 1000, 2000]),
            'EOQ': 0 # To be calculated
        })
    return pd.DataFrame(config_data)

def generate_realistic_inventory_data(sales_df, config_df, start_date):
    """
    Generates dummy inventory data based on sales and config, for stockout simulation.
    """
    inventory_data = []
    unique_skus = sales_df['SKU_ID'].unique()
    
    for sku_id in unique_skus:
        sku_sales = sales_df[sales_df['SKU_ID'] == sku_id].copy()
        
        initial_inventory = random.randint(2000, 5000)
        current_inventory = initial_inventory
        
        sku_config = config_df[config_df['SKU_ID'] == sku_id]
        if not sku_config.empty:
            lead_time_days = sku_config['Lead_Time_Days'].iloc[0]
            # Simple assumption for simulation
            reorder_point_sim = sku_sales['Sales_Quantity'].mean() * lead_time_days * 1.5
            eoq_sim = sku_sales['Sales_Quantity'].mean() * 30
        else:
            reorder_point_sim = 1000
            lead_time_days = 20
            eoq_sim = 2000

        last_order_date = None
        
        for index, row in sku_sales.sort_values(by='Date').iterrows():
            date = row['Date']
            sales_quantity = row['Sales_Quantity']
            
            # Check for stockout before sales transaction
            stockout = 1 if current_inventory <= 0 else 0
            
            current_inventory -= sales_quantity
            
            if current_inventory <= reorder_point_sim and (last_order_date is None or (date - last_order_date).days > lead_time_days):
                order_quantity = eoq_sim
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
                'Inventory_Level': max(0, current_inventory),
                'Stockout': stockout
            })

    inventory_df = pd.DataFrame(inventory_data)
    inventory_df.sort_values(by=['SKU_ID', 'Date'], inplace=True)
    inventory_df.drop_duplicates(subset=['SKU_ID', 'Date'], keep='last', inplace=True)
    
    return inventory_df


def generate_realistic_demand_signals_data(sales_df, start_date, end_date):
    """
    Generates realistic dummy demand signals data for mobile & wearables.
    """
    demand_signals_data = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    unique_skus = sales_df['SKU_ID'].unique()
    
    for sku_id in unique_skus:
        for single_date in date_range:
            base_web_traffic = random.randint(1000, 5000)
            base_social_mentions = random.randint(50, 500)
            
            # Correlate signals with promotions
            is_promo = sales_df[(sales_df['Date'] == single_date) & (sales_df['SKU_ID'] == sku_id)]['Promotion'].iloc[0] if not sales_df[(sales_df['Date'] == single_date) & (sales_df['SKU_ID'] == sku_id)].empty else 0
            if is_promo:
                web_traffic = int(base_web_traffic * random.uniform(1.5, 2.5))
                social_mentions = int(base_social_mentions * random.uniform(1.8, 3.0))
            else:
                web_traffic = int(np.random.normal(base_web_traffic, base_web_traffic * 0.2))
                social_mentions = int(np.random.normal(base_social_mentions, base_social_mentions * 0.2))
            
            demand_signals_data.append({
                'Date': single_date,
                'SKU_ID': sku_id,
                'Web_Traffic': max(0, web_traffic),
                'Social_Media_Mentions': max(0, social_mentions),
                'Competitor_Price_Index': round(random.uniform(0.9, 1.1), 2)
            })
            
    demand_signals_df = pd.DataFrame(demand_signals_data)
    demand_signals_df['Date'] = pd.to_datetime(demand_signals_df['Date'])
    
    return demand_signals_df
    
def create_template_df(data):
    """Creates a small DataFrame from a dictionary and returns it as a CSV string."""
    df = pd.DataFrame(data)
    return df.to_csv(index=False)

# --- Core Logic Functions ---

def preprocess_data(sales_df, config_df, demand_signals_df, global_config_df):
    """
    Merges and prepares all dataframes for forecasting and analysis.
    """
    # Merge sales with config and demand signals
    df = pd.merge(sales_df, config_df, on='SKU_ID', how='left')
    
    # Merge with demand signals if available
    if not demand_signals_df.empty:
        df = pd.merge(df, demand_signals_df, on=['Date', 'SKU_ID'], how='left')
    else:
        # Add dummy columns if demand signals are not provided
        df['Web_Traffic'] = 0
        df['Social_Media_Mentions'] = 0
        df['Competitor_Price_Index'] = 1.0
        
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

def forecast_with_xgboost(train_df, test_df):
    """
    Forecasts demand using XGBoost.
    """
    features = ['Promotion', 'Web_Traffic', 'Social_Media_Mentions', 'Competitor_Price_Index', 'Day_of_Week', 'Month', 'Year', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3']
    target = 'Sales_Quantity'
    
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(train_df[features], train_df[target])
    
    predictions = model.predict(test_df[features])
    return predictions

def forecast_with_random_forest(train_df, test_df):
    """
    Forecasts demand using RandomForestRegressor.
    """
    features = ['Promotion', 'Web_Traffic', 'Social_Media_Mentions', 'Competitor_Price_Index', 'Day_of_Week', 'Month', 'Year', 'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3']
    target = 'Sales_Quantity'
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train_df[features], train_df[target])
    
    predictions = model.predict(test_df[features])
    return predictions

def forecast_with_moving_average(series, horizon):
    """
    Forecasts future demand using a simple moving average.
    """
    predictions = []
    # Use the last 30 days as a window for a rolling average
    window_size = min(30, len(series))
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
    
def forecast_with_moving_median(series, horizon):
    """
    Forecasts future demand using a simple moving median.
    """
    predictions = []
    # Use the last 30 days as a window for a rolling median
    window_size = min(30, len(series))
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
    if validation_df.empty or train_df.empty:
        return {'XGBoost': np.inf, 'Random Forest': np.inf, 'Moving Average': np.inf, 'Moving Median': np.inf}

    mae_scores = {}
    
    # XGBoost
    try:
        xgboost_preds = forecast_with_xgboost(train_df, validation_df)
        mae_scores['XGBoost'] = mean_absolute_error(validation_df['Sales_Quantity'], xgboost_preds)
    except Exception as e:
        mae_scores['XGBoost'] = np.inf
        
    # Random Forest
    try:
        rf_preds = forecast_with_random_forest(train_df, validation_df)
        mae_scores['Random Forest'] = mean_absolute_error(validation_df['Sales_Quantity'], rf_preds)
    except Exception as e:
        mae_scores['Random Forest'] = np.inf

    # Moving Average
    try:
        ma_preds = forecast_with_moving_average(train_df['Sales_Quantity'], horizon=len(validation_df))
        mae_scores['Moving Average'] = mean_absolute_error(validation_df['Sales_Quantity'], ma_preds)
    except Exception as e:
        mae_scores['Moving Average'] = np.inf

    # Moving Median
    try:
        mm_preds = forecast_with_moving_median(train_df['Sales_Quantity'], horizon=len(validation_df))
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
    df['Stockout_Rate'] = df.groupby('SKU_ID')['Stockout'].transform(lambda x: x.rolling(window=30, min_periods=1).mean() * 100)
    return df[['Date', 'SKU_ID', 'Stockout_Rate']].drop_duplicates()

def calculate_inventory_kpis(df, service_level_percentage):
    """
    Calculates Safety Stock, Reorder Point, and EOQ for each SKU.
    """
    inventory_kpis = []
    
    # Get Z-score for the desired service level
    z_score = norm.ppf(service_level_percentage / 100.0)
    
    unique_skus = df['SKU_ID'].unique()
    
    for sku_id in unique_skus:
        sku_data = df[df['SKU_ID'] == sku_id].copy()
        
        # Calculate daily demand metrics from historical data
        daily_demand = sku_data['Sales_Quantity'].resample('D').sum().fillna(0)
        
        # Demand variability during lead time
        lead_time_days = sku_data['Lead_Time_Days'].iloc[0]
        
        # To avoid problems with short data series, use rolling window for std
        window_size = min(30, len(daily_demand))
        std_dev_demand_lead_time = daily_demand.rolling(window=window_size).apply(lambda x: median_abs_deviation(x) * np.sqrt(lead_time_days), raw=True).iloc[-1]
        
        # Safety Stock Calculation
        safety_stock = z_score * std_dev_demand_lead_time
        safety_stock = max(0, safety_stock) # Safety stock cannot be negative
        
        # Reorder Point Calculation
        avg_daily_demand = daily_demand.mean()
        reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
        
        # EOQ Calculation
        annual_demand = avg_daily_demand * 365
        ordering_cost = sku_data['Ordering_Cost_Per_Order'].iloc[0]
        holding_cost = sku_data['Holding_Cost_Per_Unit_Per_Day'].iloc[0]
        
        if annual_demand > 0 and holding_cost > 0:
            eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        else:
            eoq = 0
            
        inventory_kpis.append({
            'SKU_ID': sku_id,
            'Safety_Stock': math.ceil(safety_stock),
            'Reorder_Point': math.ceil(reorder_point),
            'Optimal_Order_Quantity': math.ceil(eoq),
            'Service_Level_Used (%)': service_level_percentage
        })
        
    return pd.DataFrame(inventory_kpis)

# --- Streamlit App ---

st.set_page_config(page_title="Demand & Inventory Intelligence", layout="wide")

st.title("Demand & Inventory Intelligence Platform")
st.markdown("A solution for a Mobile & Wearables company to generate demand forecasts and optimize inventory KPIs.")

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
if 'inventory_kpis' not in st.session_state:
    st.session_state.inventory_kpis = pd.DataFrame()

# --- Sidebar for Data Upload and Configuration ---
with st.sidebar:
    st.header("1. Data Input & Configuration")
    
    st.subheader("Run with Sample Data")
    st.markdown("Use this option to see a full demo with realistic data for mobile and wearable products.")
    
    if st.button("Run with Sample Data"):
        st.session_state.sales_df = generate_realistic_sales_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE)
        st.session_state.config_df = generate_realistic_config_data(st.session_state.sales_df)
        st.session_state.demand_signals_df = generate_realistic_demand_signals_data(st.session_state.sales_df, DEFAULT_START_DATE, DEFAULT_END_DATE)
        st.success("Sample data generated successfully!")
    
    st.markdown("---")
    
    st.subheader("Upload Your Own Data (Optional)")
    st.markdown("Download templates for required formats.")
    sales_template = create_template_df({'Date': ['YYYY-MM-DD'], 'SKU_ID': ['Smartphone_X'], 'Sales_Quantity': [100], 'Promotion': [0]})
    st.download_button("Sales Data Template", data=sales_template, file_name='sales_template.csv', mime='text/csv')
    config_template = create_template_df({'SKU_ID': ['Smartphone_X'], 'Price': [799.99], 'Cost': [500.00], 'Lead_Time_Days': [45]})
    st.download_button("SKU Config Template", data=config_template, file_name='config_template.csv', mime='text/csv')
    
    st.markdown("---")
    
    sales_file = st.file_uploader("Upload Sales Data", type="csv")
    if sales_file:
        st.session_state.sales_df = pd.read_csv(sales_file, parse_dates=['Date'])
    
    config_file = st.file_uploader("Upload SKU Config", type="csv")
    if config_file:
        st.session_state.config_df = pd.read_csv(config_file)
        
    demand_signals_file = st.file_uploader("Upload Demand Signals (Optional)", type="csv")
    if demand_signals_file:
        st.session_state.demand_signals_df = pd.read_csv(demand_signals_file, parse_dates=['Date'])
        
    global_config_file = st.file_uploader("Upload Global Config (Optional)", type="csv")
    if global_config_file:
        st.session_state.global_config_df = pd.read_csv(global_config_file)
        
    st.markdown("---")
    st.header("2. Analysis Parameters")
    
    model_choice = st.selectbox(
        "Select Forecasting Model",
        ("Auto-Select Best Model", "XGBoost", "Random Forest", "Moving Average", "Moving Median")
    )
    
    forecast_horizon_days = st.number_input("Forecast Horizon (in days)", min_value=1, max_value=365, value=90)
    
    service_level = st.slider("Desired Service Level (%)", min_value=90, max_value=99, value=95)
    
    if st.button("Run Analysis"):
        if not st.session_state.sales_df.empty and not st.session_state.config_df.empty:
            
            with st.spinner("Preprocessing data and forecasting..."):
                # Preprocess data
                preprocessed_df = preprocess_data(
                    st.session_state.sales_df,
                    st.session_state.config_df,
                    st.session_state.demand_signals_df,
                    st.session_state.global_config_df
                )
                st.session_state.preprocessed_df = preprocessed_df
            
                # Run Forecasting
                all_forecasts = []
                model_summary = []
                unique_skus = preprocessed_df['SKU_ID'].unique()
                
                progress_bar = st.progress(0, "Forecasting in progress...")
                
                for i, sku_id in enumerate(unique_skus):
                    sku_data = preprocessed_df[preprocessed_df['SKU_ID'] == sku_id].copy()
                    
                    selected_model = model_choice
                    if model_choice == "Auto-Select Best Model":
                        selected_model, best_mae = auto_select_best_model(sku_data)
                        model_summary.append({'SKU_ID': sku_id, 'Best_Model': selected_model, 'MAE': round(best_mae, 2)})
                    
                    # Split data for final forecast
                    last_train_date = sku_data['Date'].max() - timedelta(days=forecast_horizon_days)
                    train_df = sku_data[sku_data['Date'] <= last_train_date]
                    
                    last_date = train_df['Date'].max()
                    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon_days, freq='D')
                    
                    future_df = pd.DataFrame({'Date': future_dates, 'SKU_ID': sku_id})
                    
                    # Create future features
                    future_df['Promotion'] = 0
                    future_df['Web_Traffic'] = train_df['Web_Traffic'].mean() if 'Web_Traffic' in train_df.columns else 0
                    future_df['Social_Media_Mentions'] = train_df['Social_Media_Mentions'].mean() if 'Social_Media_Mentions' in train_df.columns else 0
                    future_df['Competitor_Price_Index'] = train_df['Competitor_Price_Index'].mean() if 'Competitor_Price_Index' in train_df.columns else 1.0
                    future_df['Day_of_Week'] = future_df['Date'].dt.dayofweek
                    future_df['Month'] = future_df['Date'].dt.month
                    future_df['Year'] = future_df['Date'].dt.year
                    future_df['Week_of_Year'] = future_df['Date'].dt.isocalendar().week.astype(int)
                    
                    future_df['Sales_Lag_1'] = train_df['Sales_Quantity'].iloc[-1] if len(train_df) >= 1 else 0
                    future_df['Sales_Lag_2'] = train_df['Sales_Quantity'].iloc[-2] if len(train_df) >= 2 else 0
                    future_df['Sales_Lag_3'] = train_df['Sales_Quantity'].iloc[-3] if len(train_df) >= 3 else 0
                    
                    # Run the forecast with the chosen model
                    if selected_model == "XGBoost":
                        forecast_predictions = forecast_with_xgboost(train_df, future_df)
                    elif selected_model == "Random Forest":
                        forecast_predictions = forecast_with_random_forest(train_df, future_df)
                    elif selected_model == "Moving Average":
                        forecast_predictions = forecast_with_moving_average(train_df['Sales_Quantity'], horizon=forecast_horizon_days)
                    elif selected_model == "Moving Median":
                        forecast_predictions = forecast_with_moving_median(train_df['Sales_Quantity'], horizon=forecast_horizon_days)
                    else:
                        forecast_predictions = []
                    
                    forecast_df = future_df.copy()
                    forecast_df['Forecasted_Sales'] = np.maximum(0, forecast_predictions).round(0).astype(int)
                    forecast_df['Model_Used'] = selected_model
                    forecast_df = forecast_df[['Date', 'SKU_ID', 'Forecasted_Sales', 'Model_Used']]
                    all_forecasts.append(forecast_df)
                    
                    progress_bar.progress((i + 1) / len(unique_skus))
                    
                st.session_state.all_forecasts_df = pd.concat(all_forecasts)
                st.session_state.model_selection_summary = pd.DataFrame(model_summary)
                
                # Calculate KPIs
                dummy_inventory_df = generate_realistic_inventory_data(st.session_state.sales_df, st.session_state.config_df, DEFAULT_START_DATE)
                st.session_state.all_stockout_rates_df = calculate_stockout_rate(dummy_inventory_df)
                
                inventory_kpis_df = calculate_inventory_kpis(st.session_state.preprocessed_df, service_level)
                st.session_state.inventory_kpis = inventory_kpis_df
            
            st.success("Analysis complete! See the results in the Dashboard.")


# --- Main Content Area ---
tab1, tab2 = st.tabs(["Dashboard", "Documentation & FAQ"])

with tab1:
    if not st.session_state.sales_df.empty:
        st.header("Dashboard")
        
        unique_skus = st.session_state.sales_df['SKU_ID'].unique()
        
        selected_sku_for_plot = st.selectbox(
            "Select a Product (SKU) to Analyze",
            unique_skus
        )
        
        forecast_roll_up_choice = st.selectbox(
            "Select Time Granularity for Plots",
            ("Daily", "Weekly", "Monthly"),
            index=2
        )
        
        st.subheader(f"Demand Forecast for {selected_sku_for_plot}")
        
        if not st.session_state.all_forecasts_df.empty:
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
                        name=f'Forecast ({model_used})',
                        line=dict(color='red', dash='dot')
                    ))

                fig_forecast.update_layout(
                    title=f"Sales Forecast for {selected_sku_for_plot}",
                    xaxis_title="Date",
                    yaxis_title="Sales Quantity",
                    hovermode="x unified",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.info("No historical sales data to plot.")
        else:
            st.info("Please run the analysis to see the forecast plots.")

        st.subheader("Inventory & KPI Recommendations")
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Inventory KPIs")
            if not st.session_state.inventory_kpis.empty:
                sku_kpis = st.session_state.inventory_kpis[st.session_state.inventory_kpis['SKU_ID'] == selected_sku_for_plot]
                if not sku_kpis.empty:
                    st.metric("Optimal Reorder Point", f"{sku_kpis['Reorder_Point'].iloc[0]:,}")
                    st.metric("Safety Stock", f"{sku_kpis['Safety_Stock'].iloc[0]:,}")
                    st.metric("Optimal Order Quantity (EOQ)", f"{sku_kpis['Optimal_Order_Quantity'].iloc[0]:,}")
                    st.info(f"Calculations based on a {sku_kpis['Service_Level_Used (%)'].iloc[0]}% service level.")
                else:
                    st.info("No inventory KPIs available for this SKU.")
            else:
                st.info("Please run the analysis to calculate inventory KPIs.")
        
        with col2:
            st.subheader("Stockout Rate")
            if not st.session_state.all_stockout_rates_df.empty:
                stockout_plot_df = aggregate_kpi_for_plot(
                    st.session_state.all_stockout_rates_df, selected_sku_for_plot, 'Stockout_Rate', forecast_roll_up_choice, 'mean'
                ).tail(12)

                if not stockout_plot_df.empty:
                    fig_stockout = px.line(
                        stockout_plot_df,
                        x="Date",
                        y="Value",
                        title=f"Historical Stockout Rate (%)",
                        labels={"Value": "Stockout Rate (%)", "Date": forecast_roll_up_choice},
                        color_discrete_sequence=['orange']
                    )
                    fig_stockout.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_stockout, use_container_width=True)
                else:
                    st.info(f"No Stockout Rate data to plot for {selected_sku_for_plot}.")
            else:
                st.info("No Stockout Rate data available. Please run the analysis.")

        if not st.session_state.model_selection_summary.empty:
            st.subheader("Model Performance Summary")
            st.dataframe(st.session_state.model_selection_summary.set_index('SKU_ID'), use_container_width=True)


    else:
        st.info("Please upload your data files or run with sample data to get started.")

with tab2:
    st.header("Documentation & FAQ")
    
    st.markdown("""
        This section provides a detailed breakdown of the data requirements, input parameters, and the calculation methodology behind the Demand & Inventory Intelligence platform.
    """)

    st.subheader("1. Data Tables and Structures")
    st.markdown("The platform requires up to four CSV files. Please use the provided templates to ensure the correct format. If a file is optional, the platform will use default values.")

    st.markdown("#### **`sales.csv` (Historical Sales Data)**")
    st.markdown("""
        * **Purpose:** This is the core dataset for demand forecasting. It contains historical sales records by SKU and date.
        * **Required Columns:**
            * `Date`: The date of the sales transaction (YYYY-MM-DD format).
            * `SKU_ID`: The unique product identifier (e.g., `Smartphone_Galaxy_Fold_X`).
            * `Sales_Quantity`: The number of units sold.
            * `Promotion`: A binary flag (1 for a promotion, 0 for no promotion).
    """)

    st.markdown("#### **`config.csv` (SKU Configuration)**")
    st.markdown("""
        * **Purpose:** Provides critical product-specific parameters used in inventory calculations.
        * **Required Columns:**
            * `SKU_ID`: The unique product identifier.
            * `Price`: The retail price of the SKU.
            * `Cost`: The cost of goods sold (COGS) for the SKU.
            * `Lead_Time_Days`: The number of days from placing a purchase order to receiving the goods. This is crucial for calculating safety stock and reorder points.
    """)
    
    st.markdown("#### **`demand_signals.csv` (Demand Signals - Optional)**")
    st.markdown("""
        * **Purpose:** External factors that can significantly influence demand, such as market trends, competitor actions, and consumer interest. This data is used by the machine learning models (XGBoost, Random Forest) to improve forecast accuracy.
        * **Required Columns:**
            * `Date`: The date of the data point (YYYY-MM-DD format).
            * `SKU_ID`: The unique product identifier.
            * `Web_Traffic`: Website visits or product page views.
            * `Social_Media_Mentions`: The number of mentions or sentiment on social media.
            * `Competitor_Price_Index`: A metric of how a product's price compares to competitors.
    """)

    st.markdown("#### **`global_config.csv` (Global Configuration - Optional)**")
    st.markdown("""
        * **Purpose:** Sets company-wide cost parameters that are used in the Economic Order Quantity (EOQ) calculation.
        * **Required Columns:**
            * `Parameter`: The name of the parameter.
            * `Value`: The value of the parameter.
            * **Example parameters**: `Holding_Cost_Per_Unit_Per_Day` (cost of storing one unit for one day), `Ordering_Cost_Per_Order` (fixed cost per order).
    """)

    st.subheader("2. Forecasting Models & Methodology")
    st.markdown("This platform offers four forecasting models and an auto-selection feature to choose the best one for each SKU.")
    
    st.markdown("""
    * **XGBoost & Random Forest**: These are powerful machine learning models that treat the forecasting problem as a regression task. They are excellent at capturing non-linear relationships between sales and external factors like promotions and demand signals.
    * **Moving Average & Moving Median**: These are simpler, statistical models. They predict future demand based on the average or median of past sales over a specific time window. They are less sensitive to external factors but can be very effective for stable demand patterns.
    * **Auto-Select Best Model**: This feature automatically splits your historical data, trains all four models, and selects the one with the lowest Mean Absolute Error (MAE). This ensures you are using the most accurate model for each product.
    """)

    st.subheader("3. Key Performance Indicators (KPIs) & Calculations")
    st.markdown("The platform calculates three essential inventory KPIs to help you make informed decisions.")

    st.markdown("#### **Reorder Point ($ROP$)**")
    st.markdown(r"""
    The `Reorder Point` is the inventory level at which a new order must be placed to avoid a stockout. It ensures there is enough stock to cover demand during the lead time plus a buffer for uncertainty.
    
    $ROP = (Average\ Daily\ Demand \times Lead\ Time) + Safety\ Stock$
    """)

    st.markdown("#### **Safety Stock ($SS$)**")
    st.markdown(r"""
    `Safety Stock` is the buffer inventory held to protect against unexpected increases in demand or delays in supply (lead time variability). A higher service level requires a larger safety stock.
    
    $SS = Z-score \times \sigma_{LT}$
    
    Where the $Z-score$ corresponds to your desired Service Level (e.g., $1.645$ for a $95\%$ service level), and $\sigma_{LT}$ is the standard deviation of demand during the lead time.
    """)

    st.markdown("#### **Economic Order Quantity ($EOQ$)**")
    st.markdown(r"""
    The `EOQ` is the ideal order quantity a company should purchase to minimize total inventory costs (holding and ordering costs).
    
    $EOQ = \sqrt{\frac{2 \times Annual\ Demand \times Ordering\ Cost}{Holding\ Cost}}$
    """)

    st.markdown("#### **Stockout Rate**")
    st.markdown("""
    The Stockout Rate is a key metric showing the percentage of periods where an SKU was out of stock. It is calculated by simulating historical inventory levels and tracking instances where demand exceeded available stock.
    """)

