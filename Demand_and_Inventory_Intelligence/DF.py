# DF_v3.py - Demand and Inventory Intelligence Streamlit App (Corrected & Enhanced)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import math
import warnings
warnings.filterwarnings('ignore') # Suppress warnings from statsmodels, etc.

# For Plotting
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration for Dummy Data Generation (mobile & wearables context) ---
DEFAULT_NUM_SKUS = 5
DEFAULT_NUM_COMPONENTS_PER_SKU = 3
DEFAULT_START_DATE = datetime(2023, 1, 1)
DEFAULT_END_DATE = datetime(2024, 12, 31)
DEFAULT_PROMOTION_FREQUENCY_DAYS = 60
DEFAULT_MAX_LEAD_TIME_DAYS = 30
DEFAULT_MAX_SKU_SHELF_LIFE_DAYS = 365
DEFAULT_SALES_CHANNELS = ["Direct-to-Consumer", "Retail Partner", "Online Marketplace"]

# --- Default Cost Parameters (used if global_config.csv is not uploaded) ---
DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY = 0.10
DEFAULT_ORDERING_COST_PER_ORDER = 50.00
DEFAULT_STOCKOUT_COST_PER_UNIT = 10.00

# --- Helper Functions ---

def generate_dummy_data():
    """Generates a realistic set of dummy data for a mobile & wearables company."""
    st.info("Generating realistic sample data for Mobile & Wearables products...")
    
    # Generate SKU-specific data
    sku_data = []
    for i in range(DEFAULT_NUM_SKUS):
        sku_id = f"Smartwatch-{i+1}"
        retail_price = round(random.uniform(150, 400), 2)
        shelf_life = random.randint(200, DEFAULT_MAX_SKU_SHELF_LIFE_DAYS)
        lead_time = random.randint(7, DEFAULT_MAX_LEAD_TIME_DAYS)
        safety_stock_factor = random.uniform(1.2, 1.5)
        
        # A more realistic approach would be to have one entry per SKU, not per channel
        sku_data.append([sku_id, retail_price, shelf_life, lead_time, safety_stock_factor])
    
    skus_df = pd.DataFrame(
        sku_data,
        columns=['SKU_ID', 'Retail_Price', 'Shelf_Life_Days', 'Lead_Time_Days', 'Safety_Stock_Factor']
    )
    
    # Generate daily sales data with seasonality and trends
    dates = pd.date_range(start=DEFAULT_START_DATE, end=DEFAULT_END_DATE)
    sales_data = []
    
    for _, sku_row in skus_df.iterrows():
        sku_id = sku_row['SKU_ID']
        base_demand = random.randint(10, 50)
        
        for date in dates:
            sales_channel = random.choice(DEFAULT_SALES_CHANNELS)
            
            # Simulate seasonality (e.g., Q4 holidays)
            month = date.month
            seasonality_multiplier = 1.0
            if month in [10, 11, 12]:
                seasonality_multiplier = random.uniform(1.2, 1.8) # Higher demand in Q4
            
            # Simulate trend (increasing sales over time)
            trend = (date - DEFAULT_START_DATE).days / 365 * 0.1 # 10% annual growth
            
            # Simulate promotions
            is_promotion = random.random() < 0.1
            promotion_effect = 1.0
            promotion_discount = 0.0
            if is_promotion:
                promotion_effect = random.uniform(1.2, 1.5)
                promotion_discount = random.uniform(0.1, 0.3)
            
            # Simulate competitor price effect
            competitor_price = sku_row['Retail_Price'] * random.uniform(0.9, 1.1)
            
            # Simulate ad spend
            online_ad_spend = random.uniform(100, 500) if random.random() > 0.7 else 0
            
            # Calculate daily demand
            demand = int(base_demand * seasonality_multiplier * (1 + trend) * promotion_effect + random.gauss(0, 5))
            demand = max(1, demand) # Demand must be at least 1
            
            sales_data.append({
                'Date': date,
                'SKU_ID': sku_id,
                'Sales_Channel': sales_channel,
                'Demand_Quantity': demand,
                'Promotion_Discount_Rate': promotion_discount,
                'Online_Ad_Spend': online_ad_spend,
                'Competitor_Price': competitor_price
            })
    
    sales_df = pd.DataFrame(sales_data)
    
    # Generate component data
    bom_data = []
    component_ids = [f"Component-C{i+1}" for i in range(DEFAULT_NUM_SKUS * DEFAULT_NUM_COMPONENTS_PER_SKU)]
    for i in range(DEFAULT_NUM_SKUS):
        sku_id = f"Smartwatch-{i+1}"
        components_for_sku = random.sample(component_ids, k=DEFAULT_NUM_COMPONENTS_PER_SKU)
        for comp in components_for_sku:
            cost_per_unit = round(random.uniform(5, 50), 2)
            bom_data.append({
                'SKU_ID': sku_id,
                'Component_ID': comp,
                'Quantity_Per_SKU': random.randint(1, 3),
                'Cost_Per_Unit': cost_per_unit
            })
            
    bom_df = pd.DataFrame(bom_data)
    
    st.session_state.sales_df = sales_df
    st.session_state.skus_df = skus_df
    st.session_state.bom_df = bom_df
    st.success("Sample data loaded successfully!")


# --- Forecasting Model Functions ---
def run_xgboost_forecast(df, forecast_periods, params):
    """Runs XGBoost forecasting model."""
    st.info("Running XGBoost model...")
    df['Date_Ordinal'] = df['ds'].apply(lambda x: x.toordinal())
    features = ['Date_Ordinal', 'Promotion_Discount_Rate', 'Online_Ad_Spend', 'Competitor_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df['y'], test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    model.fit(X_train, y_train)
    
    future = pd.DataFrame({
        'Date_Ordinal': range(df['Date_Ordinal'].max() + 1, df['Date_Ordinal'].max() + forecast_periods + 1)
    })
    
    for feature in features:
        if feature != 'Date_Ordinal':
            future[feature] = df[feature].mean()
            
    forecast = pd.Series(model.predict(future)).clip(lower=0)
    
    return forecast.tolist(), y_test, model.predict(X_test)

def run_rf_forecast(df, forecast_periods, params):
    """Runs Random Forest forecasting model."""
    st.info("Running Random Forest model...")
    df['Date_Ordinal'] = df['ds'].apply(lambda x: x.toordinal())
    features = ['Date_Ordinal', 'Promotion_Discount_Rate', 'Online_Ad_Spend', 'Competitor_Price']
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df['y'], test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    model.fit(X_train, y_train)
    
    future = pd.DataFrame({
        'Date_Ordinal': range(df['Date_Ordinal'].max() + 1, df['Date_Ordinal'].max() + forecast_periods + 1)
    })
    
    for feature in features:
        if feature != 'Date_Ordinal':
            future[feature] = df[feature].mean()
            
    forecast = pd.Series(model.predict(future)).clip(lower=0)
    
    return forecast.tolist(), y_test, model.predict(X_test)

def run_moving_average_forecast(df, forecast_periods, window_size):
    """Runs a Moving Average forecasting model."""
    st.info(f"Running Moving Average model with window size {window_size}...")
    df['Moving_Average'] = df['y'].rolling(window=window_size).mean()
    last_ma = df['Moving_Average'].iloc[-1]
    forecast = [last_ma] * forecast_periods
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    train_df['Moving_Average'] = train_df['y'].rolling(window=window_size).mean()
    last_train_ma = train_df['Moving_Average'].iloc[-1]
    y_pred_test = [last_train_ma] * len(test_df)

    return forecast, test_df['y'].tolist(), y_pred_test

def run_moving_median_forecast(df, forecast_periods, window_size):
    """Runs a Moving Median forecasting model."""
    st.info(f"Running Moving Median model with window size {window_size}...")
    df['Moving_Median'] = df['y'].rolling(window=window_size).median()
    last_mm = df['Moving_Median'].iloc[-1]
    forecast = [last_mm] * forecast_periods
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    train_df['Moving_Median'] = train_df['y'].rolling(window=window_size).median()
    last_train_mm = train_df['Moving_Median'].iloc[-1]
    y_pred_test = [last_train_mm] * len(test_df)
    
    return forecast, test_df['y'].tolist(), y_pred_test

def auto_select_best_model(df, forecast_periods, models_to_test):
    """Selects the best forecasting model based on MAE on a validation set."""
    st.info("Automatically selecting the best model based on Mean Absolute Error (MAE)...")
    
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    
    results = {}
    
    if 'XGBoost' in models_to_test:
        _, y_true_xgb, y_pred_xgb = run_xgboost_forecast(train_df, len(df) - train_size, {'n_estimators': 100, 'max_depth': 3})
        mae_xgb = mean_absolute_error(y_true_xgb, y_pred_xgb)
        results['XGBoost'] = mae_xgb
        st.write(f"XGBoost MAE: {mae_xgb:.2f}")

    if 'Random Forest' in models_to_test:
        _, y_true_rf, y_pred_rf = run_rf_forecast(train_df, len(df) - train_size, {'n_estimators': 100, 'max_depth': 3})
        mae_rf = mean_absolute_error(y_true_rf, y_pred_rf)
        results['Random Forest'] = mae_rf
        st.write(f"Random Forest MAE: {mae_rf:.2f}")
    
    if 'Moving Average' in models_to_test:
        _, y_true_ma, y_pred_ma = run_moving_average_forecast(train_df, len(df) - train_size, window_size=7)
        mae_ma = mean_absolute_error(y_true_ma, y_pred_ma)
        results['Moving Average'] = mae_ma
        st.write(f"Moving Average MAE: {mae_ma:.2f}")

    if 'Moving Median' in models_to_test:
        _, y_true_mm, y_pred_mm = run_moving_median_forecast(train_df, len(df) - train_size, window_size=7)
        mae_mm = mean_absolute_error(y_true_mm, y_pred_mm)
        results['Moving Median'] = mae_mm
        st.write(f"Moving Median MAE: {mae_mm:.2f}")

    best_model = min(results, key=results.get)
    st.success(f"Best model selected: **{best_model}** with MAE: {results[best_model]:.2f}")
    
    return best_model

def run_forecasting(df, selected_model, forecast_periods, params):
    """Main forecasting function that calls the selected model."""
    forecast = []
    y_test_actuals = []
    y_test_predictions = []
    
    df_model = df.rename(columns={'Date': 'ds', 'Demand_Quantity': 'y'})
    df_model = df_model.sort_values('ds').reset_index(drop=True)
    
    if selected_model == "XGBoost":
        forecast, y_test_actuals, y_test_predictions = run_xgboost_forecast(df_model, forecast_periods, params)
    elif selected_model == "Random Forest":
        forecast, y_test_actuals, y_test_predictions = run_rf_forecast(df_model, forecast_periods, params)
    elif selected_model == "Moving Average":
        forecast, y_test_actuals, y_test_predictions = run_moving_average_forecast(df_model, forecast_periods, params['window_size'])
    elif selected_model == "Moving Median":
        forecast, y_test_actuals, y_test_predictions = run_moving_median_forecast(df_model, forecast_periods, params['window_size'])
    else:
        st.error("Invalid model selected.")

    if y_test_actuals and y_test_predictions:
        mae = mean_absolute_error(y_test_actuals, y_test_predictions)
        rmse = np.sqrt(mean_squared_error(y_test_actuals, y_test_predictions))
    else:
        mae, rmse = None, None

    return forecast, mae, rmse

def calculate_reorder_point(historical_demand_df, lead_time_days):
    """Calculates a realistic reorder point based on demand during lead time."""
    # Ensure lead_time_days is a standard Python int
    lead_time_days = int(lead_time_days)
    
    avg_demand = historical_demand_df['Demand_Quantity'].mean()
    std_demand = historical_demand_df['Demand_Quantity'].std()
    
    # Calculate demand during lead time
    avg_demand_during_lead_time = avg_demand * lead_time_days
    std_demand_during_lead_time = std_demand * np.sqrt(lead_time_days)
    
    # Z-score for 95% service level
    service_level_z_score = norm.ppf(0.95)
    
    # Calculate safety stock
    safety_stock = service_level_z_score * std_demand_during_lead_time
    
    reorder_point = math.ceil(avg_demand_during_lead_time + safety_stock)
    
    return reorder_point

def simulate_inventory_for_sku(sku_sales_df, reorder_point, lead_time, holding_cost, stockout_cost):
    """
    Simulates inventory levels and calculates KPIs for a single SKU.
    This function has been refactored to be more robust.
    """
    # Ensure lead_time is a standard Python int
    lead_time = int(lead_time)
    
    inventory = 2 * reorder_point # Start with a reasonable inventory
    total_holding_cost = 0
    total_stockout_cost = 0
    stockout_days = 0
    
    open_orders = [] # A list of (order_arrival_date, order_quantity) tuples
    
    for _, row in sku_sales_df.iterrows():
        date = row['Date']
        demand = row['Demand_Quantity']
        
        # Check for arriving orders
        newly_arrived_orders = [o for o in open_orders if o[0] <= date]
        for order in newly_arrived_orders:
            inventory += order[1]
        open_orders = [o for o in open_orders if o[0] > date]
        
        # Fulfill demand
        if inventory < demand:
            stockout_days += 1
            stockout_quantity = demand - inventory
            total_stockout_cost += stockout_quantity * stockout_cost
            inventory = 0
        else:
            inventory -= demand
        
        # Reorder logic
        if inventory <= reorder_point:
            order_quantity = reorder_point * 2 # A simple order-up-to level
            order_arrival_date = date + timedelta(days=lead_time) # CORRECTED: Ensure lead_time is an int
            open_orders.append((order_arrival_date, order_quantity))
        
        # Calculate holding cost
        total_holding_cost += inventory * holding_cost
            
    return {
        'Total_Holding_Cost': total_holding_cost,
        'Total_Stockout_Cost': total_stockout_cost,
        'Stockout_Rate': (stockout_days / len(sku_sales_df)) * 100,
        'Total_Days': len(sku_sales_df)
    }

def calculate_inventory_kpis(sales_df, skus_df, reorder_df, holding_cost, stockout_cost):
    """Calculates inventory KPIs by running a simulation for each SKU."""
    kpi_data = []
    
    for sku in sales_df['SKU_ID'].unique():
        sku_sales = sales_df[sales_df['SKU_ID'] == sku].sort_values('Date').reset_index(drop=True)
        reorder_point = reorder_df[reorder_df['SKU_ID'] == sku]['Reorder_Point'].iloc[0]
        lead_time = skus_df[skus_df['SKU_ID'] == sku]['Lead_Time_Days'].iloc[0]
        
        kpi_results = simulate_inventory_for_sku(sku_sales, reorder_point, lead_time, holding_cost, stockout_cost)
        kpi_results['SKU_ID'] = sku
        kpi_data.append(kpi_results)
        
    return pd.DataFrame(kpi_data)

# --- Streamlit App UI ---
st.set_page_config(
    page_title="Mobile & Wearables Demand and Inventory Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = pd.DataFrame()
if 'skus_df' not in st.session_state:
    st.session_state.skus_df = pd.DataFrame()
if 'bom_df' not in st.session_state:
    st.session_state.bom_df = pd.DataFrame()
if 'reorder_df' not in st.session_state:
    st.session_state.reorder_df = pd.DataFrame()
if 'kpis_df' not in st.session_state:
    st.session_state.kpis_df = pd.DataFrame()
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if 'costs_df' not in st.session_state:
    st.session_state.costs_df = pd.DataFrame({
        'Holding_Cost_Per_Unit_Per_Day': [DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY],
        'Ordering_Cost_Per_Order': [DEFAULT_ORDERING_COST_PER_ORDER],
        'Stockout_Cost_Per_Unit': [DEFAULT_STOCKOUT_COST_PER_UNIT]
    })
if 'bom_df' not in st.session_state:
    st.session_state.bom_df = pd.DataFrame()

# --- Sidebar for user inputs ---
st.sidebar.header("Configuration")
st.sidebar.markdown("---")

with st.sidebar.expander("1. Upload Data"):
    sales_file = st.file_uploader("Upload Daily Sales Data (CSV)", type=["csv"])
    skus_file = st.file_uploader("Upload SKU Master Data (CSV)", type=["csv"])
    bom_file = st.file_uploader("Upload BOM Data (CSV)", type=["csv"])
    costs_file = st.file_uploader("Upload Global Config (CSV)", type=["csv"])

    if st.button("Use Sample Data"):
        generate_dummy_data()

with st.sidebar.expander("2. Model & Forecast Settings", expanded=True):
    model_options = ["XGBoost", "Random Forest", "Moving Average", "Moving Median", "Auto Select"]
    selected_model = st.selectbox(
        "Select Forecasting Model",
        options=model_options,
        help="Choose a model or let the app automatically select the best one."
    )
    
    forecast_periods = st.number_input(
        "Forecast Horizon (Days)",
        min_value=7,
        max_value=365,
        value=30,
        help="Number of days to forecast into the future."
    )
    
    model_params = {}
    if selected_model in ["XGBoost", "Random Forest"]:
        st.subheader("Model Hyperparameters")
        n_estimators = st.number_input("Number of Estimators", min_value=50, max_value=500, value=100, step=10)
        max_depth = st.number_input("Max Depth", min_value=2, max_value=20, value=5, step=1)
        model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    
    if selected_model in ["Moving Average", "Moving Median"]:
        st.subheader("Model Parameters")
        window_size = st.number_input("Window Size (Days)", min_value=2, max_value=90, value=7, step=1)
        model_params = {'window_size': window_size}

st.sidebar.markdown("---")
if st.sidebar.button("Run Analysis"):
    if st.session_state.sales_df.empty or st.session_state.skus_df.empty:
        st.error("Please upload data or use sample data before running the analysis.")
    else:
        with st.spinner("Running analysis..."):
            merged_df = pd.merge(st.session_state.sales_df, st.session_state.skus_df, on='SKU_ID', how='inner')
            merged_df = pd.merge(merged_df, st.session_state.bom_df, on='SKU_ID', how='left')

            st.subheader("Demand Forecasting Results")
            st.session_state.forecast_df = pd.DataFrame()
            
            for sku in st.session_state.sales_df['SKU_ID'].unique():
                sku_df = merged_df[merged_df['SKU_ID'] == sku]
                
                current_model = selected_model
                if selected_model == "Auto Select":
                    current_model = auto_select_best_model(
                        sku_df.rename(columns={'Date': 'ds', 'Demand_Quantity': 'y'}),
                        forecast_periods,
                        ["XGBoost", "Random Forest", "Moving Average", "Moving Median"]
                    )
                
                if current_model == 'XGBoost':
                    params = {'n_estimators': 100, 'max_depth': 3}
                elif current_model == 'Random Forest':
                    params = {'n_estimators': 100, 'max_depth': 3}
                elif current_model in ['Moving Average', 'Moving Median']:
                    params = {'window_size': 7}
                else:
                    params = model_params

                forecast, mae, rmse = run_forecasting(sku_df, current_model, forecast_periods, params)
                
                if forecast:
                    future_dates = pd.date_range(start=sku_df['Date'].max() + timedelta(days=1), periods=forecast_periods)
                    forecast_results = pd.DataFrame({
                        'Date': future_dates,
                        'SKU_ID': sku,
                        'Forecasted_Demand': forecast
                    })
                    st.session_state.forecast_df = pd.concat([st.session_state.forecast_df, forecast_results])
            
            if not st.session_state.forecast_df.empty:
                st.success("Demand forecasting complete!")
                
                st.subheader("Forecasted Demand Data")
                st.dataframe(st.session_state.forecast_df, use_container_width=True)
                
                st.subheader("Demand Forecast Visualization")
                selected_sku_plot = st.selectbox(
                    "Select an SKU to visualize the forecast",
                    st.session_state.forecast_df['SKU_ID'].unique()
                )
                
                sku_historical = st.session_state.sales_df[st.session_state.sales_df['SKU_ID'] == selected_sku_plot]
                sku_forecast = st.session_state.forecast_df[st.session_state.forecast_df['SKU_ID'] == selected_sku_plot]

                combined_df = pd.concat([
                    sku_historical.rename(columns={'Demand_Quantity': 'Demand_Value'}),
                    sku_forecast.rename(columns={'Forecasted_Demand': 'Demand_Value'})
                ], ignore_index=True)

                fig = px.line(
                    combined_df, 
                    x='Date', 
                    y='Demand_Value', 
                    color='SKU_ID',
                    title=f"Demand Forecast for {selected_sku_plot}"
                )
                
                fig.add_vrect(
                    x0=sku_historical['Date'].max(), x1=sku_forecast['Date'].max(),
                    fillcolor="LightSalmon", opacity=0.5,
                    layer="below", line_width=0,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            st.subheader("Inventory Optimization")
            
            reorder_points = []
            for sku in st.session_state.sales_df['SKU_ID'].unique():
                sku_historical_demand = st.session_state.sales_df[st.session_state.sales_df['SKU_ID'] == sku]
                # Cast lead time to int to prevent TypeError
                sku_lead_time = int(st.session_state.skus_df[st.session_state.skus_df['SKU_ID'] == sku]['Lead_Time_Days'].iloc[0])
                reorder_point = calculate_reorder_point(sku_historical_demand, sku_lead_time)
                reorder_points.append({'SKU_ID': sku, 'Reorder_Point': reorder_point})
            
            st.session_state.reorder_df = pd.DataFrame(reorder_points)
            st.write("Calculated Reorder Points:")
            st.dataframe(st.session_state.reorder_df)
            
            st.subheader("Inventory KPIs")
            holding_cost = st.session_state.costs_df['Holding_Cost_Per_Unit_Per_Day'].iloc[0]
            stockout_cost = st.session_state.costs_df['Stockout_Cost_Per_Unit'].iloc[0]
            
            st.session_state.kpis_df = calculate_inventory_kpis(
                st.session_state.sales_df,
                st.session_state.skus_df,
                st.session_state.reorder_df,
                holding_cost,
                stockout_cost
            )
            st.dataframe(st.session_state.kpis_df)
            
            st.success("Analysis complete!")
