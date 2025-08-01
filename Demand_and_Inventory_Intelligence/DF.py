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
import graphviz

# For Plotting
import plotly.express as px
import plotly.graph_objects as go

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


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


# --- Dummy Data Generation Functions (Used for templates) ---
@st.cache_data
def generate_sales_data(num_skus, start_date, end_date, sales_channels):
    """Generates historical sales data for multiple SKUs across multiple channels."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    sales_data = []

    for i in range(1, num_skus + 1):
        sku_id = f"SKU_{i:03d}"
        for channel in sales_channels:
            base_demand = random.randint(50, 200)
            # Adjust base demand for channels to show differentiation
            if channel == "Amazon":
                base_demand = int(base_demand * 1.2)
            elif channel == "Own Website":
                base_demand = int(base_demand * 0.8)

            seasonality_amplitude = base_demand * 0.3
            trend_slope = random.uniform(0.01, 0.05)
            noise = random.randint(-20, 20)

            for j, date in enumerate(dates):
                seasonality = seasonality_amplitude * np.sin(2 * np.pi * (date.dayofyear / 365))
                trend = trend_slope * j
                quantity = max(0, int(base_demand + seasonality + trend + noise))
                sales_data.append({
                    "Date": date,
                    "SKU_ID": sku_id,
                    "Sales_Quantity": quantity,
                    "Price": round(random.uniform(10, 100), 2),
                    "Customer_Segment": random.choice(["Retail", "Wholesale", "Online"]),
                    "Sales_Channel": channel
                })
    return pd.DataFrame(sales_data)

@st.cache_data
def generate_inventory_data(sales_df, bom_df, start_date):
    """Generates simulated inventory levels for both SKUs and Components based on sales."""
    inventory_data = []
    
    # Get unique SKUs from sales data
    unique_skus = sales_df['SKU_ID'].unique()
    
    # Get unique components from BOM data
    unique_components = bom_df['Component_ID'].unique()

    # Dates for inventory simulation
    dates = pd.date_range(start=start_date, end=sales_df['Date'].max(), freq='D')

    # Initial stock for SKUs
    initial_sku_stock = {sku: random.randint(500, 1500) for sku in unique_skus}
    current_sku_stock = initial_sku_stock.copy()

    # Initial stock for Components
    initial_comp_stock = {comp: random.randint(1000, 3000) for comp in unique_components}
    current_comp_stock = initial_comp_stock.copy()

    for date in dates:
        # Simulate SKU inventory changes
        daily_sku_sales = sales_df[sales_df['Date'] == date].groupby('SKU_ID')['Sales_Quantity'].sum()
        for sku_id, stock in current_sku_stock.items():
            sold = daily_sku_sales.get(sku_id, 0)
            # Simulate replenishment for SKUs
            replenishment = random.randint(0, 50) if random.random() < 0.2 else 0 
            current_sku_stock[sku_id] = max(0, stock - sold + replenishment)
            inventory_data.append({
                "Date": date,
                "Item_ID": sku_id,
                "Item_Type": "Finished_Good",
                "Current_Stock": current_sku_stock[sku_id]
            })
        
        # Simulate Component inventory changes based on SKU sales (simplified consumption)
        # This is a basic simulation; a real system would link to production orders
        for sku_id, sold_qty in daily_sku_sales.items():
            components_consumed = bom_df[bom_df['Parent_SKU_ID'] == sku_id]
            for _, comp_row in components_consumed.iterrows():
                component_id = comp_row['Component_ID']
                qty_required = comp_row['Quantity_Required']
                consumed_qty = sold_qty * qty_required
                current_comp_stock[component_id] = max(0, current_comp_stock.get(component_id, 0) - consumed_qty)
        
        # Add component stock levels for the day
        for comp_id, stock in current_comp_stock.items():
            # Simulate some replenishment for components
            replenishment = random.randint(0, 100) if random.random() < 0.1 else 0
            current_comp_stock[comp_id] = stock + replenishment # Add replenishment after consumption
            inventory_data.append({
                "Date": date,
                "Item_ID": comp_id,
                "Item_Type": bom_df[bom_df['Component_ID'] == comp_id]['Component_Type'].iloc[0], # Get component type from BOM
                "Current_Stock": current_comp_stock[comp_id]
            })

    return pd.DataFrame(inventory_data)

@st.cache_data
def generate_promotion_data(sales_df, promo_frequency_days, sales_channels):
    """Generates dummy promotional data, potentially channel-specific."""
    promotion_data = []
    unique_dates = sorted(sales_df['Date'].unique())
    unique_skus = sales_df['SKU_ID'].unique()

    for i in range(0, len(unique_dates), promo_frequency_days):
        promo_date = unique_dates[i]
        num_promos = random.randint(1, 3) # Number of SKUs on promo on this day
        skus_on_promo = random.sample(list(unique_skus), min(num_promos, len(unique_skus)))

        for sku_id in skus_on_promo:
            # Promotions can be channel-specific or across all channels
            promo_channel = random.choice(sales_channels + ["All"]) # "All" means applies to all channels
            
            channels_for_promo = sales_channels if promo_channel == "All" else [promo_channel]

            for channel in channels_for_promo:
                promotion_data.append({
                    "Date": promo_date,
                    "SKU_ID": sku_id,
                    "Promotion_Type": random.choice(["Discount", "BOGO", "Bundle"]),
                    "Discount_Percentage": round(random.uniform(0.05, 0.30), 2) if "Discount" in random.choice(["Discount", "BOGO", "Bundle"]) else None,
                    "Sales_Channel": channel # Assign channel to promotion
                })
    return pd.DataFrame(promotion_data)

@st.cache_data
def generate_external_factors_data(start_date, end_date):
    """Generates dummy external factors data (assumed to be global/not channel-specific for simplicity)."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    external_factors = []

    for date in dates:
        external_factors.append({
            "Date": date,
            "Economic_Index": round(random.uniform(90, 110), 2),
            "Holiday_Flag": 1 if (date.month == 1 and date.day == 1) or \
                                 (date.month == 12 and date.day == 25) else 0,
            "Temperature_Celsius": round(random.uniform(5, 35), 1),
            "Competitor_Activity_Index": round(random.uniform(0.5, 1.5), 2)
        })
    return pd.DataFrame(external_factors)

@st.cache_data
def generate_lead_times_data(skus, bom_df, max_lead_time_days, max_sku_shelf_life_days):
    """Generates dummy supplier lead times for SKUs and components, including shelf life for SKUs."""
    lead_times = []
    
    # Add SKUs
    for sku_id in skus:
        lead_times.append({
            "Item_ID": sku_id,
            "Item_Type": "Finished_Good",
            "Supplier_ID": f"SUP_{random.randint(1, 3)}",
            "Lead_Time_Days": random.randint(5, max_lead_time_days),
            "Shelf_Life_Days": random.choice([None, random.randint(90, max_sku_shelf_life_days), random.randint(30, 60)]),
            "Min_Order_Quantity": random.choice([1, 10, 50]),
            "Order_Multiple": random.choice([1, 5, 10])
        })
    
    # Add Components from BOM
    unique_components = bom_df['Component_ID'].unique()
    for comp_id in unique_components:
        comp_type = bom_df[bom_df['Component_ID'] == comp_id]['Component_Type'].iloc[0]
        lead_times.append({
            "Item_ID": comp_id,
            "Item_Type": comp_type,
            "Supplier_ID": f"SUP_{random.randint(1, 5)}",
            "Lead_Time_Days": random.randint(5, int(max_lead_time_days * 0.7)),
            "Shelf_Life_Days": None,
            "Min_Order_Quantity": random.choice([100, 500, 1000]),
            "Order_Multiple": random.choice([50, 100, 200])
        })
    
    return pd.DataFrame(lead_times)

@st.cache_data
def generate_bom_data(num_skus, num_components_per_sku):
    """Generates a dummy Bill of Materials (BOM) linking SKUs to components."""
    bom_data = []
    
    component_types = ["Raw_Material", "Packaging", "Sub_Assembly"]
    
    component_count = 1
    for i in range(1, num_skus + 1):
        sku_id = f"SKU_{i:03d}"
        
        num_components = random.randint(1, num_components_per_sku)
        for _ in range(num_components):
            comp_id = f"COMP_{component_count:03d}"
            bom_data.append({
                "Parent_SKU_ID": sku_id,
                "Component_ID": comp_id,
                "Component_Type": random.choice(component_types),
                "Quantity_Required": random.randint(1, 5) # Units of component per unit of SKU
            })
            component_count += 1
            
    return pd.DataFrame(bom_data)
    
@st.cache_data
def generate_cost_config_data(skus, components):
    """Generates dummy cost configuration data."""
    cost_data = []

    # Add SKUs
    for sku_id in skus:
        cost_data.append({
            "Item_ID": sku_id,
            "Item_Type": "Finished_Good",
            "Holding_Cost_Per_Unit_Per_Day": round(DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY * random.uniform(0.5, 1.5), 2),
            "Ordering_Cost_Per_Order": None, # Ordering cost is per order, not per SKU
            "Unit_Cost": round(random.uniform(50, 200), 2)
        })

    # Add Components
    for comp_id in components:
        cost_data.append({
            "Item_ID": comp_id,
            "Item_Type": "Component",
            "Holding_Cost_Per_Unit_Per_Day": round(DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY * random.uniform(0.2, 0.8), 2),
            "Ordering_Cost_Per_Order": None, # Ordering cost is per order, not per component
            "Unit_Cost": round(random.uniform(5, 50), 2)
        })
    
    # Add a global row for ordering cost
    cost_data.append({
        "Item_ID": "GLOBAL_CONFIG",
        "Item_Type": "Global",
        "Holding_Cost_Per_Unit_Per_Day": None,
        "Ordering_Cost_Per_Order": DEFAULT_ORDERING_COST_PER_ORDER,
        "Unit_Cost": None
    })

    return pd.DataFrame(cost_data)


# --- Data Preprocessing ---
def preprocess_data(sales_df, promo_df, external_df, roll_up_choice):
    """
    Combines and aggregates sales data with external factors and promotions
    based on the selected roll-up frequency.
    """
    
    # Ensure 'Date' columns are datetime objects
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    promo_df['Date'] = pd.to_datetime(promo_df['Date'])
    external_df['Date'] = pd.to_datetime(external_df['Date'])

    # Aggregate sales to the chosen frequency
    if roll_up_choice == 'Weekly':
        sales_df['Date'] = sales_df['Date'].dt.to_period('W').dt.start_time
    elif roll_up_choice == 'Monthly':
        sales_df['Date'] = sales_df['Date'].dt.to_period('M').dt.start_time
    
    agg_sales_df = sales_df.groupby(['Date', 'SKU_ID', 'Sales_Channel'])['Sales_Quantity'].sum().reset_index()
    
    # Merge with promotions
    agg_sales_df = pd.merge(agg_sales_df, promo_df.drop_duplicates(subset=['Date', 'SKU_ID', 'Sales_Channel']),
                            on=['Date', 'SKU_ID', 'Sales_Channel'], how='left')
    agg_sales_df['Promotion_Flag'] = agg_sales_df['Promotion_Type'].notna().astype(int)
    
    # Merge with external factors (assuming they are daily)
    if roll_up_choice == 'Weekly':
        external_df['Date'] = external_df['Date'].dt.to_period('W').dt.start_time
        agg_external_df = external_df.groupby('Date').mean().reset_index()
    elif roll_up_choice == 'Monthly':
        external_df['Date'] = external_df['Date'].dt.to_period('M').dt.start_time
        agg_external_df = external_df.groupby('Date').mean().reset_index()
    else: # Daily
        agg_external_df = external_df.copy()
        
    agg_sales_df = pd.merge(agg_sales_df, agg_external_df, on='Date', how='left')
    
    # Handle missing values after merge (fill NaNs)
    agg_sales_df['Promotion_Flag'] = agg_sales_df['Promotion_Flag'].fillna(0)
    
    # Create new features from date
    agg_sales_df['Day_of_Week'] = agg_sales_df['Date'].dt.dayofweek
    agg_sales_df['Month'] = agg_sales_df['Date'].dt.month
    agg_sales_df['Year'] = agg_sales_df['Date'].dt.year

    return agg_sales_df.dropna() # Drop any remaining NaNs

# --- Forecasting Models ---

def train_and_forecast_model(model_name, data_df, sku_id, sales_channel, forecast_horizon):
    """
    Trains a selected model and generates a forecast for a specific SKU and channel.
    Returns a dataframe with the forecast.
    """
    st.info(f"Training {model_name} for SKU: {sku_id}, Channel: {sales_channel}...")
    
    # Filter data for a specific SKU and channel
    sku_channel_df = data_df[(data_df['SKU_ID'] == sku_id) & (data_df['Sales_Channel'] == sales_channel)].copy()
    sku_channel_df.sort_values('Date', inplace=True)

    if sku_channel_df.empty:
        return pd.DataFrame()

    # Define features and target for ML models
    features = [col for col in sku_channel_df.columns if col not in ['Date', 'SKU_ID', 'Sales_Channel', 'Sales_Quantity', 'Promotion_Type']]
    target = 'Sales_Quantity'

    # Handle Moving Average as a special case
    if model_name == 'Moving Average':
        # Calculate a simple moving average for the last 'n' periods
        moving_avg_window = 30 # A configurable parameter
        if len(sku_channel_df) < moving_avg_window:
            st.warning(f"Not enough historical data for SKU {sku_id} to calculate a {moving_avg_window}-day moving average. Skipping.")
            return pd.DataFrame()

        last_n_sales = sku_channel_df['Sales_Quantity'].tail(moving_avg_window)
        forecast_value = last_n_sales.mean()
        
        future_dates = pd.date_range(start=sku_channel_df['Date'].max() + timedelta(days=1), periods=forecast_horizon, freq='D')
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Quantity': forecast_value})
        
        return forecast_df
        
    # Handle Moving Median as a special case
    if model_name == 'Moving Median':
        # Calculate a simple moving median for the last 'n' periods
        moving_median_window = 30 # A configurable parameter
        if len(sku_channel_df) < moving_median_window:
            st.warning(f"Not enough historical data for SKU {sku_id} to calculate a {moving_median_window}-day moving median. Skipping.")
            return pd.DataFrame()
            
        last_n_sales = sku_channel_df['Sales_Quantity'].tail(moving_median_window)
        forecast_value = last_n_sales.median()
        
        future_dates = pd.date_range(start=sku_channel_df['Date'].max() + timedelta(days=1), periods=forecast_horizon, freq='D')
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Quantity': forecast_value})
        
        return forecast_df


    # Split data for ML models (XGBoost and Random Forest)
    X = sku_channel_df[features]
    y = sku_channel_df[target]
    
    # Use a fixed test size for evaluation, then train on all data for final forecast
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == 'XGBoost':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        st.error("Invalid model selected.")
        return pd.DataFrame(), None
        
    model.fit(X_train, y_train)
    
    # Evaluate model performance on test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    st.success(f"{model_name} MAE: {mae:.2f}, MSE: {mse:.2f}")

    # Now, train the model on ALL historical data for the final forecast
    model.fit(X, y)
    
    # Generate future dates and create a DataFrame for forecasting
    last_date = sku_channel_df['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)
    
    # Create a DataFrame for future features. This is a simplification; a real
    # app would need to forecast these external factors.
    future_data = {
        'Date': future_dates
    }
    
    # Populate future data with placeholder values or last known values
    for feature in features:
        if feature in ['Day_of_Week', 'Month', 'Year']:
            future_data[feature] = [d.dayofweek if feature == 'Day_of_Week' else d.month if feature == 'Month' else d.year for d in future_dates]
        elif feature == 'Sales_Channel':
            future_data[feature] = [sales_channel] * forecast_horizon
        else:
            future_data[feature] = [sku_channel_df[feature].iloc[-1]] * forecast_horizon
    
    future_df = pd.DataFrame(future_data)
    future_df.set_index('Date', inplace=True)
    
    # Predict future demand
    forecast_values = model.predict(future_df)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Quantity': np.maximum(0, forecast_values)})
    
    return forecast_df

def run_auto_model_selection(data_df, sku_id, sales_channel, forecast_horizon):
    """
    Trains and evaluates all available models and selects the best one based on MAE.
    """
    
    available_models = ['XGBoost', 'Random Forest', 'Moving Average', 'Moving Median']

    results = {}
    best_model_name = None
    best_mae = float('inf')

    st.subheader("Auto Model Selection in Progress...")

    for model_name in available_models:
        st.write(f"Evaluating {model_name}...")
        
        # We need a reproducible test set to compare models fairly.
        sku_channel_df = data_df[(data_df['SKU_ID'] == sku_id) & (data_df['Sales_Channel'] == sales_channel)].copy()
        if sku_channel_df.empty:
            st.warning(f"No data for SKU: {sku_id}, Channel: {sales_channel}. Skipping model selection.")
            return None, None
            
        # Use last N periods as the test set to mimic a real-world scenario
        test_size = min(int(len(sku_channel_df) * 0.2), 30) # Use a max of 30 periods for testing
        train_df = sku_channel_df.iloc[:-test_size]
        test_df = sku_channel_df.iloc[-test_size:]

        # For statistical models, we'll just run them and compare performance on the
        # last `test_size` periods.
        
        if model_name == 'Moving Average':
            moving_avg_window = 30
            if len(train_df) < moving_avg_window:
                mae = float('inf')
                st.warning("Not enough data for Moving Average.")
            else:
                last_n_sales = train_df['Sales_Quantity'].tail(moving_avg_window)
                forecast_value = last_n_sales.mean()
                forecast_values = [forecast_value] * test_size
                mae = mean_absolute_error(test_df['Sales_Quantity'], forecast_values)
        elif model_name == 'Moving Median':
            moving_median_window = 30
            if len(train_df) < moving_median_window:
                mae = float('inf')
                st.warning("Not enough data for Moving Median.")
            else:
                last_n_sales = train_df['Sales_Quantity'].tail(moving_median_window)
                forecast_value = last_n_sales.median()
                forecast_values = [forecast_value] * test_size
                mae = mean_absolute_error(test_df['Sales_Quantity'], forecast_values)
        else: # ML Models
            features = [col for col in sku_channel_df.columns if col not in ['Date', 'SKU_ID', 'Sales_Channel', 'Sales_Quantity', 'Promotion_Type']]
            X_train = train_df[features]
            y_train = train_df['Sales_Quantity']
            X_test = test_df[features]
            y_test = test_df['Sales_Quantity']
            
            if model_name == 'XGBoost':
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            else: # Random Forest
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

        results[model_name] = mae
        if mae < best_mae:
            best_mae = mae
            best_model_name = model_name
            
    st.success(f"Best model selected: **{best_model_name}** with MAE of {best_mae:.2f}")

    return best_model_name, results


# --- Inventory Optimization ---

def calculate_safety_stock(forecast_df, inventory_df, lead_times_df, service_level, roll_up_choice):
    """
    Calculates safety stock based on forecast, lead time, and a desired service level.
    """
    safety_stock_results = []
    
    unique_skus = forecast_df['SKU_ID'].unique()
    z_score = norm.ppf(service_level) # Z-score for the desired service level

    for sku_id in unique_skus:
        sku_forecast_df = forecast_df[forecast_df['SKU_ID'] == sku_id].copy()
        
        if sku_forecast_df.empty:
            continue
            
        # Get lead time for the SKU
        lead_time_days = lead_times_df[lead_times_df['Item_ID'] == sku_id]['Lead_Time_Days'].iloc[0]
        
        # Determine the number of periods in the lead time
        if roll_up_choice == 'Daily':
            periods_in_lead_time = lead_time_days
        elif roll_up_choice == 'Weekly':
            periods_in_lead_time = math.ceil(lead_time_days / 7)
        else: # Monthly
            periods_in_lead_time = math.ceil(lead_time_days / 30)

        # Get historical demand during lead time periods
        last_historical_date = sku_forecast_df['Date'].min()
        historical_demand_df = inventory_df[(inventory_df['Item_ID'] == sku_id) & (inventory_df['Date'] < last_historical_date)]
        
        if historical_demand_df.empty:
            st.warning(f"No historical data for SKU {sku_id} to calculate demand variability.")
            demand_std = 0
        else:
            demand_std = historical_demand_df.groupby('Date')['Sales_Quantity'].sum().std()

        # Safety Stock formula: Z-score * Std Dev of Demand during Lead Time * sqrt(Lead Time in periods)
        safety_stock = z_score * demand_std * np.sqrt(periods_in_lead_time)
        
        safety_stock_results.append({
            'Item_ID': sku_id,
            'Item_Type': 'Finished_Good',
            'Lead_Time_Days': lead_time_days,
            'Service_Level': service_level,
            'Safety_Stock_Units': round(safety_stock, 2)
        })
        
    return pd.DataFrame(safety_stock_results)


def calculate_reorder_point_and_inventory_cost(forecast_df, safety_stock_df, lead_times_df, cost_config_df):
    """
    Calculates reorder point and total inventory cost.
    """
    
    inventory_results = []
    
    # Get global ordering cost
    ordering_cost_per_order = cost_config_df[cost_config_df['Item_ID'] == 'GLOBAL_CONFIG']['Ordering_Cost_Per_Order'].iloc[0]
    
    for _, row in safety_stock_df.iterrows():
        sku_id = row['Item_ID']
        safety_stock = row['Safety_Stock_Units']
        lead_time_days = row['Lead_Time_Days']
        
        # Get SKU-specific data
        sku_forecast = forecast_df[forecast_df['SKU_ID'] == sku_id]
        sku_cost = cost_config_df[cost_config_df['Item_ID'] == sku_id]

        if sku_forecast.empty or sku_cost.empty:
            continue
            
        holding_cost = sku_cost['Holding_Cost_Per_Unit_Per_Day'].iloc[0]
        unit_cost = sku_cost['Unit_Cost'].iloc[0]
        
        # Average daily forecast demand during lead time
        lead_time_demand = sku_forecast['Forecasted_Quantity'].head(lead_time_days).sum()
        
        # Reorder Point (ROP) = (Demand during Lead Time) + Safety Stock
        reorder_point = lead_time_demand + safety_stock
        
        # Economic Order Quantity (EOQ) formula
        # Simplified assumption: average daily demand is the annual demand divided by 365
        avg_daily_demand = sku_forecast['Forecasted_Quantity'].mean()
        annual_demand = avg_daily_demand * 365
        
        if annual_demand > 0 and holding_cost > 0 and ordering_cost_per_order > 0:
            eoq = np.sqrt((2 * annual_demand * ordering_cost_per_order) / (unit_cost * holding_cost * 365))
        else:
            eoq = 0
            
        # Total Inventory Cost (simplified) = Holding Cost + Ordering Cost
        avg_inventory_level = (eoq / 2) + safety_stock
        total_holding_cost = avg_inventory_level * holding_cost * 365
        
        num_orders_per_year = annual_demand / eoq if eoq > 0 else 0
        total_ordering_cost = num_orders_per_year * ordering_cost_per_order
        
        total_inventory_cost = total_holding_cost + total_ordering_cost

        inventory_results.append({
            'Item_ID': sku_id,
            'Reorder_Point': round(reorder_point, 2),
            'Safety_Stock_Units': round(safety_stock, 2),
            'EOQ': round(eoq, 2),
            'Average_Inventory_Level': round(avg_inventory_level, 2),
            'Total_Holding_Cost_USD': round(total_holding_cost, 2),
            'Total_Ordering_Cost_USD': round(total_ordering_cost, 2),
            'Total_Inventory_Cost_USD': round(total_inventory_cost, 2)
        })

    return pd.DataFrame(inventory_results)


# --- KPI & Analysis Functions ---

def calculate_kpis(sales_df, forecast_df, inventory_df, lead_times_df):
    """
    Calculates key performance indicators like forecast accuracy, stockout rate, and inventory turnover.
    """
    
    kpi_results = {}
    
    # 1. Forecast Accuracy (MAPE - Mean Absolute Percentage Error)
    merged_df = pd.merge(sales_df, forecast_df, on=['Date', 'SKU_ID', 'Sales_Channel'], how='inner')
    if not merged_df.empty:
        merged_df = merged_df[merged_df['Sales_Quantity'] > 0]
        if not merged_df.empty:
            mape = np.mean(np.abs((merged_df['Sales_Quantity'] - merged_df['Forecasted_Quantity']) / merged_df['Sales_Quantity'])) * 100
            kpi_results['MAPE'] = f"{mape:.2f}%"
        else:
            kpi_results['MAPE'] = "N/A (No sales data for forecast period)"
    else:
        kpi_results['MAPE'] = "N/A (No overlapping data)"
        
    # 2. Stockout Rate
    stockout_rates = []
    for sku_id in sales_df['SKU_ID'].unique():
        sku_sales = sales_df[sales_df['SKU_ID'] == sku_id].copy()
        sku_inventory = inventory_df[(inventory_df['Item_ID'] == sku_id) & (inventory_df['Item_Type'] == 'Finished_Good')].copy()
        
        if sku_sales.empty or sku_inventory.empty:
            continue
            
        merged_stockout_df = pd.merge(sku_sales, sku_inventory, left_on=['Date', 'SKU_ID'], right_on=['Date', 'Item_ID'], how='left')
        merged_stockout_df['Is_Stockout'] = (merged_stockout_df['Sales_Quantity'] > 0) & (merged_stockout_df['Current_Stock'] <= 0)
        
        total_sales_days = merged_stockout_df['Date'].nunique()
        stockout_days = merged_stockout_df[merged_stockout_df['Is_Stockout']]['Date'].nunique()
        
        stockout_rate = (stockout_days / total_sales_days) * 100 if total_sales_days > 0 else 0
        stockout_rates.append({'SKU_ID': sku_id, 'Stockout_Rate': stockout_rate})
        
    stockout_rate_df = pd.DataFrame(stockout_rates)
    if not stockout_rate_df.empty:
        kpi_results['Average Stockout Rate'] = f"{stockout_rate_df['Stockout_Rate'].mean():.2f}%"
    else:
        kpi_results['Average Stockout Rate'] = "N/A"

    # 3. Inventory Turnover
    # Simplified calculation: (Total Sales Quantity) / (Average Inventory)
    total_sales_qty = sales_df['Sales_Quantity'].sum()
    avg_inventory = inventory_df[inventory_df['Item_Type'] == 'Finished_Good']['Current_Stock'].mean()
    
    if avg_inventory > 0:
        inventory_turnover = total_sales_qty / avg_inventory
        kpi_results['Inventory Turnover'] = f"{inventory_turnover:.2f}"
    else:
        kpi_results['Inventory Turnover'] = "N/A"
        
    return kpi_results


def aggregate_kpi_for_plot(df, selected_sku, kpi_column, roll_up_choice, aggregation_func):
    """Aggregates a KPI dataframe for plotting."""
    
    if selected_sku and kpi_column in df.columns:
        filtered_df = df[df['SKU_ID'] == selected_sku].copy()
        
        if filtered_df.empty:
            return pd.DataFrame()
            
        # Ensure Date is a datetime object
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
        
        # Aggregate to the chosen frequency
        if roll_up_choice == 'Weekly':
            filtered_df['Date'] = filtered_df['Date'].dt.to_period('W').dt.start_time
        elif roll_up_choice == 'Monthly':
            filtered_df['Date'] = filtered_df['Date'].dt.to_period('M').dt.start_time
        
        if aggregation_func == 'sum':
            aggregated_df = filtered_df.groupby('Date')[kpi_column].sum().reset_index()
        else: # 'mean'
            aggregated_df = filtered_df.groupby('Date')[kpi_column].mean().reset_index()
            
        aggregated_df.rename(columns={kpi_column: 'Value'}, inplace=True)
        return aggregated_df
    
    return pd.DataFrame()


# --- Streamlit UI ---

def main():
    st.set_page_config(layout="wide", page_title="Demand & Inventory Intelligence")
    
    # Custom CSS for a better look
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            border: 2px solid #4CAF50;
            border-radius: 20px;
            color: white;
            background-color: #4CAF50;
        }
        .stButton>button:hover {
            border: 2px solid #45a049;
            background-color: #45a049;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: nowrap;
            border-radius: 4px 4px 0 0;
            background-color: #f0f2f6;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            padding-left: 20px;
            padding-right: 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
            border-bottom: 2px solid #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Demand and Inventory Intelligence")
    st.subheader("A Comprehensive Forecasting and Optimization Solution")
    
    # --- Session State Initialization ---
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'sales_df' not in st.session_state:
        st.session_state.sales_df = pd.DataFrame()
    if 'inventory_df' not in st.session_state:
        st.session_state.inventory_df = pd.DataFrame()
    if 'bom_df' not in st.session_state:
        st.session_state.bom_df = pd.DataFrame()
    if 'promo_df' not in st.session_state:
        st.session_state.promo_df = pd.DataFrame()
    if 'external_factors_df' not in st.session_state:
        st.session_state.external_factors_df = pd.DataFrame()
    if 'lead_times_df' not in st.session_state:
        st.session_state.lead_times_df = pd.DataFrame()
    if 'cost_config_df' not in st.session_state:
        st.session_state.cost_config_df = pd.DataFrame()
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = pd.DataFrame()
    if 'safety_stock_df' not in st.session_state:
        st.session_state.safety_stock_df = pd.DataFrame()
    if 'inventory_cost_df' not in st.session_state:
        st.session_state.inventory_cost_df = pd.DataFrame()
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = pd.DataFrame()
    if 'kpi_results' not in st.session_state:
        st.session_state.kpi_results = {}
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None

    # --- Sidebar for Data Upload and Model Parameters ---
    with st.sidebar:
        st.header("Data Source")
        data_source = st.radio("Choose Data Source", ("Upload Your Own", "Run on Sample Data"))
        
        if data_source == "Upload Your Own":
            st.warning("Please upload all necessary files for a complete analysis.")
            sales_file = st.file_uploader("Upload Sales Data (sales.csv)", type=["csv"])
            inventory_file = st.file_uploader("Upload Inventory Data (inventory.csv)", type=["csv"])
            bom_file = st.file_uploader("Upload Bill of Materials (bom.csv)", type=["csv"])
            promo_file = st.file_uploader("Upload Promotion Data (promotions.csv)", type=["csv"])
            external_file = st.file_uploader("Upload External Factors (external_factors.csv)", type=["csv"])
            lead_time_file = st.file_uploader("Upload Lead Times (lead_times.csv)", type=["csv"])
            cost_file = st.file_uploader("Upload Cost Configuration (cost_config.csv)", type=["csv"])
            
            if st.button("Load Uploaded Data"):
                if all([sales_file, inventory_file, bom_file, promo_file, external_file, lead_time_file, cost_file]):
                    try:
                        st.session_state.sales_df = pd.read_csv(sales_file)
                        st.session_state.inventory_df = pd.read_csv(inventory_file)
                        st.session_state.bom_df = pd.read_csv(bom_file)
                        st.session_state.promo_df = pd.read_csv(promo_file)
                        st.session_state.external_factors_df = pd.read_csv(external_file)
                        st.session_state.lead_times_df = pd.read_csv(lead_time_file)
                        st.session_state.cost_config_df = pd.read_csv(cost_file)
                        st.session_state.data_loaded = True
                        st.success("All data files loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading files: {e}")
                else:
                    st.error("Please upload all required files.")
        
        else: # Run on Sample Data
            with st.expander("Configure Sample Data Generation", expanded=False):
                num_skus = st.slider("Number of SKUs", 1, 20, DEFAULT_NUM_SKUS)
                num_components = st.slider("Components per SKU", 1, 10, DEFAULT_NUM_COMPONENTS_PER_SKU)
                forecast_roll_up_choice = st.radio("Roll-up Frequency", ["Daily", "Weekly", "Monthly"])
            
            if st.button("Generate and Load Sample Data"):
                with st.spinner("Generating sample data..."):
                    bom_df = generate_bom_data(num_skus, num_components)
                    skus = bom_df['Parent_SKU_ID'].unique()
                    components = bom_df['Component_ID'].unique()
                    
                    st.session_state.sales_df = generate_sales_data(num_skus, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS)
                    st.session_state.inventory_df = generate_inventory_data(st.session_state.sales_df, bom_df, DEFAULT_START_DATE)
                    st.session_state.promo_df = generate_promotion_data(st.session_state.sales_df, DEFAULT_PROMOTION_FREQUENCY_DAYS, DEFAULT_SALES_CHANNELS)
                    st.session_state.external_factors_df = generate_external_factors_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
                    st.session_state.lead_times_df = generate_lead_times_data(skus, bom_df, DEFAULT_MAX_LEAD_TIME_DAYS, DEFAULT_MAX_SKU_SHELF_LIFE_DAYS)
                    st.session_state.cost_config_df = generate_cost_config_data(skus, components)
                    st.session_state.bom_df = bom_df
                    st.session_state.data_loaded = True
                    st.success("Sample data generated and loaded successfully!")
    
    if st.session_state.data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.header("Forecasting & Inventory Parameters")
        forecast_roll_up_choice = st.sidebar.radio("Forecast Roll-up", ["Daily", "Weekly", "Monthly"], key='sidebar_rollup')
        forecast_horizon = st.sidebar.slider("Forecast Horizon (in days)", 1, 365, 30)
        
        model_options = ["Auto", 'XGBoost', 'Random Forest', 'Moving Average', 'Moving Median']

        selected_model = st.sidebar.radio("Choose Forecasting Model", model_options)
        
        service_level = st.sidebar.slider("Desired Service Level (%)", 50, 99, 95) / 100
        
        if st.sidebar.button("Run Models and Optimize"):
            with st.spinner("Preprocessing data and running models..."):
                # Preprocess data
                preprocessed_data = preprocess_data(st.session_state.sales_df, st.session_state.promo_df, st.session_state.external_factors_df, 'Daily')
                st.session_state.preprocessed_data = preprocessed_data
                
                # Get unique SKUs and Channels for iteration
                unique_skus = st.session_state.sales_df['SKU_ID'].unique()
                unique_channels = st.session_state.sales_df['Sales_Channel'].unique()
                
                # Run forecasting for all SKUs and Channels
                all_forecasts = []
                for sku in unique_skus:
                    for channel in unique_channels:
                        # Auto model selection
                        if selected_model == "Auto":
                            best_model_for_sku, _ = run_auto_model_selection(preprocessed_data, sku, channel, forecast_horizon)
                            if best_model_for_sku:
                                forecast_df_part = train_and_forecast_model(best_model_for_sku, preprocessed_data, sku, channel, forecast_horizon)
                                if not forecast_df_part.empty:
                                    forecast_df_part['SKU_ID'] = sku
                                    forecast_df_part['Sales_Channel'] = channel
                                    all_forecasts.append(forecast_df_part)
                                    st.session_state.best_model = best_model_for_sku
                        else:
                            forecast_df_part = train_and_forecast_model(selected_model, preprocessed_data, sku, channel, forecast_horizon)
                            if not forecast_df_part.empty:
                                forecast_df_part['SKU_ID'] = sku
                                forecast_df_part['Sales_Channel'] = channel
                                all_forecasts.append(forecast_df_part)
                
                if all_forecasts:
                    st.session_state.forecast_df = pd.concat(all_forecasts, ignore_index=True)
                else:
                    st.error("No forecasts were generated. Please check your data and parameters.")
                    st.session_state.forecast_df = pd.DataFrame()
                    
                if not st.session_state.forecast_df.empty:
                    # Calculate Safety Stock and Inventory Costs
                    st.subheader("Running Inventory Optimization...")
                    st.session_state.safety_stock_df = calculate_safety_stock(st.session_state.forecast_df, st.session_state.sales_df, st.session_state.lead_times_df, service_level, forecast_roll_up_choice)
                    st.session_state.inventory_cost_df = calculate_reorder_point_and_inventory_cost(st.session_state.forecast_df, st.session_state.safety_stock_df, st.session_state.lead_times_df, st.session_state.cost_config_df)
                    st.success("Models have finished running and inventory optimization is complete!")
                    
                    # Calculate KPIs
                    st.subheader("Calculating KPIs...")
                    st.session_state.kpi_results = calculate_kpis(st.session_state.sales_df, st.session_state.forecast_df, st.session_state.inventory_df, st.session_state.lead_times_df)
                    
                else:
                    st.warning("No forecast data available to run inventory optimization.")

    # --- Main Content Area ---
    if st.session_state.data_loaded:
        st.subheader("Data Overview")
        st.write(f"Sales Data: {st.session_state.sales_df.shape[0]} rows")
        st.write(f"SKUs: {st.session_state.sales_df['SKU_ID'].nunique()}")
        st.write(f"Sales Channels: {st.session_state.sales_df['Sales_Channel'].nunique()}")
        
        tab1, tab2, tab3 = st.tabs(["Demand Planning", "Inventory Optimization", "KPIs & Analysis"])
        
        with tab1:
            st.header("Demand Planning")
            
            # Allow user to select SKU and Channel for visualization
            if not st.session_state.forecast_df.empty:
                unique_skus_for_plot = st.session_state.sales_df['SKU_ID'].unique()
                unique_channels_for_plot = st.session_state.sales_df['Sales_Channel'].unique()
                
                selected_sku_for_plot = st.selectbox("Select SKU for Plotting", unique_skus_for_plot)
                selected_channel_for_plot = st.selectbox("Select Sales Channel for Plotting", unique_channels_for_plot)
                
                # Plot historical sales and forecast
                historical_sales = st.session_state.sales_df[(st.session_state.sales_df['SKU_ID'] == selected_sku_for_plot) & (st.session_state.sales_df['Sales_Channel'] == selected_channel_for_plot)]
                forecast_sales = st.session_state.forecast_df[(st.session_state.forecast_df['SKU_ID'] == selected_sku_for_plot) & (st.session_state.forecast_df['Sales_Channel'] == selected_channel_for_plot)]
                
                if not historical_sales.empty and not forecast_sales.empty:
                    historical_sales.set_index('Date', inplace=True)
                    historical_sales = historical_sales.resample('D')['Sales_Quantity'].sum().reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=historical_sales['Date'], y=historical_sales['Sales_Quantity'], mode='lines', name='Historical Sales'))
                    fig.add_trace(go.Scatter(x=forecast_sales['Date'], y=forecast_sales['Forecasted_Quantity'], mode='lines', name='Forecasted Demand', line=dict(color='orange')))
                    
                    fig.update_layout(
                        title=f"Historical Sales and Forecast for {selected_sku_for_plot} ({selected_channel_for_plot})",
                        xaxis_title="Date",
                        yaxis_title="Sales Quantity",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif not historical_sales.empty:
                    st.warning("No forecast data available to plot. Please run the models first.")
                else:
                    st.info("No historical sales data available for the selected SKU and channel.")
            else:
                st.info("Please generate sample data or upload your data and run the models to see the forecast.")


        with tab2:
            st.header("Inventory Optimization")
            if not st.session_state.inventory_cost_df.empty:
                st.subheader("Inventory Metrics and Costs")
                st.dataframe(st.session_state.inventory_cost_df, use_container_width=True)
                
                # Allow user to select an SKU to see its Reorder Point over time (simplified)
                unique_skus_for_inv = st.session_state.inventory_cost_df['Item_ID'].unique()
                selected_sku_for_inv = st.selectbox("Select SKU to Visualize", unique_skus_for_inv, key='inv_plot_sku')
                
                if selected_sku_for_inv:
                    inv_df_sku = st.session_state.inventory_cost_df[st.session_state.inventory_cost_df['Item_ID'] == selected_sku_for_inv].iloc[0]
                    
                    # A static plot to visualize the ROP, Safety Stock, and EOQ
                    fig_inv = go.Figure()
                    fig_inv.add_trace(go.Bar(x=['Reorder Point', 'Safety Stock', 'EOQ'], y=[inv_df_sku['Reorder_Point'], inv_df_sku['Safety_Stock_Units'], inv_df_sku['EOQ']], name='Inventory Levels'))
                    
                    fig_inv.update_layout(
                        title=f"Inventory Levels for {selected_sku_for_inv}",
                        yaxis_title="Units",
                        barmode='group'
                    )
                    st.plotly_chart(fig_inv, use_container_width=True)
            else:
                st.info("No inventory optimization data available. Please run the models first.")


        with tab3:
            st.header("KPIs & Analysis")
            if st.session_state.kpi_results:
                st.subheader("Overall Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Average MAPE", st.session_state.kpi_results.get('MAPE', 'N/A'))
                col2.metric("Avg. Stockout Rate", st.session_state.kpi_results.get('Average Stockout Rate', 'N/A'))
                col3.metric("Inventory Turnover", st.session_state.kpi_results.get('Inventory Turnover', 'N/A'))
                
                # Example of a granular plot - Inventory levels over time
                st.subheader("Inventory Level over Time")
                if not st.session_state.inventory_df.empty:
                    unique_skus_for_inv_plot = st.session_state.inventory_df[st.session_state.inventory_df['Item_Type'] == 'Finished_Good']['Item_ID'].unique()
                    selected_sku_for_inv_plot = st.selectbox("Select SKU to view Inventory", unique_skus_for_inv_plot, key='inv_level_plot_sku')
                    
                    inventory_plot_df = st.session_state.inventory_df[st.session_state.inventory_df['Item_ID'] == selected_sku_for_inv_plot]
                    
                    if not inventory_plot_df.empty:
                        fig_inv_level = px.line(
                            inventory_plot_df,
                            x="Date",
                            y="Current_Stock",
                            title=f"Inventory Level for {selected_sku_for_inv_plot}",
                            labels={"Current_Stock": "Units in Stock", "Date": "Date"}
                        )
                        fig_inv_level.update_layout(hovermode="x unified")
                        st.plotly_chart(fig_inv_level, use_container_width=True)
                    else:
                        st.info("No inventory data to plot for this SKU.")
                else:
                    st.info("No inventory data available. Please load data first.")

    else:
        st.info("Please use the sidebar to either upload your data or run the app on sample data.")
    
if __name__ == "__main__":
    main()
