
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
import math # For math.ceil
import graphviz # Import graphviz

# For Prophet and ARIMA
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA # Corrected import path for ARIMA
import warnings
warnings.filterwarnings('ignore') # Suppress warnings from statsmodels, etc.

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
            "Lead_Time_Days": random.randint(2, max_lead_time_days // 2),
            "Shelf_Life_Days": random.choice([None, random.randint(30, 180), random.randint(7, 20)]),
            "Min_Order_Quantity": random.choice([1, 20, 100]),
            "Order_Multiple": random.choice([1, 10, 20])
        })
    
    return pd.DataFrame(lead_times)

@st.cache_data
def generate_bom_data(skus, num_components_per_sku):
    """Generates dummy BOM data."""
    bom_data = []
    component_id_counter = 1000

    for sku_id in skus:
        num_components = random.randint(1, num_components_per_sku)
        for _ in range(num_components):
            component_id = f"COMP_{component_id_counter:04d}"
            component_id_counter += 1
            current_component_type = random.choice(["Raw_Material", "Sub_Assembly"])
            bom_data.append({
                "Parent_SKU_ID": sku_id,
                "Component_ID": component_id,
                "Quantity_Required": random.randint(1, 5),
                "Component_Type": current_component_type,
                "Shelf_Life_Days": random.choice([None, random.randint(30, 180), random.randint(7, 20)])
            })
    return pd.DataFrame(bom_data)


@st.cache_data
def generate_actual_orders_data(num_skus, start_date, end_date, sales_channels):
    """Generates dummy actual customer orders data across multiple channels."""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    orders_data = []
    order_id_counter = 1

    for date in dates:
        num_orders_today = random.randint(0, 5)
        for _ in range(num_orders_today):
            order_id = f"ORD_{order_id_counter:05d}"
            order_id_counter += 1
            num_line_items = random.randint(1, 3)
            skus_in_order = random.sample([f"SKU_{i:03d}" for i in range(1, num_skus + 1)], min(num_line_items, num_skus))
            order_channel = random.choice(sales_channels) # Channel for this specific order

            for sku_id in skus_in_order:
                orders_data.append({
                    "Date": date,
                    "Order_ID": order_id,
                    "SKU_ID": sku_id,
                    "Ordered_Quantity": random.randint(1, 10),
                    "Sales_Channel": order_channel # Assign channel to order
                })
    return pd.DataFrame(orders_data)

@st.cache_data
def generate_actual_shipments_data(actual_orders_df):
    """Generates dummy actual shipments data based on orders, simulating some stockouts."""
    shipments_data = []
    for index, row in actual_orders_df.iterrows():
        shipped_qty = row['Ordered_Quantity']
        if random.random() < 0.1:
            shipped_qty = max(0, shipped_qty - random.randint(1, 3))
        
        shipments_data.append({
            "Date": row['Date'],
            "Order_ID": row['Order_ID'],
            "SKU_ID": row['SKU_ID'],
            "Shipped_Quantity": shipped_qty,
            "Sales_Channel": row['Sales_Channel'] # Carry over channel to shipment
        })
    return pd.DataFrame(shipments_data)

@st.cache_data
def generate_global_config_data():
    """Generates a dummy global_config.csv for template/sample run."""
    config_data = {
        "Parameter": ["Holding_Cost_Per_Unit_Per_Day", "Ordering_Cost_Per_Order"],
        "Value": [DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY, DEFAULT_ORDERING_COST_PER_ORDER]
    }
    return pd.DataFrame(config_data)


# --- ML and Logic Functions ---
@st.cache_data
def prepare_data_for_forecasting(sales_df, inventory_df_raw, promotions_df, external_factors_df_raw):
    """
    Merges all relevant dataframes and creates features for the ML model.
    Handles optional promotions and external factors data, and sales channels.
    This function specifically prepares data for Finished Goods (SKUs).
    """
    try:
        # Filter inventory_df_raw to only include Finished_Good for merging with sales
        inventory_df_skus = inventory_df_raw[inventory_df_raw['Item_Type'] == 'Finished_Good'].rename(columns={'Item_ID': 'SKU_ID', 'Current_Stock': 'Current_Stock'}).copy()
        
        # Merge sales and SKU inventory data
        merged_df = pd.merge(sales_df, inventory_df_skus[['Date', 'SKU_ID', 'Current_Stock']], on=['Date', 'SKU_ID'], how='left')

        # Handle promotions data, now potentially channel-specific
        if promotions_df.empty:
            merged_df['Promotion_Active'] = 0
            merged_df['Discount_Percentage'] = 0
            # Ensure all possible promo_type columns are initialized to 0 if no promotions data
            for promo_type in ["Discount", "BOGO", "Bundle"]: # Assuming these are the possible types
                merged_df[f'Promo_Type_{promo_type}'] = 0
        else:
            # Promotions are merged on Date, SKU_ID, AND Sales_Channel
            merged_df = pd.merge(merged_df, promotions_df, on=['Date', 'SKU_ID', 'Sales_Channel'], how='left')
            merged_df['Promotion_Active'] = merged_df['Promotion_Type'].notna().astype(int)
            merged_df['Discount_Percentage'] = merged_df['Discount_Percentage'].fillna(0)
            promo_type_dummies = pd.get_dummies(merged_df['Promotion_Type'], prefix='Promo_Type', dummy_na=False)
            merged_df = pd.concat([merged_df, promo_type_dummies], axis=1)
            merged_df = merged_df.drop(columns=['Promotion_Type'])
            # Ensure all possible promo_type columns are present, filling with 0 if not
            for promo_type in ["Discount", "BOGO", "Bundle"]:
                if f'Promo_Type_{promo_type}' not in merged_df.columns:
                    merged_df[f'Promo_Type_{promo_type}'] = 0


        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        if external_factors_df_raw.empty:
            merged_df['Economic_Index'] = 100.0
            merged_df['Temperature_Celsius'] = 20.0
            merged_df['Competitor_Activity_Index'] = 1.0
            merged_df['Is_Holiday'] = 0 # Directly set Is_Holiday if no external factors
        else:
            external_factors_df_raw['Date'] = pd.to_datetime(external_factors_df_raw['Date'])
            # External factors are typically not channel-specific, so merge only on Date
            merged_df = pd.merge(merged_df, external_factors_df_raw, on='Date', how='left')
            # Fill NaNs for external factors with their means or a default value
            for col in ['Economic_Index', 'Temperature_Celsius', 'Competitor_Activity_Index']:
                merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
            
            # Use Holiday_Flag if present, otherwise calculate from fixed holidays
            if 'Holiday_Flag' in merged_df.columns:
                merged_df['Is_Holiday'] = merged_df['Holiday_Flag'].fillna(0)
            else:
                fixed_holidays = [(1, 1), (12, 25)]
                merged_df['Is_Holiday'] = merged_df['Date'].apply(lambda x: 1 if (x.month, x.day) in fixed_holidays else 0)
            merged_df.drop(columns=['Holiday_Flag'], errors='ignore', inplace=True) # Drop original Holiday_Flag if exists


        # Create dummy variables for Sales_Channel
        channel_dummies = pd.get_dummies(merged_df['Sales_Channel'], prefix='Channel')
        merged_df = pd.concat([merged_df, channel_dummies], axis=1)
        
        # Sort by SKU_ID, Sales_Channel, and Date for correct lag feature calculation
        merged_df = merged_df.sort_values(by=['SKU_ID', 'Sales_Channel', 'Date']).reset_index(drop=True)

        merged_df['Year'] = merged_df['Date'].dt.year
        merged_df['Month'] = merged_df['Date'].dt.month
        merged_df['Day'] = merged_df['Date'].dt.day
        merged_df['DayOfWeek'] = merged_df['Date'].dt.dayofweek
        merged_df['DayOfYear'] = merged_df['Date'].dt.dayofyear
        merged_df['WeekOfYear'] = merged_df['Date'].dt.isocalendar().week
        merged_df['Is_Weekend'] = ((merged_df['Date'].dt.dayofweek == 5) | (merged_df['Date'].dt.dayofweek == 6)).astype(int)

        # Calculate lag features using groupby for efficiency
        merged_df['Sales_Quantity_Lag_1'] = merged_df.groupby(['SKU_ID', 'Sales_Channel'])['Sales_Quantity'].shift(1).fillna(0)
        merged_df['Sales_Quantity_Lag_7'] = merged_df.groupby(['SKU_ID', 'Sales_Channel'])['Sales_Quantity'].shift(7).fillna(0)

        merged_df = merged_df.drop(columns=['Price', 'Customer_Segment'], errors='ignore')
        merged_df = merged_df.fillna(0) # Final fillna for any remaining NaNs

        return merged_df

    except Exception as e:
        st.error(f"Error during data preparation: {e}")
        return None

@st.cache_resource
def train_forecast_model(data_df, model_choice):
    """Trains an ML Regressor model for each SKU-Channel combination based on user choice."""
    trained_models = {}
    evaluation_metrics = {}

    # Iterate over unique SKU-Channel combinations
    unique_sku_channels = data_df[['SKU_ID', 'Sales_Channel']].drop_duplicates().values

    for sku_id, channel in unique_sku_channels:
        sku_channel_data = data_df[(data_df['SKU_ID'] == sku_id) & (data_df['Sales_Channel'] == channel)].copy()

        # Define features for tree-based models
        tree_features = [col for col in sku_channel_data.columns if col not in ['SKU_ID', 'Sales_Quantity', 'Date', 'Sales_Channel']]
        
        X_tree = sku_channel_data[tree_features]
        y_tree = sku_channel_data['Sales_Quantity']

        # For time-series models (Prophet/ARIMA), use only Date and Sales_Quantity
        ts_data = sku_channel_data[['Date', 'Sales_Quantity']].rename(columns={'Date': 'ds', 'Sales_Quantity': 'y'})


        if len(sku_channel_data) < 10: # Minimum data points for training
            st.warning(f"Not enough data to train for {sku_id} in {channel}. Skipping.")
            continue

        # Split data for evaluation (80% train, 20% test)
        split_point = int(len(sku_channel_data) * 0.8)
        
        best_model_for_sku_channel = None
        best_rmse = float('inf')
        selected_model_name = ""
        
        current_sku_channel_model_results = {}

        # --- Train XGBoost ---
        if model_choice in ["XGBoost Regressor", "Auto Select Best Model (All)"]:
            X_train_tree, X_test_tree = X_tree.iloc[:split_point], X_tree.iloc[split_point:]
            y_train_tree, y_test_tree = y_tree.iloc[:split_point], y_tree.iloc[split_point:]
            if len(X_train_tree) > 0 and len(X_test_tree) > 0:
                try:
                    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42) 
                    xgb_model.fit(X_train_tree, y_train_tree)
                    y_pred_xgb = xgb_model.predict(X_test_tree)
                    rmse_xgb = np.sqrt(mean_squared_error(y_test_tree, y_pred_xgb))
                    current_sku_channel_model_results["XGBoost Regressor"] = {"model": xgb_model, "rmse": rmse_xgb, "mae": mean_absolute_error(y_test_tree, y_pred_xgb)}
                except Exception as e:
                    st.warning(f"Error training XGBoost for {sku_id} in {channel}: {e}")
            else:
                st.warning(f"Insufficient data for train/test split for {sku_id} in {channel} for XGBoost.")

        # --- Train Random Forest ---
        if model_choice in ["Random Forest Regressor", "Auto Select Best Model (All)"]:
            X_train_tree, X_test_tree = X_tree.iloc[:split_point], X_tree.iloc[split_point:]
            y_train_tree, y_test_tree = y_tree.iloc[:split_point], y_tree.iloc[split_point:]
            if len(X_train_tree) > 0 and len(X_test_tree) > 0:
                try:
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X_train_tree, y_train_tree)
                    y_pred_rf = rf_model.predict(X_test_tree)
                    rmse_rf = np.sqrt(mean_squared_error(y_test_tree, y_pred_rf))
                    current_sku_channel_model_results["Random Forest Regressor"] = {"model": rf_model, "rmse": rmse_rf, "mae": mean_absolute_error(y_test_tree, y_pred_rf)}
                except Exception as e:
                    st.warning(f"Error training Random Forest for {sku_id} in {channel}: {e}")
            else:
                st.warning(f"Insufficient data for train/test split for {sku_id} in {channel} for Random Forest.")

        # --- Train Prophet ---
        if model_choice in ["Prophet (Time Series)", "Auto Select Best Model (All)"]:
            ts_train, ts_test = ts_data.iloc[:split_point], ts_data.iloc[split_point:]
            if len(ts_train) >= 2 and len(ts_test) > 0: # Prophet needs at least 2 data points for training
                try:
                    prophet_model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
                    prophet_model.fit(ts_train)
                    
                    future = prophet_model.make_future_dataframe(periods=len(ts_test), include_history=False)
                    forecast = prophet_model.predict(future)
                    y_pred_prophet = forecast['yhat'].values
                    
                    rmse_prophet = np.sqrt(mean_squared_error(ts_test['y'], y_pred_prophet))
                    current_sku_channel_model_results["Prophet (Time Series)"] = {"model": prophet_model, "rmse": rmse_prophet, "mae": mean_absolute_error(ts_test['y'], y_pred_prophet)}
                except Exception as e:
                    st.warning(f"Error training Prophet for {sku_id} in {channel}: {e}")
            else:
                st.warning(f"Insufficient data for Prophet for {sku_id} in {channel}. Need at least 2 training data points.")

        # --- Train ARIMA ---
        if model_choice in ["ARIMA (Time Series)", "Auto Select Best Model (All)"]:
            ts_train, ts_test = ts_data.iloc[:split_point], ts_data.iloc[split_point:]
            if len(ts_train) >= 5 and len(ts_test) > 0: # ARIMA needs more data, arbitrary 5 for demo
                try:
                    # Using a simple (1,1,1) order for demonstration. Auto-ARIMA would be better for production.
                    arima_model = ARIMA(ts_train['y'], order=(1,1,1))
                    arima_model_fit = arima_model.fit()
                    
                    y_pred_arima = arima_model_fit.forecast(steps=len(ts_test)).values
                    
                    mae_arima = mean_absolute_error(ts_test['y'], y_pred_arima)
                    rmse_arima = np.sqrt(mean_squared_error(ts_test['y'], y_pred_arima))

                    current_sku_channel_model_results["ARIMA (Time Series)"] = {"model": arima_model_fit, "rmse": rmse_arima, "mae": mae_arima}
                except Exception as e:
                    st.warning(f"Error training ARIMA for {sku_id} in {channel}: {e}. Try a different model or more data.")
            else:
                st.warning(f"Insufficient data for ARIMA for {sku_id} in {channel}. Need at least 5 training data points.")

        # --- Select Best Model for SKU-Channel or use chosen model ---
        if model_choice == "Auto Select Best Model (All)":
            if not current_sku_channel_model_results:
                st.error(f"No models could be trained successfully for SKU {sku_id} in {channel}. Skipping forecasting for this combination.")
                continue

            for name, result in current_sku_channel_model_results.items():
                if result["rmse"] < best_rmse:
                    best_rmse = result["rmse"]
                    best_model_for_sku_channel = result["model"]
                    selected_model_name = name
            
            trained_models[(sku_id, channel)] = {"model": best_model_for_sku_channel, "type": selected_model_name}
            evaluation_metrics[(sku_id, channel)] = {"MAE": current_sku_channel_model_results[selected_model_name]["mae"], "RMSE": best_rmse, "Model": selected_model_name}
        else: # If a specific model was chosen, use its result
            if model_choice in current_sku_channel_model_results:
                trained_models[(sku_id, channel)] = {"model": current_sku_channel_model_results[model_choice]["model"], "type": model_choice}
                evaluation_metrics[(sku_id, channel)] = {"MAE": current_sku_channel_model_results[model_choice]["mae"], "RMSE": current_sku_channel_model_results[model_choice]["rmse"], "Model": model_choice}
            else:
                st.error(f"Selected model '{model_choice}' could not be trained for SKU {sku_id} in {channel}. Skipping forecasting for this combination.")

    return trained_models, evaluation_metrics

@st.cache_data
def predict_demand(_trained_models, processed_data, forecast_horizon_days, external_factors_df_raw, all_sales_channels_in_data):
    """Generates future demand forecasts for each SKU-Channel combination using trained models."""
    forecasts = {}
    current_date = processed_data['Date'].max()

    # Pre-calculate common features for all future dates for efficiency
    future_dates_range = pd.date_range(start=current_date + timedelta(days=1), periods=forecast_horizon_days, freq='D')
    
    # Prepare external factors for the forecast horizon
    future_external_factors = pd.DataFrame({'Date': future_dates_range})
    if not external_factors_df_raw.empty:
        future_external_factors = pd.merge(future_external_factors, external_factors_df_raw, on='Date', how='left')
        for col in ['Economic_Index', 'Temperature_Celsius', 'Competitor_Activity_Index']:
            # Fill NaNs in future external factors with the last known value from history or a default
            future_external_factors[col] = future_external_factors[col].fillna(processed_data[col].iloc[-1] if col in processed_data.columns else (100.0 if col == 'Economic_Index' else (20.0 if col == 'Temperature_Celsius' else 1.0)))
        
        if 'Holiday_Flag' in future_external_factors.columns:
            future_external_factors['Is_Holiday'] = future_external_factors['Holiday_Flag'].fillna(0)
        else:
            fixed_holidays = [(1, 1), (12, 25)]
            future_external_factors['Is_Holiday'] = future_external_factors['Date'].apply(lambda x: 1 if (x.month, x.day) in fixed_holidays else 0)
        future_external_factors.drop(columns=['Holiday_Flag'], errors='ignore', inplace=True)
    else:
        future_external_factors['Economic_Index'] = 100.0
        future_external_factors['Temperature_Celsius'] = 20.0
        future_external_factors['Competitor_Activity_Index'] = 1.0
        fixed_holidays = [(1, 1), (12, 25)]
        future_external_factors['Is_Holiday'] = future_external_factors['Date'].apply(lambda x: 1 if (x.month, x.day) in fixed_holidays else 0)


    for (sku_id, channel), model_info in _trained_models.items():
        model = model_info["model"]
        model_type = model_info["type"]
        sku_channel_forecast_data = []

        # Get the last known row for this SKU-Channel combination
        last_sku_channel_row = processed_data[(processed_data['SKU_ID'] == sku_id) & (processed_data['Sales_Channel'] == channel)].sort_values(by='Date', ascending=False).iloc[0]
        
        if model_type in ["XGBoost Regressor", "Random Forest Regressor", "Auto Select Best Model (All)"]:
            # Initialize lagged sales from the last historical data point
            last_sales_quantity = last_sku_channel_row['Sales_Quantity']
            last_sales_quantity_lag_1 = last_sku_channel_row['Sales_Quantity_Lag_1']
            last_sales_quantity_lag_7 = last_sku_channel_row['Sales_Quantity_Lag_7']

            # Get the features the model was trained on from the processed data
            model_features = [col for col in processed_data.columns if col not in ['SKU_ID', 'Sales_Quantity', 'Date', 'Sales_Channel']]

            for i in range(forecast_horizon_days):
                forecast_date = future_dates_range[i]
                
                # Get relevant external factors for this forecast date
                ext_factor_row = future_external_factors[future_external_factors['Date'] == forecast_date].iloc[0]

                future_row_dict = {
                    'Current_Stock': last_sku_channel_row['Current_Stock'], # Assuming stock doesn't change drastically for short forecast
                    'Promotion_Active': 0, # Assume no promotions in forecast unless specified
                    'Discount_Percentage': 0,
                    'Year': forecast_date.year,
                    'Month': forecast_date.month,
                    'Day': forecast_date.day,
                    'DayOfWeek': forecast_date.dayofweek,
                    'DayOfYear': forecast_date.dayofyear,
                    'WeekOfYear': forecast_date.isocalendar().week,
                    'Is_Weekend': 1 if (forecast_date.dayofweek == 5 or forecast_date.dayofweek == 6) else 0,
                    'Sales_Quantity_Lag_1': last_sales_quantity,
                    'Sales_Quantity_Lag_7': last_sales_quantity_lag_7,
                    'Economic_Index': ext_factor_row['Economic_Index'],
                    'Temperature_Celsius': ext_factor_row['Temperature_Celsius'],
                    'Competitor_Activity_Index': ext_factor_row['Competitor_Activity_Index'],
                    'Is_Holiday': ext_factor_row['Is_Holiday']
                }

                # Add all possible channel dummy columns and set them correctly
                for c in all_sales_channels_in_data:
                    future_row_dict[f'Channel_{c}'] = 1 if c == channel else 0

                # Ensure all Promo_Type_ dummies are present, setting to 0 if not
                for col in model_features:
                    if col.startswith('Promo_Type_') and col not in future_row_dict:
                        future_row_dict[col] = 0

                future_df_row = pd.DataFrame([future_row_dict])
                # Ensure the order of columns matches the model's expected features
                future_df_row = future_df_row[model_features]

                try:
                    predicted_quantity = max(0, int(model.predict(future_df_row)[0]))
                except Exception as e:
                    st.error(f"Error predicting for {sku_id} on {forecast_date} with {model_type} for channel {channel}: {e}. Setting to 0.")
                    predicted_quantity = 0

                sku_channel_forecast_data.append({
                    "Date": forecast_date,
                    "SKU_ID": sku_id,
                    "Sales_Channel": channel,
                    "Forecasted_Quantity": predicted_quantity
                })
                # Update lagged sales for the next iteration
                last_sales_quantity_lag_7 = last_sales_quantity_lag_1
                last_sales_quantity_lag_1 = predicted_quantity

        elif model_type == "Prophet (Time Series)":
            future_dates_prophet = pd.DataFrame({'ds': future_dates_range})
            forecast_prophet = model.predict(future_dates_prophet)
            
            for _, row in forecast_prophet.iterrows():
                sku_channel_forecast_data.append({
                    "Date": row['ds'],
                    "SKU_ID": sku_id,
                    "Sales_Channel": channel,
                    "Forecasted_Quantity": max(0, int(row['yhat']))
                })

        elif model_type == "ARIMA (Time Series)":
            try:
                forecast_arima = model.forecast(steps=forecast_horizon_days)
                
                for i, pred_qty in enumerate(forecast_arima):
                    sku_channel_forecast_data.append({
                        "Date": future_dates_range[i],
                        "SKU_ID": sku_id,
                        "Sales_Channel": channel,
                        "Forecasted_Quantity": max(0, int(pred_qty))
                    })
            except Exception as e:
                st.error(f"Error forecasting with ARIMA for {sku_id} in {channel}: {e}. Check model fit or data.")
                # Fallback to zero if ARIMA fails to forecast
                for i in range(forecast_horizon_days):
                    forecast_date = future_dates_range[i]
                    sku_channel_forecast_data.append({
                        "Date": forecast_date,
                        "SKU_ID": sku_id,
                        "Sales_Channel": channel,
                        "Forecasted_Quantity": 0
                    })

        forecasts[(sku_id, channel)] = pd.DataFrame(sku_channel_forecast_data)
    return forecasts

@st.cache_data
def calculate_auto_indent(forecasts, inventory_df_raw, lead_times_df, evaluation_metrics, service_level, reorder_point_days,
                          safety_stock_method, fixed_safety_stock_days, safety_stock_cap_factor,
                          holding_cost_per_unit_per_day_global, ordering_cost_per_order_global):
    """
    Calculates auto-indent (order) recommendations for each SKU.
    Forecasts are aggregated to SKU level for central inventory management.
    Incorporates selected Safety Stock Calculation Method, capping logic, and cost calculations.
    """
    indent_recommendations = []
    current_date = datetime.now()
    z_score = norm.ppf(service_level)

    # Filter inventory for SKUs
    inventory_df_skus = inventory_df_raw[inventory_df_raw['Item_Type'] == 'Finished_Good'].rename(columns={'Item_ID': 'SKU_ID', 'Current_Stock': 'Current_Stock'}).copy()

    # Aggregate channel-level forecasts to SKU-level for central inventory
    sku_level_forecasts = {}
    for (sku_id, channel), forecast_df in forecasts.items():
        if sku_id not in sku_level_forecasts:
            sku_level_forecasts[sku_id] = forecast_df[['Date', 'Forecasted_Quantity']].copy()
        else:
            sku_level_forecasts[sku_id] = pd.merge(
                sku_level_forecasts[sku_id],
                forecast_df[['Date', 'Forecasted_Quantity']],
                on='Date',
                how='outer',
                suffixes=('_x', '_y')
            ).fillna(0)
            sku_level_forecasts[sku_id]['Forecasted_Quantity'] = sku_level_forecasts[sku_id]['Forecasted_Quantity_x'] + sku_level_forecasts[sku_id]['Forecasted_Quantity_y']
            sku_level_forecasts[sku_id] = sku_level_forecasts[sku_id].drop(columns=['Forecasted_Quantity_x', 'Forecasted_Quantity_y'])
        sku_level_forecasts[sku_id] = sku_level_forecasts[sku_id].sort_values(by='Date').reset_index(drop=True)


    for sku_id, forecast_df in sku_level_forecasts.items():
        latest_inventory = inventory_df_skus[inventory_df_skus['SKU_ID'] == sku_id].sort_values(by='Date', ascending=False)
        current_stock = latest_inventory['Current_Stock'].iloc[0] if not latest_inventory.empty else 0

        sku_lead_time_row = lead_times_df[(lead_times_df['Item_ID'] == sku_id) & (lead_times_df['Item_Type'] == 'Finished_Good')]
        if sku_lead_time_row.empty:
            st.warning(f"Warning: Lead time or properties not found for SKU {sku_id}. Skipping auto-indent.")
            continue
        
        lead_time_days = sku_lead_time_row['Lead_Time_Days'].iloc[0]
        sku_shelf_life_days = sku_lead_time_row['Shelf_Life_Days'].iloc[0] if 'Shelf_Life_Days' in sku_lead_time_row.columns else None
        min_order_quantity = sku_lead_time_row['Min_Order_Quantity'].iloc[0] if 'Min_Order_Quantity' in sku_lead_time_row.columns else 1
        order_multiple = sku_lead_time_row['Order_Multiple'].iloc[0] if 'Order_Multiple' in sku_lead_time_row.columns else 1

        holding_cost_per_unit_per_day = holding_cost_per_unit_per_day_global
        ordering_cost_per_order = ordering_cost_per_order_global


        safety_stock = 0
        representative_rmse = 0
        for (s, c), metrics in evaluation_metrics.items():
            if s == sku_id: # Use RMSE from any channel for this SKU as a general indicator
                representative_rmse = metrics.get("RMSE", 0)
                break 

        if safety_stock_method == "King's Method (Statistical)":
            forecast_error_std = representative_rmse
            if forecast_error_std == 0:
                st.warning(f"Warning: No RMSE found for SKU {sku_id}. Using simple safety stock factor for King's Method fallback.")
                safety_stock = forecast_df['Forecasted_Quantity'].head(lead_time_days).sum() * 0.1 # Fallback heuristic
            else:
                safety_stock = int(z_score * forecast_error_std * np.sqrt(lead_time_days))
        elif safety_stock_method == "Fixed Days of Demand":
            average_daily_forecast = forecast_df['Forecasted_Quantity'].head(lead_time_days).mean()
            if np.isnan(average_daily_forecast):
                average_daily_forecast = 0
            safety_stock = int(average_daily_forecast * fixed_safety_stock_days)
        
        original_safety_stock = safety_stock

        if sku_shelf_life_days is not None and sku_shelf_life_days > 0:
            demand_within_shelf_life = forecast_df['Forecasted_Quantity'].head(int(sku_shelf_life_days)).sum()
            max_allowed_safety_stock = int(demand_within_shelf_life * safety_stock_cap_factor)
            
            if safety_stock > max_allowed_safety_stock:
                st.info(f"Safety stock for SKU **{sku_id}** capped from {safety_stock} to {max_allowed_safety_stock} due to shelf life ({sku_shelf_life_days} days).")
                safety_stock = max(0, max_allowed_safety_stock) # Ensure non-negative after capping
        
        safety_stock = max(0, safety_stock)


        forecast_during_lead_time = forecast_df['Forecasted_Quantity'].head(lead_time_days).sum()
        target_stock_level = int(forecast_during_lead_time + safety_stock)

        reorder_point_demand_buffer = forecast_df['Forecasted_Quantity'].head(reorder_point_days).sum()
        reorder_point = int(reorder_point_demand_buffer + safety_stock)

        order_quantity = 0
        if current_stock < reorder_point:
            order_quantity = max(0, target_stock_level - current_stock)
            
            order_quantity = max(order_quantity, min_order_quantity)

            if order_multiple > 0:
                order_quantity = math.ceil(order_quantity / order_multiple) * order_multiple
            order_quantity = int(order_quantity)

        estimated_holding_cost_30_days = target_stock_level * holding_cost_per_unit_per_day * 30
        estimated_ordering_cost = ordering_cost_per_order if order_quantity > 0 else 0


        indent_recommendations.append({
            "Item_ID": sku_id,
            "Item_Type": "Finished_Good",
            "Current_Stock": current_stock,
            "Forecasted_Demand_Lead_Time": forecast_during_lead_time,
            "Lead_Time_Days": lead_time_days,
            "Shelf_Life_Days": sku_shelf_life_days,
            "Calculated_Safety_Stock": original_safety_stock,
            "Capped_Safety_Stock": safety_stock,
            "Target_Stock_Level": target_stock_level,
            "Reorder_Point": reorder_point,
            "Order_Quantity": order_quantity,
            "Min_Order_Quantity": min_order_quantity,
            "Order_Multiple": order_multiple,
            "Estimated_Holding_Cost_30_Days": round(estimated_holding_cost_30_days, 2),
            "Estimated_Ordering_Cost": round(estimated_ordering_cost, 2),
            "Recommendation_Date": current_date
        })
    return pd.DataFrame(indent_recommendations)

@st.cache_data
def calculate_component_indent(sku_indent_recommendations_df, bom_df, inventory_df_raw, lead_times_df, service_level, reorder_point_days,
                               safety_stock_method, fixed_safety_stock_days, safety_stock_cap_factor,
                               holding_cost_per_unit_per_day_global, ordering_cost_per_order_global):
    """
    Calculates auto-indent (order) recommendations for components based on derived demand from SKU orders,
    and also applies safety stock and reorder point logic for components.
    """
    component_indent_recommendations = []
    current_date = datetime.now()
    z_score = norm.ppf(service_level)

    # Filter inventory for components
    inventory_df_components = inventory_df_raw[inventory_df_raw['Item_Type'].isin(['Raw_Material', 'Sub_Assembly'])].rename(columns={'Item_ID': 'Component_ID'}).copy()

    # Calculate raw component requirements first, aggregated across all SKUs
    raw_component_requirements = {}
    for index, sku_row in sku_indent_recommendations_df.iterrows():
        sku_id = sku_row['Item_ID']
        sku_order_quantity = sku_row['Order_Quantity']

        if sku_order_quantity > 0:
            components_for_sku = bom_df[bom_df['Parent_SKU_ID'] == sku_id]
            for _, comp_row in components_for_sku.iterrows():
                component_id = comp_row['Component_ID']
                qty_required = comp_row['Quantity_Required']
                raw_required_quantity = sku_order_quantity * qty_required

                if component_id not in raw_component_requirements:
                    raw_component_requirements[component_id] = {
                        "Component_ID": component_id,
                        "Raw_Required_Quantity_Total": 0,
                        "Component_Type": comp_row['Component_Type'],
                        "Source_SKUs": []
                    }
                raw_component_requirements[component_id]["Raw_Required_Quantity_Total"] += raw_required_quantity
                raw_component_requirements[component_id]["Source_SKUs"].append(f"{sku_id} ({sku_order_quantity})")
    
    # Now iterate through each unique component to apply inventory logic
    for comp_id, comp_data in raw_component_requirements.items():
        component_id = comp_data['Component_ID']
        total_raw_required_quantity = comp_data['Raw_Required_Quantity_Total']
        component_type = comp_data['Component_Type']
        source_skus = comp_data['Source_SKUs']

        latest_inventory = inventory_df_components[inventory_df_components['Component_ID'] == component_id].sort_values(by='Date', ascending=False)
        current_stock = latest_inventory['Current_Stock'].iloc[0] if not latest_inventory.empty else 0

        comp_lead_time_row = lead_times_df[(lead_times_df['Item_ID'] == component_id) & (lead_times_df['Item_Type'] == component_type)]
        if comp_lead_time_row.empty:
            st.warning(f"Warning: Lead time or properties not found for Component {component_id}. Skipping auto-indent.")
            continue
        
        lead_time_days = comp_lead_time_row['Lead_Time_Days'].iloc[0]
        comp_shelf_life_days = comp_lead_time_row['Shelf_Life_Days'].iloc[0] if 'Shelf_Life_Days' in comp_lead_time_row.columns else None
        min_order_quantity = comp_lead_time_row['Min_Order_Quantity'].iloc[0] if 'Min_Order_Quantity' in comp_lead_time_row.columns else 1
        order_multiple = comp_lead_time_row['Order_Multiple'].iloc[0] if 'Order_Multiple' in comp_lead_time_row.columns else 1

        holding_cost_per_unit_per_day = holding_cost_per_unit_per_day_global
        ordering_cost_per_order = ordering_cost_per_order_global

        safety_stock = 0
        
        # Estimate average daily derived demand for components
        # This assumes the total_raw_required_quantity is for the entire SKU forecast horizon
        # A proxy for SKU forecast horizon days is used if no SKU orders are driving demand
        sku_forecast_horizon_days = sku_indent_recommendations_df['Forecasted_Demand_Lead_Time'].sum() / sku_indent_recommendations_df['Order_Quantity'].sum() if sku_indent_recommendations_df['Order_Quantity'].sum() > 0 else 30 
        if sku_forecast_horizon_days == 0: sku_forecast_horizon_days = 30 # Avoid division by zero

        average_daily_derived_demand = total_raw_required_quantity / sku_forecast_horizon_days

        if safety_stock_method == "King's Method (Statistical)":
            # For components, without a direct RMSE, we use a heuristic for demand variability.
            # A factor of 0.5 is used to represent variability in derived demand.
            safety_stock = int(z_score * average_daily_derived_demand * np.sqrt(lead_time_days) * 0.5) 
            if safety_stock == 0 and average_daily_derived_demand > 0: 
                 safety_stock = int(average_daily_derived_demand * 0.1 * lead_time_days) # Fallback heuristic
        elif safety_stock_method == "Fixed Days of Demand":
            safety_stock = int(average_daily_derived_demand * fixed_safety_stock_days)
        
        original_safety_stock = safety_stock

        if comp_shelf_life_days is not None and comp_shelf_life_days > 0:
            demand_within_shelf_life = average_daily_derived_demand * comp_shelf_life_days
            max_allowed_safety_stock = int(demand_within_shelf_life * safety_stock_cap_factor)
            
            if safety_stock > max_allowed_safety_stock:
                st.info(f"Safety stock for Component **{component_id}** capped from {safety_stock} to {max_allowed_safety_stock} due to shelf life ({comp_shelf_life_days} days).")
                safety_stock = max(0, max_allowed_safety_stock) # Ensure non-negative after capping
        
        safety_stock = max(0, safety_stock)

        # Derived demand during component lead time
        derived_demand_lead_time = average_daily_derived_demand * lead_time_days
        target_stock_level = int(derived_demand_lead_time + safety_stock)

        reorder_point_derived_demand_buffer = average_daily_derived_demand * reorder_point_days
        reorder_point = int(reorder_point_derived_demand_buffer + safety_stock)

        order_quantity = 0
        if current_stock < reorder_point:
            order_quantity = max(0, target_stock_level - current_stock)
            
            order_quantity = max(order_quantity, min_order_quantity)

            if order_multiple > 0:
                order_quantity = math.ceil(order_quantity / order_multiple) * order_multiple
            order_quantity = int(order_quantity)

        estimated_holding_cost_30_days = target_stock_level * holding_cost_per_unit_per_day * 30
        estimated_ordering_cost = ordering_cost_per_order if order_quantity > 0 else 0

        component_indent_recommendations.append({
            "Item_ID": component_id,
            "Item_Type": component_type,
            "Current_Stock": current_stock,
            "Derived_Demand_Lead_Time": int(derived_demand_lead_time),
            "Lead_Time_Days": lead_time_days,
            "Shelf_Life_Days": comp_shelf_life_days,
            "Calculated_Safety_Stock": original_safety_stock,
            "Capped_Safety_Stock": safety_stock,
            "Target_Stock_Level": target_stock_level,
            "Reorder_Point": reorder_point,
            "Order_Quantity": order_quantity,
            "Min_Order_Quantity": min_order_quantity,
            "Order_Multiple": order_multiple,
            "Estimated_Holding_Cost_30_Days": round(estimated_holding_cost_30_days, 2),
            "Estimated_Ordering_Cost": round(estimated_ordering_cost, 2),
            "Recommendation_Date": current_date,
            "Source_SKUs_Driving_Demand": ", ".join(source_skus)
        })
    return pd.DataFrame(component_indent_recommendations)


@st.cache_data
def calculate_fill_rates(orders_df, shipments_df):
    """
    Calculates Order Fill Rate, Line Fill Rate, and Unit Fill Rate, now considering Sales_Channel.
    """
    if orders_df.empty or shipments_df.empty:
        return {
            "Order Fill Rate": "N/A",
            "Line Fill Rate": "N/A",
            "Unit Fill Rate": "N/A"
        }

    orders_df['Date'] = pd.to_datetime(orders_df['Date'])
    shipments_df['Date'] = pd.to_datetime(shipments_df['Date'])

    # Merge on Date, Order_ID, SKU_ID, and Sales_Channel
    merged_fill_rate_df = pd.merge(orders_df, shipments_df, on=['Date', 'Order_ID', 'SKU_ID', 'Sales_Channel'], how='left', suffixes=('_ordered', '_shipped'))
    merged_fill_rate_df['Shipped_Quantity'] = merged_fill_rate_df['Shipped_Quantity'].fillna(0)

    total_units_ordered = merged_fill_rate_df['Ordered_Quantity'].sum()
    total_units_shipped = merged_fill_rate_df['Shipped_Quantity'].sum()
    unit_fill_rate = (total_units_shipped / total_units_ordered * 100) if total_units_ordered > 0 else 0

    merged_fill_rate_df['Line_Fulfilled'] = (merged_fill_rate_df['Ordered_Quantity'] == merged_fill_rate_df['Shipped_Quantity']).astype(int)
    total_lines = len(merged_fill_rate_df)
    fulfilled_lines = merged_fill_rate_df['Line_Fulfilled'].sum()
    line_fill_rate = (fulfilled_lines / total_lines * 100) if total_lines > 0 else 0

    # Order fulfillment status also needs to consider channel if orders can be channel-specific
    # For now, we'll keep it at Order_ID level, assuming an order is fulfilled if all its lines across all channels are.
    # A more granular approach would be to track Order Fill Rate per channel.
    order_fulfillment_status = merged_fill_rate_df.groupby('Order_ID')['Line_Fulfilled'].min()
    total_orders = len(order_fulfillment_status)
    fulfilled_orders = order_fulfillment_status.sum()
    order_fill_rate = (fulfilled_orders / total_orders * 100) if total_orders > 0 else 0

    return {
        "Order Fill Rate": f"{order_fill_rate:.2f}%",
        "Line Fill Rate": f"{line_fill_rate:.2f}%",
        "Unit Fill Rate": f"{unit_fill_rate:.2f}%"
    }

@st.cache_data
def calculate_stockout_rate_over_time(orders_df, shipments_df):
    """
    Calculates the stockout rate over time (daily) for each SKU.
    Stockout rate is (Ordered - Shipped) / Ordered * 100.
    """
    if orders_df.empty or shipments_df.empty:
        return pd.DataFrame()

    orders_df['Date'] = pd.to_datetime(orders_df['Date'])
    shipments_df['Date'] = pd.to_datetime(shipments_df['Date'])

    # Merge orders and shipments to compare ordered vs. shipped quantities
    merged_df = pd.merge(orders_df, shipments_df, on=['Date', 'Order_ID', 'SKU_ID', 'Sales_Channel'], how='left', suffixes=('_ordered', '_shipped'))
    merged_df['Shipped_Quantity'] = merged_df['Shipped_Quantity'].fillna(0)

    # Calculate unfulfilled quantity
    merged_df['Unfulfilled_Quantity'] = merged_df['Ordered_Quantity'] - merged_df['Shipped_Quantity']
    merged_df['Unfulfilled_Quantity'] = merged_df['Unfulfilled_Quantity'].apply(lambda x: max(0, x)) # Ensure non-negative

    # Group by Date and SKU_ID to calculate daily stockout rate per SKU
    daily_stockout = merged_df.groupby(['Date', 'SKU_ID']).agg(
        Total_Ordered=('Ordered_Quantity', 'sum'),
        Total_Unfulfilled=('Unfulfilled_Quantity', 'sum')
    ).reset_index()

    # Calculate Stockout Rate
    daily_stockout['Stockout_Rate'] = (daily_stockout['Total_Unfulfilled'] / daily_stockout['Total_Ordered']) * 100
    daily_stockout['Stockout_Rate'] = daily_stockout['Stockout_Rate'].fillna(0) # If Total_Ordered is 0, stockout rate is 0
    daily_stockout['Stockout_Rate'] = daily_stockout['Stockout_Rate'].replace([np.inf, -np.inf], 0) # Handle division by zero edge case

    return daily_stockout[['Date', 'SKU_ID', 'Stockout_Rate']]


@st.cache_data
def calculate_inventory_kpis(sales_df, inventory_df_raw):
    """Calculates daily inventory turnover and days of inventory outstanding per SKU."""
    kpi_data = []
    
    # Ensure Date is datetime and sort
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    inventory_df_raw['Date'] = pd.to_datetime(inventory_df_raw['Date'])
    
    # Filter inventory for SKUs only for these KPIs, as sales are SKU-level
    inventory_df_skus = inventory_df_raw[inventory_df_raw['Item_Type'] == 'Finished_Good'].rename(columns={'Item_ID': 'SKU_ID'}).copy()

    # Aggregate sales to SKU-daily level (sum across channels)
    daily_sales_sku = sales_df.groupby(['Date', 'SKU_ID'])['Sales_Quantity'].sum().reset_index()

    # Merge daily sales with SKU inventory
    merged_kpi_df = pd.merge(daily_sales_sku, inventory_df_skus[['Date', 'SKU_ID', 'Current_Stock']], on=['Date', 'SKU_ID'], how='left')
    merged_kpi_df = merged_kpi_df.sort_values(by=['SKU_ID', 'Date']).reset_index(drop=True)

    for sku_id in merged_kpi_df['SKU_ID'].unique():
        sku_kpi_data = merged_kpi_df[merged_kpi_df['SKU_ID'] == sku_id].copy()
        
        # Calculate average inventory for a rolling 7-day window
        sku_kpi_data['Avg_Inventory_7_Day'] = sku_kpi_data['Current_Stock'].rolling(window=7, min_periods=1).mean()
        sku_kpi_data['Sales_Quantity_7_Day_Sum'] = sku_kpi_data['Sales_Quantity'].rolling(window=7, min_periods=1).sum()

        # Inventory Turnover (Annualized based on 7-day window)
        sku_kpi_data['Inventory_Turnover'] = (sku_kpi_data['Sales_Quantity_7_Day_Sum'] / sku_kpi_data['Avg_Inventory_7_Day']) * (365 / 7)
        sku_kpi_data['Inventory_Turnover'] = sku_kpi_data['Inventory_Turnover'].replace([np.inf, -np.inf], np.nan).fillna(0) # Handle division by zero or NaN

        # Days of Inventory Outstanding (DIO)
        sku_kpi_data['Days_Inventory_Outstanding'] = (sku_kpi_data['Avg_Inventory_7_Day'] / sku_kpi_data['Sales_Quantity_7_Day_Sum']) * 7
        sku_kpi_data['Days_Inventory_Outstanding'] = sku_kpi_data['Days_Inventory_Outstanding'].replace([np.inf, -np.inf], np.nan).fillna(0) # Handle division by zero or NaN
        
        # Only keep relevant columns
        sku_kpi_data = sku_kpi_data[['Date', 'SKU_ID', 'Inventory_Turnover', 'Days_Inventory_Outstanding']]
        kpi_data.append(sku_kpi_data)

    if kpi_data:
        return pd.concat(kpi_data, ignore_index=True)
    return pd.DataFrame()

# Function to aggregate data for plotting based on roll-up choice
def aggregate_kpi_for_plot(df, sku_id, value_col, roll_up_choice, aggregation_type='mean'):
    """Aggregates KPI data for plotting based on roll-up choice and selected SKU."""
    df_filtered = df[df['SKU_ID'] == sku_id].copy()
    
    if df_filtered.empty:
        return pd.DataFrame()

    if roll_up_choice == "Day":
        return df_filtered.rename(columns={value_col: 'Value'})[['Date', 'Value']]
    elif roll_up_choice == "Week":
        df_filtered['Period'] = df_filtered['Date'].dt.to_period('W').astype(str)
        if aggregation_type == 'mean':
            agg_df = df_filtered.groupby('Period')[value_col].mean().reset_index()
        else: # sum
            agg_df = df_filtered.groupby('Period')[value_col].sum().reset_index()
        agg_df['Date'] = agg_df['Period'].apply(lambda x: pd.Period(x).start_time)
        return agg_df.rename(columns={value_col: 'Value'})[['Date', 'Value']]
    elif roll_up_choice == "Month":
        df_filtered['Period'] = df_filtered['Date'].dt.to_period('M').astype(str)
        if aggregation_type == 'mean':
            agg_df = df_filtered.groupby('Period')[value_col].mean().reset_index()
        else: # sum
            agg_df = df_filtered.groupby('Period')[value_col].sum().reset_index()
        agg_df['Date'] = agg_df['Period'].apply(lambda x: pd.Period(x).start_time)
        return agg_df.rename(columns={value_col: 'Value'})[['Date', 'Value']]


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Demand Forecasting & Auto-Indent")

st.title("📦 QuantumFlow: Demand & Inventory Intelligence")
st.markdown("Adjust the parameters below to simulate demand forecasting and generate order recommendations for SKUs and their components.")

# Initialize session state variables if they don't exist
if 'forecast_run' not in st.session_state:
    st.session_state.forecast_run = False
    st.session_state.processed_data_df = pd.DataFrame() # Processed SKU sales data for forecasting
    st.session_state.all_forecasts_df = pd.DataFrame() # SKU forecasts
    st.session_state.sku_indent_recommendations_df = pd.DataFrame() # SKU indent
    st.session_state.component_indent_recommendations_df = pd.DataFrame() # NEW: Component indent
    st.session_state.all_inventory_kpis_df = pd.DataFrame() # SKU KPIs
    st.session_state.all_stockout_rates_df = pd.DataFrame() # SKU Stockout
    st.session_state.sales_df = pd.DataFrame()
    st.session_state.inventory_df_raw = pd.DataFrame() # Raw inventory data (SKUs + Components)
    st.session_state.actual_orders_df = pd.DataFrame()
    st.session_state.actual_shipments_df = pd.DataFrame()
    st.session_state.all_sales_channels_in_data = []
    st.session_state.evaluation_metrics = {} # SKU evaluation metrics
    st.session_state.lead_times_df = pd.DataFrame()
    st.session_state.bom_df = pd.DataFrame()
    st.session_state.holding_cost_per_unit_per_day_param = DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY
    st.session_state.ordering_cost_per_order_param = DEFAULT_ORDERING_COST_PER_ORDER


# --- Sidebar for inputs ---
st.sidebar.header("⚙️ Simulation Parameters")

with st.sidebar.container(border=True):
    st.subheader("Forecast & Inventory Settings")
    forecast_horizon_days_input = st.slider("Forecast Horizon (Days)", 7, 90, 30)
    service_level_input = st.slider("Service Level (%)", 80, 99, 95) / 100.0
    reorder_point_days_input = st.number_input(
        "Reorder Point Buffer (Days)",
        min_value=0,
        max_value=30, # Arbitrary max, can be adjusted
        value=7,
        help="Number of days of forecasted demand to add as a buffer to the reorder point. Set to 0 for no buffer."
    )

with st.sidebar.container(border=True):
    st.subheader("Safety Stock Method")
    safety_stock_method_choice = st.selectbox(
        "Select Safety Stock Calculation Method",
        ("King's Method (Statistical)", "Fixed Days of Demand")
    )

    fixed_safety_stock_days_input = 0
    if safety_stock_method_choice == "Fixed Days of Demand":
        fixed_safety_stock_days_input = st.slider("Fixed Safety Stock Days", 1, 30, 7)

    safety_stock_cap_factor_input = st.slider(
        "Safety Stock Cap (x Shelf Life Demand)",
        0.5, 3.0, 1.5, 0.1,
        help="This caps safety stock at a multiple of the forecasted demand during an item's shelf life.\n\n"
             "- **0.5 - 1.0:** Suitable for very fast-moving, highly perishable, or extremely expensive items to minimize waste.\n"
             "- **1.0 - 1.5:** A common balance for moderately perishable or sensitive items.\n"
             "- **2.0 - 3.0:** For slower-moving items with longer shelf lives where availability is prioritized and holding costs are less critical."
    )

with st.sidebar.container(border=True):
    st.subheader("ML Model Selection")
    model_choice = st.selectbox(
        "Select ML Model",
        ("Auto Select Best Model (All)", "XGBoost Regressor", "Random Forest Regressor", "Prophet (Time Series)", "ARIMA (Time Series)")
    )

st.sidebar.markdown("---")
st.sidebar.header("📥 Upload Your Data")
st.sidebar.markdown("Please upload your data in CSV format. You can download templates below.")

# Data Uploaders
with st.sidebar.expander("Upload Data Files"):
    uploaded_sales_file = st.file_uploader("Upload Sales Data (sales_data.csv)", type="csv")
    uploaded_inventory_file = st.file_uploader("Upload Inventory Data (inventory_data.csv)", type="csv", help="Should include Item_ID and Item_Type (Finished_Good/Raw_Material/Sub_Assembly)")
    uploaded_promotions_file = st.file_uploader("Upload Promotions Data (promotions_data.csv)", type="csv")
    uploaded_external_factors_file = st.file_uploader("Upload External Factors Data (external_factors_data.csv)", type="csv")
    uploaded_lead_times_file = st.file_uploader("Upload Lead Times Data (lead_times_data.csv)", type="csv", help="Should include Item_ID and Item_Type (Finished_Good/Raw_Material/Sub_Assembly)")
    uploaded_bom_file = st.file_uploader("Upload BOM Data (bom_data.csv)", type="csv")
    uploaded_actual_orders_file = st.file_uploader("Upload Actual Orders Data (actual_orders.csv)", type="csv")
    uploaded_actual_shipments_file = st.file_uploader("Upload Actual Shipments Data (actual_shipments.csv)", type="csv")
    uploaded_global_config_file = st.file_uploader("Upload Global Config Data (global_config.csv)", type="csv", help="Contains global parameters like default holding and ordering costs.")


st.sidebar.markdown("---")
st.sidebar.header("⬇️ Generate & Download Data Templates")
with st.sidebar.expander("Click to generate templates"):
    if st.button("Generate Sales Template"):
        temp_sales_df = generate_sales_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS)
        st.download_button(
            label="Download sales_data_template.csv",
            data=temp_sales_df.to_csv(index=False).encode('utf-8'),
            file_name="sales_data_template.csv",
            mime="text/csv"
        )

    # Need BOM first to generate inventory and lead times with components
    temp_skus_for_templates = [f"SKU_{i:03d}" for i in range(1, DEFAULT_NUM_SKUS + 1)]
    temp_bom_df_for_templates = generate_bom_data(temp_skus_for_templates, DEFAULT_NUM_COMPONENTS_PER_SKU)
    
    if st.button("Generate BOM Template"):
        st.download_button(
            label="Download bom_data_template.csv",
            data=temp_bom_df_for_templates.to_csv(index=False).encode('utf-8'),
            file_name="bom_data_template.csv",
            mime="text/csv"
        )

    if st.button("Generate Inventory Template"):
        temp_sales_df_for_inv = generate_sales_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS)
        temp_inventory_df = generate_inventory_data(temp_sales_df_for_inv, temp_bom_df_for_templates, DEFAULT_START_DATE)
        st.download_button(
            label="Download inventory_data_template.csv",
            data=temp_inventory_df.to_csv(index=False).encode('utf-8'),
            file_name="inventory_data_template.csv",
            mime="text/csv"
        )

    if st.button("Generate Promotions Template"):
        temp_sales_df_for_promo = generate_sales_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS)
        temp_promotions_df = generate_promotion_data(temp_sales_df_for_promo, DEFAULT_PROMOTION_FREQUENCY_DAYS, DEFAULT_SALES_CHANNELS)
        st.download_button(
            label="Download promotions_data_template.csv",
            data=temp_promotions_df.to_csv(index=False).encode('utf-8'),
            file_name="promotions_data_template.csv",
            mime="text/csv"
        )

    if st.button("Generate External Factors Template"):
        temp_external_factors_df = generate_external_factors_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
        st.download_button(
            label="Download external_factors_data_template.csv",
            data=temp_external_factors_df.to_csv(index=False).encode('utf-8'),
            file_name="external_factors_data_template.csv",
            mime="text/csv"
        )

    if st.button("Generate Lead Times Template"):
        temp_sales_df_for_lt = generate_sales_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS)
        temp_unique_skus_for_lt = temp_sales_df_for_lt['SKU_ID'].unique()
        temp_lead_times_df = generate_lead_times_data(temp_unique_skus_for_lt, temp_bom_df_for_templates, DEFAULT_MAX_LEAD_TIME_DAYS, DEFAULT_MAX_SKU_SHELF_LIFE_DAYS)
        st.download_button(
            label="Download lead_times_data_template.csv",
            data=temp_lead_times_df.to_csv(index=False).encode('utf-8'),
            file_name="lead_times_data_template.csv",
            mime="text/csv"
        )

    if st.button("Generate Actual Orders Template"):
        temp_orders_df = generate_actual_orders_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS)
        st.download_button(
            label="Download actual_orders_template.csv",
            data=temp_orders_df.to_csv(index=False).encode('utf-8'),
            file_name="actual_orders_template.csv",
            mime="text/csv"
        )

    if st.button("Generate Actual Shipments Template"):
        temp_orders_df_for_ship = generate_actual_orders_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS)
        temp_shipments_df = generate_actual_shipments_data(temp_orders_df_for_ship)
        st.download_button(
            label="Download actual_shipments_template.csv",
            data=temp_shipments_df.to_csv(index=False).encode('utf-8'),
            file_name="actual_shipments_template.csv",
            mime="text/csv"
        )
    
    if st.button("Generate Global Config Template"):
        temp_global_config_df = generate_global_config_data()
        st.download_button(
            label="Download global_config_template.csv",
            data=temp_global_config_df.to_csv(index=False).encode('utf-8'),
            file_name="global_config_template.csv",
            mime="text/csv"
        )


st.sidebar.markdown("---")
st.sidebar.header("🚀 Run Forecast")

data_source_choice = st.sidebar.radio(
    "Choose Data Source",
    ("Use Uploaded Data", "Use Sample Data"),
    key="data_source_radio"
)

run_forecast_button = st.sidebar.button("Run Forecast", key="run_forecast_button")

st.sidebar.markdown("---")
st.sidebar.header("📊 Display Options")
forecast_roll_up_choice = st.sidebar.radio(
    "Forecast Roll-up",
    ("Day", "Week", "Month"),
    help="Aggregate forecasted demand by Day, Week, or Month."
)


if run_forecast_button:
    sales_df_local = pd.DataFrame()
    inventory_df_raw_local = pd.DataFrame()
    promotions_df_local = pd.DataFrame()
    external_factors_df_local = pd.DataFrame()
    lead_times_df_local = pd.DataFrame()
    bom_df_local = pd.DataFrame()
    actual_orders_df_local = pd.DataFrame()
    actual_shipments_df_local = pd.DataFrame()
    global_config_df_local = pd.DataFrame()

    holding_cost_per_unit_per_day_param_local = DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY
    ordering_cost_per_order_param_local = DEFAULT_ORDERING_COST_PER_ORDER


    if data_source_choice == "Use Sample Data":
        st.info("Running forecast on **Sample Data**.")
        sales_df_local = generate_sales_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS)
        unique_skus = sales_df_local['SKU_ID'].unique()
        
        # Generate BOM first, then use it for inventory and lead times
        bom_df_local = generate_bom_data(unique_skus, DEFAULT_NUM_COMPONENTS_PER_SKU)
        inventory_df_raw_local = generate_inventory_data(sales_df_local, bom_df_local, DEFAULT_START_DATE)
        lead_times_df_local = generate_lead_times_data(unique_skus, bom_df_local, DEFAULT_MAX_LEAD_TIME_DAYS, DEFAULT_MAX_SKU_SHELF_LIFE_DAYS)

        promotions_df_local = generate_promotion_data(sales_df_local, DEFAULT_PROMOTION_FREQUENCY_DAYS, DEFAULT_SALES_CHANNELS)
        external_factors_df_local = generate_external_factors_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
        actual_orders_df_local = generate_actual_orders_data(DEFAULT_NUM_SKUS, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_SALES_CHANNELS)
        actual_shipments_df_local = generate_actual_shipments_data(actual_orders_df_local)
        global_config_df_local = generate_global_config_data() # Generate sample global config

    elif data_source_choice == "Use Uploaded Data":
        st.info("Running forecast on **Uploaded Data**.")
        required_files_uploaded = True
        required_files = {
            "Sales Data": uploaded_sales_file,
            "Inventory Data": uploaded_inventory_file,
            "Lead Times Data": uploaded_lead_times_file,
            "BOM Data": uploaded_bom_file
        }

        for file_name, file_obj in required_files.items():
            if file_obj is None:
                st.error(f"Please upload the '{file_name}' CSV file to proceed with 'Run Forecast'.")
                required_files_uploaded = False
                break
        
        if not required_files_uploaded:
            st.stop()

        try:
            sales_df_local = pd.read_csv(uploaded_sales_file, parse_dates=['Date'])
            inventory_df_raw_local = pd.read_csv(uploaded_inventory_file, parse_dates=['Date'])
            lead_times_df_local = pd.read_csv(uploaded_lead_times_file)
            bom_df_local = pd.read_csv(uploaded_bom_file)

            # Ensure Sales_Channel column exists in uploaded sales_df, if not, add a default
            if 'Sales_Channel' not in sales_df_local.columns:
                st.warning("No 'Sales_Channel' column found in Sales Data. Assuming 'Default' channel for all sales.")
                sales_df_local['Sales_Channel'] = 'Default'
            
            # Ensure Item_Type column exists in uploaded inventory_df_raw_local
            if 'Item_Type' not in inventory_df_raw_local.columns:
                st.error("Missing 'Item_Type' column in Inventory Data. Please ensure it has 'Finished_Good', 'Raw_Material', or 'Sub_Assembly'.")
                st.stop()
            # Ensure Item_Type column exists in uploaded lead_times_df_local
            if 'Item_Type' not in lead_times_df_local.columns:
                st.error("Missing 'Item_Type' column in Lead Times Data. Please ensure it has 'Finished_Good', 'Raw_Material', or 'Sub_Assembly'.")
                st.stop()
            # Ensure Component_Type column exists in uploaded bom_df_local
            if 'Component_Type' not in bom_df_local.columns:
                st.error("Missing 'Component_Type' column in BOM Data. Please ensure it has 'Raw_Material' or 'Sub_Assembly'.")
                st.stop()


            if uploaded_promotions_file is not None:
                promotions_df_local = pd.read_csv(uploaded_promotions_file, parse_dates=['Date'])
                if 'Sales_Channel' not in promotions_df_local.columns:
                    st.warning("No 'Sales_Channel' column found in Promotions Data. Assuming 'Default' channel for all promotions.")
                    promotions_df_local['Sales_Channel'] = 'Default'
            if uploaded_external_factors_file is not None:
                external_factors_df_local = pd.read_csv(uploaded_external_factors_file, parse_dates=['Date'])
                external_factors_df_local['Date'] = pd.to_datetime(external_factors_df_local['Date'])
            if uploaded_actual_orders_file is not None:
                actual_orders_df_local = pd.read_csv(uploaded_actual_orders_file, parse_dates=['Date'])
                if 'Sales_Channel' not in actual_orders_df_local.columns:
                    st.warning("No 'Sales_Channel' column found in Actual Orders Data. Assuming 'Default' channel for all orders.")
                    actual_orders_df_local['Sales_Channel'] = 'Default'
            if uploaded_actual_shipments_file is not None:
                actual_shipments_df_local = pd.read_csv(uploaded_actual_shipments_file, parse_dates=['Date'])
                if 'Sales_Channel' not in actual_shipments_df_local.columns:
                    st.warning("No 'Sales_Channel' column found in Actual Shipments Data. Assuming 'Default' channel for all shipments.")
                    actual_shipments_df_local['Sales_Channel'] = 'Default'
            if uploaded_global_config_file is not None:
                global_config_df_local = pd.read_csv(uploaded_global_config_file)

        except Exception as e:
            st.error(f"An error occurred while reading uploaded files: {e}. Please check your file formats and required columns.")
            st.stop()
    
    # Extract global cost parameters from global_config_df or use defaults
    if not global_config_df_local.empty:
        try:
            holding_cost_row = global_config_df_local[global_config_df_local['Parameter'] == 'Holding_Cost_Per_Unit_Per_Day']
            if not holding_cost_row.empty: 
                holding_cost_per_unit_per_day_param_local = float(holding_cost_row['Value'].iloc[0])
            else:
                st.warning(f"Parameter 'Holding_Cost_Per_Unit_Per_Day' not found in global_config.csv. Using default: ${DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY:.2f}")

            ordering_cost_row = global_config_df_local[global_config_df_local['Parameter'] == 'Ordering_Cost_Per_Order']
            if not ordering_cost_row.empty:
                ordering_cost_per_order_param_local = float(ordering_cost_row['Value'].iloc[0])
            else:
                st.warning(f"Parameter 'Ordering_Cost_Per_Order' not found in global_config.csv. Using default: ${DEFAULT_ORDERING_COST_PER_ORDER:.2f}")

        except Exception as e:
            st.error(f"Error parsing global config parameters: {e}. Using default cost parameters.")
            holding_cost_per_unit_per_day_param_local = DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY
            ordering_cost_per_order_param_local = DEFAULT_ORDERING_COST_PER_ORDER
    else:
        st.info(f"global_config.csv not uploaded. Using default holding cost (${DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY:.2f}/unit/day) and ordering cost (${DEFAULT_ORDERING_COST_PER_ORDER:.2f}/order).")


    with st.spinner("Processing data, training models, and forecasting demand..."):
        try:
            # Determine all unique sales channels present in the sales data
            all_sales_channels_in_data_local = sales_df_local['Sales_Channel'].unique().tolist()

            # Prepare data for SKU forecasting (only finished goods sales and inventory)
            processed_data_df_local = prepare_data_for_forecasting(sales_df_local, inventory_df_raw_local, promotions_df_local, external_factors_df_local)

            if processed_data_df_local is not None and not processed_data_df_local.empty:
                trained_models, evaluation_metrics_local = train_forecast_model(processed_data_df_local, model_choice)

                if trained_models:
                    st.success(f"Demand forecasting models trained using **{model_choice}**!")
                    
                    forecasted_demand_by_sku_channel = predict_demand(trained_models, processed_data_df_local, forecast_horizon_days_input, external_factors_df_local, all_sales_channels_in_data_local)

                    all_forecasts_df_local = pd.concat(forecasted_demand_by_sku_channel.values(), ignore_index=True)

                    # Calculate SKU-level Auto-Indent Recommendations
                    sku_indent_recommendations_df_local = calculate_auto_indent(
                        {k: all_forecasts_df_local[(all_forecasts_df_local['SKU_ID'] == k[0]) & (all_forecasts_df_local['Sales_Channel'] == k[1])] for k in evaluation_metrics_local.keys()},
                        inventory_df_raw_local, lead_times_df_local,
                        evaluation_metrics_local, service_level_input, reorder_point_days_input,
                        safety_stock_method_choice, fixed_safety_stock_days_input, safety_stock_cap_factor_input,
                        holding_cost_per_unit_per_day_param_local, ordering_cost_per_order_param_local
                    )

                    # Calculate Component-level Auto-Indent Recommendations
                    component_indent_recommendations_df_local = calculate_component_indent(
                        sku_indent_recommendations_df_local, bom_df_local, inventory_df_raw_local, lead_times_df_local,
                        service_level_input, reorder_point_days_input, safety_stock_method_choice, fixed_safety_stock_days_input,
                        safety_stock_cap_factor_input, holding_cost_per_unit_per_day_param_local, ordering_cost_per_order_param_local
                    )


                    # Calculate overall KPI dataframes (SKU-level)
                    all_inventory_kpis_df_local = calculate_inventory_kpis(sales_df_local, inventory_df_raw_local)
                    all_stockout_rates_df_local = calculate_stockout_rate_over_time(actual_orders_df_local, actual_shipments_df_local)

                    # Store results in session state
                    st.session_state.forecast_run = True
                    st.session_state.processed_data_df = processed_data_df_local
                    st.session_state.all_forecasts_df = all_forecasts_df_local
                    st.session_state.sku_indent_recommendations_df = sku_indent_recommendations_df_local
                    st.session_state.component_indent_recommendations_df = component_indent_recommendations_df_local # Store component indent
                    st.session_state.all_inventory_kpis_df = all_inventory_kpis_df_local
                    st.session_state.all_stockout_rates_df = all_stockout_rates_df_local
                    st.session_state.sales_df = sales_df_local
                    st.session_state.inventory_df_raw = inventory_df_raw_local # Store raw inventory
                    st.session_state.actual_orders_df = actual_orders_df_local
                    st.session_state.actual_shipments_df = actual_shipments_df_local
                    st.session_state.all_sales_channels_in_data = all_sales_channels_in_data_local
                    st.session_state.evaluation_metrics = evaluation_metrics_local
                    st.session_state.lead_times_df = lead_times_df_local
                    st.session_state.bom_df = bom_df_local
                    st.session_state.holding_cost_per_unit_per_day_param = holding_cost_per_unit_per_day_param_local
                    st.session_state.ordering_cost_per_order_param = ordering_cost_per_order_param_local

                else:
                    st.error("No models were trained. Cannot proceed with forecasting and indent logic.")
                    st.session_state.forecast_run = False
            else:
                st.error("Data processing failed or resulted in an empty DataFrame. Please check your uploaded data format.")
                st.session_state.forecast_run = False

        except Exception as e:
            st.error(f"An unexpected error occurred during model execution: {e}")
            st.session_state.forecast_run = False

# Display results only if forecast has been run successfully
if st.session_state.forecast_run:
    st.markdown("---")
    
    # Use st.tabs for better user experience
    tab_flow, tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Detailed Process Flow", # New Tab for Process Flow
        "Data & Parameters Explanation",
        "Model Evaluation",
        "Future Demand Forecasts",
        "SKU Auto-Indent",
        "Component Auto-Indent",
        "Fill Rates",
        "Supply Chain KPIs & Visualizations"
    ])

    with tab_flow: # New: Detailed Process Flow Tab
        st.header("📚 QuantumFlow: Detailed Process Flow")
        st.markdown("This section outlines the end-to-end workflow of the QuantumFlow application, from data ingestion to generating order recommendations and performance monitoring.")

        # Create a Graphviz graph object
        graph = graphviz.Digraph(comment='QuantumFlow Process', graph_attr={'rankdir': 'TB', 'splines': 'spline', 'nodesep': '0.8', 'ranksep': '1.2'}, edge_attr={'arrowhead': 'normal', 'arrowsize': '0.8'})

        # Define node styles
        node_style_general = 'shape=box, style="filled,rounded", fillcolor="#e3f2fd", color="#2196f3", fontname="Inter", fontsize=12'
        node_style_decision = 'shape=diamond, style="filled,rounded", fillcolor="#fff3e0", color="#ff9800", fontname="Inter", fontsize=12'
        node_style_start_end = 'shape=circle, style="filled", fillcolor="#e0f7fa", color="#00bcd4", fontname="Inter", fontsize=12'
        subgraph_style = 'style=dashed, color="#cbd5e1", labeljust=l, labelloc=t, fontname="Inter", fontsize=14, fontcolor="#475569"'

        # Start Node
        graph.node('Start', 'Start', **{'shape': 'circle', 'style': 'filled', 'fillcolor': '#e0f7fa', 'color': '#00bcd4', 'fontname': 'Inter', 'fontsize': '12'})

        # Data Management Subgraph
        with graph.subgraph(name='cluster_0') as c0:
            c0.attr(label='Data Management', **{'style': 'dashed', 'color': '#cbd5e1', 'labeljust': 'l', 'labelloc': 't', 'fontname': 'Inter', 'fontsize': '14', 'fontcolor': '#475569'})
            c0.node('B', '1. Data Ingestion & Preparation', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c0.node('B1', 'Collect Raw Data', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c0.node('B2', 'Merge & Clean Data', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c0.node('B3', 'Feature Engineering', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})

            c0.edge('B', 'B1', label='Raw Data Files', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
            c0.edge('B', 'B2', label='Cleaned Data', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
            c0.edge('B', 'B3') # No explicit label for this one in original SVG
            c0.edge('B1', 'B2')
            c0.edge('B2', 'B3')


        # Demand Forecasting Subgraph
        with graph.subgraph(name='cluster_1') as c1:
            c1.attr(label='Demand Forecasting', **{'style': 'dashed', 'color': '#cbd5e1', 'labeljust': 'l', 'labelloc': 't', 'fontname': 'Inter', 'fontsize': '14', 'fontcolor': '#475569'})
            c1.node('C1', 'Select ML Model', **{'shape': 'diamond', 'style': 'filled,rounded', 'fillcolor': '#fff3e0', 'color': '#ff9800', 'fontname': 'Inter', 'fontsize': '12'})
            c1.node('C2', 'Train Model', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c1.node('C3', 'Predict Future Demand', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})

            c1.edge('C1', 'C2', label='Model Choice', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
            c1.edge('C2', 'C3', label='Evaluation Metrics (RMSE, MAE)', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')


        # Inventory Optimization Subgraph
        with graph.subgraph(name='cluster_2') as c2:
            c2.attr(label='Inventory Optimization', **{'style': 'dashed', 'color': '#cbd5e1', 'labeljust': 'l', 'labelloc': 't', 'fontname': 'Inter', 'fontsize': '14', 'fontcolor': '#475569'})
            c2.node('D1', 'Aggregate Forecasts to SKU Level', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c2.node('D2', 'Calculate Safety Stock', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c2.node('D3', 'Determine Reorder Point', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c2.node('D4', 'Calculate SKU Order Quantity', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})

            c2.edge('D1', 'D2', label='Aggregated Forecasts', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
            c2.edge('D1', 'D3', label='Service Level, RMSE', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
            c2.edge('D2', 'D4')
            c2.edge('D3', 'D4')


        # BOM Subgraph
        with graph.subgraph(name='cluster_3') as c3:
            c3.attr(label='Bill of Materials (BOM)', **{'style': 'dashed', 'color': '#cbd5e1', 'labeljust': 'l', 'labelloc': 't', 'fontname': 'Inter', 'fontsize': '14', 'fontcolor': '#475569'})
            c3.node('E1', 'Explode BOM for SKU Orders', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c3.node('E2', 'Calculate Component Requirements', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c3.node('E3', 'Generate Component Procurement Plan', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})

            c3.edge('E1', 'E2', label='Component Requirements', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
            c3.edge('E1', 'E3', label='Component Lead Times, Shelf Life', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
            c3.edge('E2', 'E3')


        # Execution & Monitoring Subgraph
        with graph.subgraph(name='cluster_4') as c4:
            c4.attr(label='Execution & Monitoring', **{'style': 'dashed', 'color': '#cbd5e1', 'labeljust': 'l', 'labelloc': 't', 'fontname': 'Inter', 'fontsize': '14', 'fontcolor': '#475569'})
            c4.node('F1', 'Generate Procurement/Production Orders', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c4.node('F2', 'Execute Orders', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c4.node('F3', 'Monitor Inventory & Shipments', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c4.node('F4', 'Calculate Supply Chain KPIs', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})
            c4.node('F5', 'Visualize Performance & Insights', **{'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#e3f2fd', 'color': '#2196f3', 'fontname': 'Inter', 'fontsize': '12'})

            c4.edge('F1', 'F2', label='Actual Orders, Shipments', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
            c4.edge('F1', 'F3', label='KPIs & Insights', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
            c4.edge('F2', 'F3')
            c4.edge('F3', 'F4')
            c4.edge('F4', 'F5')


        # Global Edges
        graph.edge('Start', 'B')
        graph.edge('B3', 'C1', label='Prepared Features', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
        graph.edge('C3', 'D1', label='Forecasted Demand', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
        graph.edge('D4', 'E1', label='SKU Order Quantities', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
        graph.edge('E3', 'F1', label='Component Orders', labelfontsize='10', labelfontname='Inter', fontcolor='#64748b')
        graph.edge('F5', 'End')
        graph.node('End', 'End', **{'shape': 'circle', 'style': 'filled', 'fillcolor': '#e0f7fa', 'color': '#00bcd4', 'fontname': 'Inter', 'fontsize': '12'})

        st.graphviz_chart(graph)

        st.markdown("---")
        st.subheader("1. Data Ingestion & Preparation")
        st.markdown("Raw historical data is collected from various sources and transformed into a structured format suitable for machine learning.")
        st.markdown("""
        **Steps:**
        - **Data Generation/Upload:** Historical Sales, Inventory Levels (for SKUs and Components), Promotions, External Factors (Economic Index, Holidays, Temperature, Competitor Activity), Supplier Lead Times (for SKUs and Components), Bill of Materials (BOM), Actual Orders, Actual Shipments, Global Configuration.
        - **Data Merging:** All relevant datasets are merged based on common keys like 'Date', 'SKU_ID' (or 'Item_ID' for inventory/lead times), and 'Sales_Channel' to create a comprehensive dataset.
        - **Feature Engineering:** New features are derived from existing data to capture trends, seasonality, and other patterns.

        **Key Calculations/Transformations:**
        - `Date` parsing to datetime objects.
        - Creation of time-based features: `Year`, `Month`, `Day`, `DayOfWeek`, `DayOfYear`, `WeekOfYear`, `Is_Weekend`, `Is_Holiday`.
        - Lagged sales quantities: `Sales_Quantity_Lag_1` (previous day's sales), `Sales_Quantity_Lag_7` (sales from 7 days prior). Calculated per SKU and Sales Channel.
        - One-hot encoding for categorical features: `Promotion_Type` (e.g., `Promo_Type_Discount`), `Sales_Channel` (e.g., `Channel_Amazon`).
        - Handling of missing values (`fillna(0)`) for numerical features introduced by merges (e.g., promotions, external factors).
        """)

        st.markdown("---")
        st.subheader("2. Demand Forecasting Model Training")
        st.markdown("Machine learning models are trained on the prepared historical data to learn demand patterns for Finished Goods (SKUs).")
        st.markdown("""
        **Steps:**
        - **Model Selection:** User chooses between XGBoost Regressor, Random Forest Regressor, Prophet (Time Series), ARIMA (Time Series), or 'Auto Select Best Model (All)'.
        - **Data Splitting:** Historical data for each unique SKU-Channel combination is split into training (80%) and testing (20%) sets using a time-based split.
        - **Model Training:** The selected model is trained on the training data for each SKU-Channel pair.
        - **Model Evaluation:** Trained models are evaluated on the test set to assess their accuracy.

        **Key Calculations/Metrics:**
        - **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual sales.
        - **Root Mean Squared Error (RMSE):** Square root of the average of squared differences between predicted and actual sales. This metric is crucial for Safety Stock calculation.
        - For 'Auto Select Best Model', the model with the lowest RMSE for each SKU-Channel combination is chosen.
        """)

        st.markdown("---")
        st.subheader("3. Future Demand Prediction")
        st.markdown("Trained models are used to forecast future sales quantities for Finished Goods (SKUs) for a specified horizon.")
        st.markdown("""
        **Steps:**
        - **Future Data Generation:** A dataframe of future dates is created for the specified forecast horizon.
        - **Feature Projection:** Features (time-based, lagged sales, external factors, promotion flags, channel dummies) are projected into the future for each forecast date. Lagged sales are iteratively updated with the new predictions.
        - **Prediction:** Each SKU-Channel's trained model predicts the `Forecasted_Quantity` for each day in the forecast horizon.

        **Key Calculations:**
        - Iterative update of lagged sales: `Sales_Quantity_Lag_1` for day N becomes the predicted quantity from day N-1; `Sales_Quantity_Lag_7` from day N-7.
        - Handling of future external factors (if provided) or using historical averages/defaults.
        - Ensuring predicted quantities are non-negative (`max(0, predicted_value)`).
        """)

        st.markdown("---")
        st.subheader("4. SKU-Level Auto-Indent Recommendations")
        st.markdown("Based on aggregated SKU-level forecasts, current SKU stock, and SKU lead times, optimal Finished Good order quantities are determined.")
        st.markdown("""
        **Steps:**
        - **Forecast Aggregation:** Channel-specific forecasts are summed to get a single SKU-level daily forecast.
        - **Safety Stock Calculation:** Determines buffer stock to cover demand variability during lead time.
        - **Reorder Point Calculation:** The inventory level that triggers a new order.
        - **Target Stock Level:** The desired inventory level after an order arrives.
        - **Order Quantity Determination:** Calculates how much to order to replenish stock when the reorder point is hit.
        - **Cost Estimation:** Calculates estimated holding and ordering costs.

        **Key Calculations:**
        - **Safety Stock (King's Method):** Z-score multiplied by RMSE multiplied by the square root of Lead Time (Days).
        - **Safety Stock (Fixed Days):** Average Daily Forecast multiplied by Fixed Safety Stock Days.
        - **Safety Stock Capping:** Safety stock is capped at a multiple of forecasted demand during the item's shelf life to prevent excessive inventory holding and waste for perishable goods.
        - **Forecast During Lead Time:** Sum of forecasted quantities over the item's lead time.
        - **Target Stock Level:** Forecast During Lead Time plus Capped Safety Stock.
        - **Reorder Point:** Forecasted Demand for Reorder Point Buffer Days plus Capped Safety Stock.
        - **Order Quantity:** Maximum of 0 or (Target Stock Level minus Current Stock). This quantity is then adjusted to meet Min_Order_Quantity and Order_Multiple constraints.
        - **Estimated Holding Cost:** Target Stock Level multiplied by Holding Cost Per Unit Per Day multiplied by 30 Days.
        - **Estimated Ordering Cost:** Ordering Cost Per Order (if an order is placed).
        """)

        st.markdown("---")
        st.subheader("5. BOM-Level Component Requirements & Optimization")
        st.markdown("SKU order recommendations are translated into specific requirements for raw materials and sub-assemblies, and then optimized based on component-level inventory and properties.")
        st.markdown("""
        **Steps:**
        - **Raw Component Requirement Calculation (BOM Explosion):** For each SKU order, the Bill of Materials is used to determine the raw quantity of each component needed.
        - **Component Inventory Check:** Current stock levels of each component are checked.
        - **Component Safety Stock Calculation:** Similar to SKUs, safety stock is calculated for components based on derived demand and component lead times. Given that component demand is derived (not directly forecasted with RMSE), a heuristic approach is used for variability in the 'King's Method' for components.
        - **Component Reorder Point:** Determined based on derived demand and safety stock.
        - **Component Order Quantity:** Calculates the quantity of each component to order, considering current stock, reorder point, minimum order quantities, and order multiples.
        - **Cost Estimation:** Calculates estimated holding and ordering costs for components.

        **Key Calculations:**
        - **Derived Component Demand:** SKU Order Quantity multiplied by Component Quantity Required (from BOM).
        - **Component Safety Stock (King's Method - Heuristic):** Z-score multiplied by Average Daily Derived Demand multiplied by the square root of Component Lead Time (Days) multiplied by a Heuristic Variability Factor (e.g., 0.5).
        - **Component Safety Stock (Fixed Days):** Average Daily Derived Demand multiplied by Fixed Safety Stock Days.
        - **Component Safety Stock Capping:** Applies the same shelf-life based capping logic as for SKUs.
        - **Component Derived Demand During Lead Time:** Average Daily Derived Demand multiplied by Component Lead Time (Days).
        - **Component Target Stock Level:** Component Derived Demand During Lead Time plus Capped Component Safety Stock.
        - **Component Reorder Point:** Average Daily Derived Demand multiplied by Reorder Point Buffer Days plus Capped Component Safety Stock.
        - **Component Order Quantity:** Maximum of 0 or (Component Target Stock Level minus Current Component Stock). Adjusted for Min_Order_Quantity and Order_Multiple.
        - **Estimated Holding Cost:** Component Target Stock Level multiplied by Holding Cost Per Unit Per Day multiplied by 30 Days.
        - **Estimated Ordering Cost:** Ordering Cost Per Order (if a component order is placed).
        """)

        st.markdown("---")
        st.subheader("6. Execution & Monitoring")
        st.markdown("The final stage involves generating actionable orders, tracking their fulfillment, and continuously monitoring supply chain performance through key performance indicators (KPIs).")
        st.markdown("""
        **Steps:**
        - **Generate Orders:** Procurement orders for raw materials/components and production orders for finished goods are generated based on the auto-indent recommendations.
        - **Execute Orders:** Orders are placed with suppliers or production is initiated.
        - **Monitor Inventory & Shipments:** Real-time tracking of stock levels and customer order fulfillment.
        - **Calculate Supply Chain KPIs:** Key metrics are computed to assess the efficiency and effectiveness of the supply chain.
        - **Visualize Performance & Insights:** KPIs and other insights are visualized to provide actionable intelligence to decision-makers.

        **Key Calculations/Metrics:**
        - **Fill Rates:**
            - **Order Fill Rate:** Percentage of orders fulfilled completely.
            - **Line Fill Rate:** Percentage of order lines fulfilled completely.
            - **Unit Fill Rate:** Percentage of units ordered that were shipped.
        - **Stockout Rate:** (Total Unfulfilled Quantity / Total Ordered Quantity) * 100%
        - **Inventory Turnover:** (Cost of Goods Sold / Average Inventory). (Approximated here as (Sales Quantity / Average Inventory) * Annualization Factor).
        - **Days of Inventory Outstanding (DIO):** (Average Inventory / Cost of Goods Sold per day). (Approximated here as (Average Inventory / Sales Quantity per day)).
        """)


    with tab0: # Data & Parameters Explanation Tab
        st.header("📚 Data Tables and Parameter Inputs Explanation")
        st.markdown("This section provides details on the various data tables required and the configurable parameters used in the QuantumFlow application.")

        st.subheader("Data Tables:")
        st.markdown("The application relies on several CSV files for historical data and configuration. Templates can be generated from the sidebar.")
        
        st.markdown("---")
        st.markdown("#### Sales Data (`sales_data.csv`)")
        st.markdown("Contains historical sales records for each SKU across different channels. This is the primary data source for demand forecasting.")
        st.markdown("""
        - `Date`: Date of the sales transaction (YYYY-MM-DD).
        - `SKU_ID`: Unique identifier for the Stock Keeping Unit.
        - `Sales_Quantity`: Number of units sold.
        - `Price`: Price per unit at the time of sale.
        - `Customer_Segment`: Segment of the customer (e.g., Retail, Wholesale, Online).
        - `Sales_Channel`: Channel through which the sale occurred (e.g., Distributor Network, Amazon, Own Website).
        """)

        st.markdown("---")
        st.markdown("#### Inventory Data (`inventory_data.csv`)")
        st.markdown("Tracks the historical stock levels for both Finished Goods and Components. **Crucial for both SKU and Component inventory optimization.**")
        st.markdown("""
        - `Date`: Date of the inventory record (YYYY-MM-DD).
        - `Item_ID`: Unique identifier for the item (SKU or Component).
        - `Item_Type`: Type of item (`Finished_Good`, `Raw_Material`, `Sub_Assembly`).
        - `Current_Stock`: Number of units in stock on that date.
        """)

        st.markdown("---")
        st.markdown("#### Promotions Data (`promotions_data.csv`) - *Optional*")
        st.markdown("Records past promotional activities that might have influenced sales.")
        st.markdown("""
        - `Date`: Date the promotion was active (YYYY-MM-DD).
        - `SKU_ID`: SKU affected by the promotion.
        - `Promotion_Type`: Type of promotion (e.g., Discount, BOGO, Bundle).
        - `Discount_Percentage`: Discount applied (e.g., 0.15 for 15%).
        - `Sales_Channel`: Channel where the promotion was active (e.g., Amazon, All).
        """)

        st.markdown("---")
        st.markdown("#### External Factors Data (`external_factors_data.csv`) - *Optional*")
        st.markdown("Includes external variables that could impact demand.")
        st.markdown("""
        - `Date`: Date of the external factor record (YYYY-MM-DD).
        - `Economic_Index`: A numerical index representing economic conditions.
        - `Holiday_Flag`: Binary flag (1 if holiday, 0 otherwise).
        - `Temperature_Celsius`: Average temperature for the day.
        - `Competitor_Activity_Index`: A numerical index representing competitor activity.
        """)

        st.markdown("---")
        st.markdown("#### Lead Times Data (`lead_times_data.csv`)")
        st.markdown("Defines supplier lead times and ordering constraints for both finished goods (SKUs) and components. **Crucial for both SKU and Component inventory optimization.**")
        st.markdown("""
        - `Item_ID`: Unique identifier for the item (SKU or Component).
        - `Item_Type`: Type of item (`Finished_Good`, `Raw_Material`, `Sub_Assembly`).
        - `Supplier_ID`: Identifier for the supplier.
        - `Lead_Time_Days`: Number of days from order placement to receipt.
        - `Shelf_Life_Days`: Shelf life of the item in days (for perishables/expirables).
        - `Min_Order_Quantity`: Minimum quantity that can be ordered from the supplier.
        - `Order_Multiple`: Orders must be in multiples of this quantity.
        """)

        st.markdown("---")
        st.markdown("#### Bill of Materials (BOM) Data (`bom_data.csv`)")
        st.markdown("Defines the components required to produce each finished good (SKU). **Essential for component demand derivation.**")
        st.markdown("""
        - `Parent_SKU_ID`: The finished good SKU that requires components.
        - `Component_ID`: Unique identifier for the component.
        - `Quantity_Required`: Number of units of the component needed for one unit of the Parent_SKU.
        - `Component_Type`: Type of component (`Raw_Material`, `Sub_Assembly`).
        - `Shelf_Life_Days`: Shelf life of the component in days (for perishables/expirables).
        """)

        st.markdown("---")
        st.markdown("#### Actual Orders Data (`actual_orders.csv`) - *Optional*")
        st.markdown("Records actual customer orders placed.")
        st.markdown("""
        - `Date`: Date the order was placed (YYYY-MM-DD).
        - `Order_ID`: Unique identifier for the order.
        - `SKU_ID`: SKU ordered.
        - `Ordered_Quantity`: Quantity of the SKU ordered.
        - `Sales_Channel`: Channel through which the order was placed.
        """)

        st.markdown("---")
        st.markdown("#### Actual Shipments Data (`actual_shipments.csv`) - *Optional*")
        st.markdown("Records actual quantities shipped against customer orders.")
        st.markdown("""
        - `Date`: Date the shipment occurred (YYYY-MM-DD).
        - `Order_ID`: Corresponding order ID.
        - `SKU_ID`: SKU shipped.
        - `Shipped_Quantity`: Quantity of the SKU shipped.
        - `Sales_Channel`: Channel associated with the shipment.
        """)
        
        st.markdown("---")
        st.markdown("#### Global Config Data (`global_config.csv`) - *Optional*")
        st.markdown("Contains global cost parameters for inventory optimization.")
        st.markdown("""
        - `Parameter`: Name of the configuration parameter (e.g., `Holding_Cost_Per_Unit_Per_Day`, `Ordering_Cost_Per_Order`).
        - `Value`: The numerical value for the parameter.
        """)

        st.subheader("Parameter Inputs:")
        st.markdown("These parameters are configured in the sidebar and influence the forecasting and inventory optimization logic.")

        st.markdown("---")
        st.markdown("#### Forecast Horizon (Days)")
        st.markdown("Defines how many days into the future the demand forecasting model will predict sales. A longer horizon provides more foresight but might be less accurate.")

        st.markdown("---")
        st.markdown("#### Service Level (%)")
        st.markdown("The desired probability of not stocking out during the lead time. A higher service level (e.g., 99%) implies a higher safety stock to meet more demand fluctuations, while a lower one (e.g., 90%) accepts more risk of stockouts. **Applies to both SKUs and Components.**")

        st.markdown("---")
        st.markdown("#### Reorder Point Buffer (Days)")
        st.markdown("An additional buffer added to the reorder point, expressed in days of forecasted demand. This provides an extra cushion before placing an order, useful for managing unexpected delays or demand spikes. Set to 0 for no buffer. **Applies to both SKUs and Components.**")

        st.markdown("---")
        st.markdown("#### Select Safety Stock Calculation Method")
        st.markdown("Determines the methodology used to calculate safety stock: **Applies to both SKUs and Components.**")
        st.markdown("""
        - **King's Method (Statistical):** Uses the forecast error (RMSE for SKUs, or a heuristic for components) and the desired service level to statistically determine the safety stock needed to cover demand variability during lead time.
        - **Fixed Days of Demand:** Calculates safety stock as a fixed number of days' worth of average forecasted demand.
        """)

        st.markdown("---")
        st.markdown("#### Fixed Safety Stock Days (visible if 'Fixed Days of Demand' is selected)")
        st.markdown("When using the 'Fixed Days of Demand' method, this parameter specifies how many days of average daily demand should be held as safety stock. **Applies to both SKUs and Components.**")

        st.markdown("---")
        st.markdown("#### Safety Stock Cap (x Shelf Life Demand)")
        st.markdown("This parameter caps the calculated safety stock at a multiple of the forecasted demand during an item's shelf life. It's particularly important for perishable or short-shelf-life items to prevent excessive inventory holding and waste. For example, a value of 1.5 means safety stock won't exceed 1.5 times the demand expected within the item's shelf life. **Applies to both SKUs and Components.**")

        st.markdown("---")
        st.markdown("#### Select ML Model")
        st.markdown("Allows selection of the machine learning model used for demand forecasting (for SKUs only):")
        st.markdown("""
        - **Auto Select Best Model (All):** The application will train all available models (XGBoost, Random Forest, Prophet, ARIMA) for each SKU-channel combination and automatically select the one with the lowest RMSE.
        - **XGBoost Regressor:** A powerful gradient boosting algorithm known for its performance and speed.
        - **Random Forest Regressor:** An ensemble learning method that builds multiple decision trees and averages their predictions.
        - **Prophet (Time Series):** A forecasting tool developed by Facebook, optimized for business forecasts with strong seasonality and trend components.
        - **ARIMA (Time Series):** AutoRegressive Integrated Moving Average, a classic statistical method for time series forecasting.
        """)

        st.markdown("---")
        st.markdown("#### Forecast Roll-up")
        st.markdown("Determines the granularity at which forecasted demand and KPIs are displayed in the visualizations and tables (Day, Week, or Month).")


    with tab1: # Model Evaluation tab
        st.header("📊 Model Evaluation Metrics (Per SKU-Channel Combination)")
        if st.session_state.evaluation_metrics:
            metrics_df = pd.DataFrame([
                {"SKU_Channel": f"{sku_id} ({channel})", **metrics} 
                for (sku_id, channel), metrics in st.session_state.evaluation_metrics.items()
            ])
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("No model evaluation metrics available.")

    with tab2: # Future Demand Forecasts tab
        st.header("📈 Future Demand Forecasts (Per SKU and Channel)")
        # Apply roll-up logic for detailed forecasts
        if forecast_roll_up_choice == "Day":
            st.dataframe(st.session_state.all_forecasts_df, use_container_width=True)
        elif forecast_roll_up_choice == "Week":
            all_forecasts_df_display = st.session_state.all_forecasts_df.copy()
            all_forecasts_df_display['YearWeek'] = all_forecasts_df_display['Date'].dt.to_period('W').astype(str)
            weekly_forecasts = all_forecasts_df_display.groupby(['SKU_ID', 'Sales_Channel', 'YearWeek'])['Forecasted_Quantity'].sum().reset_index()
            weekly_forecasts = weekly_forecasts.rename(columns={'YearWeek': 'Week_Starting_Date'})
            st.dataframe(weekly_forecasts, use_container_width=True)
        elif forecast_roll_up_choice == "Month":
            all_forecasts_df_display = st.session_state.all_forecasts_df.copy()
            all_forecasts_df_display['YearMonth'] = all_forecasts_df_display['Date'].dt.to_period('M').astype(str)
            monthly_forecasts = all_forecasts_df_display.groupby(['SKU_ID', 'Sales_Channel', 'YearMonth'])['Forecasted_Quantity'].sum().reset_index()
            monthly_forecasts = monthly_forecasts.rename(columns={'YearMonth': 'Month'})
            st.dataframe(monthly_forecasts, use_container_width=True)

    with tab3: # SKU Auto-Indent tab
        st.header("🛒 SKU-Level Auto-Indent Recommendations (Aggregated Across Channels)")
        st.info("Note: Auto-indent recommendations are aggregated to the SKU level, assuming a central inventory for all sales channels. For channel-specific inventory management, further logic would be required.")
        if not st.session_state.sku_indent_recommendations_df.empty:
            st.dataframe(st.session_state.sku_indent_recommendations_df, use_container_width=True)
        else:
            st.info("No SKU auto-indent recommendations available. Please run the forecast first.")

    with tab4: # NEW: Component Auto-Indent tab
        st.header("⚙️ Component-Level Auto-Indent Recommendations")
        st.info("Note: Component demand is derived from SKU orders. Safety stock and reorder points for components are calculated based on this derived demand and component-specific lead times and properties.")
        if not st.session_state.component_indent_recommendations_df.empty:
            st.dataframe(st.session_state.component_indent_recommendations_df, use_container_width=True)
        else:
            st.info("No Component auto-indent recommendations available. Please run the forecast first.")


    with tab5: # Fill Rates tab
        st.header("✅ Actual Fill Rates")
        if not st.session_state.actual_orders_df.empty and not st.session_state.actual_shipments_df.empty:
            fill_rates = calculate_fill_rates(st.session_state.actual_orders_df, st.session_state.actual_shipments_df)
            st.json(fill_rates)
        else:
            st.info("Upload 'Actual Orders Data' and 'Actual Shipments Data' to see Fill Rate calculations, or run on Sample Data.")

    with tab6: # Supply Chain KPIs & Visualizations tab
        st.header("📈 Forecast Visualization")
        if not st.session_state.processed_data_df.empty and not st.session_state.all_forecasts_df.empty:
            unique_skus_for_plot = st.session_state.all_forecasts_df['SKU_ID'].unique()
            if len(unique_skus_for_plot) > 0:
                # Consolidated SKU selection dropdown
                selected_sku_for_plot = st.selectbox(
                    "Select SKU to Visualize",
                    unique_skus_for_plot,
                    key='unified_sku_select' # Unique key for the consolidated selectbox
                )
                
                # Allow selection of channel for forecast visualization (still specific to forecast)
                channels_for_selected_sku = st.session_state.all_forecasts_df[st.session_state.all_forecasts_df['SKU_ID'] == selected_sku_for_plot]['Sales_Channel'].unique().tolist()
                selected_channel_for_plot = st.selectbox(
                    "Select Sales Channel for Forecast Visualization",
                    channels_for_selected_sku,
                    key='forecast_channel_select'
                )

                # Filter historical data for the selected SKU and Channel
                history_df = st.session_state.processed_data_df[
                    (st.session_state.processed_data_df['SKU_ID'] == selected_sku_for_plot) &
                    (st.session_state.processed_data_df['Sales_Channel'] == selected_channel_for_plot)
                ][['Date', 'Sales_Quantity']].copy()
                history_df.rename(columns={'Sales_Quantity': 'Quantity'}, inplace=True)
                history_df['Type'] = 'Historical Sales'

                # Filter forecasted data for the selected SKU and Channel
                forecast_plot_df = st.session_state.all_forecasts_df[
                    (st.session_state.all_forecasts_df['SKU_ID'] == selected_sku_for_plot) &
                    (st.session_state.all_forecasts_df['Sales_Channel'] == selected_channel_for_plot)
                ][['Date', 'Forecasted_Quantity']].copy()
                forecast_plot_df.rename(columns={'Forecasted_Quantity': 'Quantity'}, inplace=True)
                forecast_plot_df['Type'] = 'Forecasted Demand'

                # Combine for plotting
                combined_plot_df = pd.concat([history_df, forecast_plot_df], ignore_index=True)
                combined_plot_df = combined_plot_df.sort_values(by='Date').reset_index(drop=True)

                # Apply roll-up for plotting
                if forecast_roll_up_choice == "Day":
                    plot_df = combined_plot_df
                    x_axis_title = "Date"
                elif forecast_roll_up_choice == "Week":
                    combined_plot_df['Period'] = combined_plot_df['Date'].dt.to_period('W').astype(str)
                    plot_df = combined_plot_df.groupby(['Period', 'Type'])['Quantity'].sum().reset_index()
                    plot_df['Date'] = plot_df['Period'].apply(lambda x: pd.Period(x).start_time) # Convert back to datetime for plotting
                    x_axis_title = "Week"
                elif forecast_roll_up_choice == "Month":
                    combined_plot_df['Period'] = combined_plot_df['Date'].dt.to_period('M').astype(str)
                    plot_df = combined_plot_df.groupby(['Period', 'Type'])['Quantity'].sum().reset_index()
                    plot_df['Date'] = plot_df['Period'].apply(lambda x: pd.Period(x).start_time) # Convert back to datetime for plotting
                    x_axis_title = "Month"

                # Filter for last 12 periods for historical data
                history_plot_df = plot_df[plot_df['Type'] == 'Historical Sales'].copy()
                forecast_plot_df_filtered = plot_df[plot_df['Type'] == 'Forecasted Demand'].copy()

                if forecast_roll_up_choice == "Day":
                    history_plot_df = history_plot_df.tail(12) # Last 12 days
                elif forecast_roll_up_choice == "Week":
                    history_plot_df = history_plot_df.tail(12) # Last 12 weeks
                elif forecast_roll_up_choice == "Month":
                    history_plot_df = history_plot_df.tail(12) # Last 12 months
                
                final_plot_df = pd.concat([history_plot_df, forecast_plot_df_filtered], ignore_index=True)
                final_plot_df = final_plot_df.sort_values(by='Date')


                fig = px.line(
                    final_plot_df,
                    x="Date",
                    y="Quantity",
                    color="Type",
                    title=f"Historical Sales vs. Forecasted Demand for {selected_sku_for_plot} ({selected_channel_for_plot} - Roll-up: {forecast_roll_up_choice})",
                    labels={"Quantity": "Quantity", "Date": x_axis_title},
                    line_dash="Type", # Use line dash to distinguish historical vs forecast
                    color_discrete_map={'Historical Sales': 'blue', 'Forecasted Demand': 'red'}
                )
                fig.update_layout(hovermode="x unified") # Improves hover experience
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No SKUs available for plotting.")
        else:
            st.info("No data available to plot forecasts. Please run the forecast first.")
        
        st.markdown("---")
        st.subheader("📊 Supply Chain KPIs")

        if not st.session_state.all_inventory_kpis_df.empty:
            unique_skus_for_kpi_plot = st.session_state.all_inventory_kpis_df['SKU_ID'].unique()
            if len(unique_skus_for_kpi_plot) > 0:
                # Plot Inventory Turnover
                inventory_turnover_plot_df = aggregate_kpi_for_plot(
                    st.session_state.all_inventory_kpis_df, selected_sku_for_plot, 'Inventory_Turnover', forecast_roll_up_choice, 'mean'
                ).tail(12) # Last 12 periods
                
                if not inventory_turnover_plot_df.empty:
                    fig_turnover = px.line(
                        inventory_turnover_plot_df,
                        x="Date",
                        y="Value",
                        title=f"Inventory Turnover for {selected_sku_for_plot} (Roll-up: {forecast_roll_up_choice})",
                        labels={"Value": "Turnover Ratio", "Date": forecast_roll_up_choice},
                        color_discrete_sequence=['green']
                    )
                    fig_turnover.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_turnover, use_container_width=True)
                else:
                    st.info(f"No Inventory Turnover data to plot for {selected_sku_for_plot}.")

                # Plot Days of Inventory Outstanding
                dio_plot_df = aggregate_kpi_for_plot(
                    st.session_state.all_inventory_kpis_df, selected_sku_for_plot, 'Days_Inventory_Outstanding', forecast_roll_up_choice, 'mean'
                ).tail(12) # Last 12 periods

                if not dio_plot_df.empty:
                    fig_dio = px.line(
                        dio_plot_df,
                        x="Date",
                        y="Value",
                        title=f"Days of Inventory Outstanding (DIO) for {selected_sku_for_plot} (Roll-up: {forecast_roll_up_choice})",
                        labels={"Value": "Days", "Date": forecast_roll_up_choice},
                        color_discrete_sequence=['purple']
                    )
                    fig_dio.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_dio, use_container_width=True)
                else:
                    st.info(f"No Days of Inventory Outstanding data to plot for {selected_sku_for_plot}.")
            else:
                st.info("No SKUs available for Inventory KPI plotting.")
        else:
            st.info("No Inventory KPI data available. Please ensure Sales and Inventory data are uploaded or run on Sample Data.")

        if not st.session_state.all_stockout_rates_df.empty:
            unique_skus_for_stockout_plot = st.session_state.all_stockout_rates_df['SKU_ID'].unique()
            if len(unique_skus_for_stockout_plot) > 0:
                stockout_plot_df = aggregate_kpi_for_plot(
                    st.session_state.all_stockout_rates_df, selected_sku_for_plot, 'Stockout_Rate', forecast_roll_up_choice, 'mean'
                ).tail(12) # Last 12 periods

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
                    st.plotly_chart(fig_stockout, use_container_width=True)
                else:
                    st.info(f"No Stockout Rate data to plot for {selected_sku_for_plot}.")
            else:
                st.info("No SKUs available for Stockout Rate plotting.")
        else:
            st.info("No Stockout Rate data available. Please ensure Actual Orders and Actual Shipments data are uploaded or run on Sample Data.")
