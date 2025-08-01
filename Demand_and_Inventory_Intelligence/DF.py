# DFv13.py - Demand and Inventory Intelligence Streamlit App
# Features: Full multi-echelon simulation, detailed cost analysis, BOM integration,
#           comprehensive reporting, forecast model selector and FAQ.
#
# Changes in this version:
# - Refactored run_full_simulation into smaller, more manageable functions.
# - Optimized Reorder Point (ROP) and Safety Stock (SS) calculations (pre-calculated).
# - Added Streamlit caching (@st.cache_data) for performance improvements.
# - Implemented basic schema validation for uploaded CSV files.
# - Ensured consistent int() casting for timedelta days component.
# - NEW: Added a dedicated "How it's Calculated" section for business users.
# - NEW: Added a download button to export simulation results as CSV.
# - FIX: Addressed TypeError when df_sales['Date'].max() is called on empty data,
#        by adding a check before drawing the forecast start vline.

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
import io
import base64

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
DEFAULT_REGIONS = ["North America", "Europe", "Asia"]

# Define locations by type
DEFAULT_FACTORIES = ["Factory-A"]
DEFAULT_DISTRIBUTION_CENTERS = [f"DC-{chr(ord('A') + i)}" for i in range(2)]
DEFAULT_RETAIL_STORES = [f"Store-{chr(ord('A') + i)}" for i in range(3)]
DEFAULT_LOCATIONS = DEFAULT_FACTORIES + DEFAULT_DISTRIBUTION_CENTERS + DEFAULT_RETAIL_STORES

# --- Schema Definitions for Validation ---
SALES_SCHEMA = {
    "Date": pd.api.types.is_datetime64_any_dtype,
    "SKU_ID": pd.api.types.is_string_dtype,
    "Location": pd.api.types.is_string_dtype,
    "Sales_Quantity": pd.api.types.is_integer_dtype,
    "Price": pd.api.types.is_numeric_dtype,
    "Customer_Segment": pd.api.types.is_string_dtype,
    "Sales_Channel": pd.api.types.is_string_dtype,
}

INVENTORY_SCHEMA = {
    "Date": pd.api.types.is_datetime64_any_dtype,
    "SKU_ID": pd.api.types.is_string_dtype,
    "Location": pd.api.types.is_string_dtype,
    "Current_Stock": pd.api.types.is_integer_dtype,
}

LEAD_TIMES_SCHEMA = {
    "Item_ID": pd.api.types.is_string_dtype,
    "Item_Type": pd.api.types.is_string_dtype, # e.g., Finished_Good, Component
    "From_Location": pd.api.types.is_string_dtype,
    "To_Location": pd.api.types.is_string_dtype,
    "Lead_Time_Days": pd.api.types.is_integer_dtype,
    "Min_Order_Quantity": pd.api.types.is_integer_dtype,
    "Order_Multiple": pd.api.types.is_integer_dtype,
}

BOM_SCHEMA = {
    "Parent_SKU_ID": pd.api.types.is_string_dtype,
    "Component_ID": pd.api.types.is_string_dtype,
    "Quantity_Required": pd.api.types.is_integer_dtype,
    "Component_Type": pd.api.types.is_string_dtype, # e.g., Raw_Material
    "Shelf_Life_Days": pd.api.types.is_integer_dtype,
}

GLOBAL_CONFIG_SCHEMA = {
    "Parameter": pd.api.types.is_string_dtype,
    "Value": pd.api.types.is_numeric_dtype,
}

ACTUAL_ORDERS_SCHEMA = {
    "Date": pd.api.types.is_datetime64_any_dtype,
    "Order_ID": pd.api.types.is_string_dtype,
    "SKU_ID": pd.api.types.is_string_dtype,
    "Ordered_Quantity": pd.api.types.is_integer_dtype,
    "Sales_Channel": pd.api.types.is_string_dtype,
}

ACTUAL_SHIPMENTS_SCHEMA = {
    "Date": pd.api.types.is_datetime64_any_dtype,
    "Order_ID": pd.api.types.is_string_dtype,
    "SKU_ID": pd.api.types.is_string_dtype,
    "Shipped_Quantity": pd.api.types.is_integer_dtype,
    "Sales_Channel": pd.api.types.is_string_dtype,
}

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def generate_realistic_data(num_skus, num_components, num_factories, num_dcs, num_stores, network_type):
    """
    Generates a complete set of realistic dummy dataframes for the app.
    Caches the output for performance.
    """
    st.info("Generating a complete set of realistic dummy data...")
    
    # Generate SKUs and Components
    skus = [f'SKU-{i+1}' for i in range(num_skus)]
    components = [f'COMP-{i+1}' for i in range(num_skus * num_components)]
    all_items = skus + components

    # Define dynamic locations based on user input
    factories = [f'Factory-{i+1}' for i in range(num_factories)]
    distribution_centers = [f'DC-{i+1}' for i in range(num_dcs)]
    retail_stores = [f'Store-{i+1}' for i in range(num_stores)]
    
    all_sim_locations = factories + distribution_centers + retail_stores

    # Sales Data
    date_range = pd.date_range(start=DEFAULT_START_DATE, end=DEFAULT_END_DATE, freq='D')
    sales_data = []
    
    num_dates = len(date_range)
    day_of_year_sin = np.sin(np.linspace(0, 2 * np.pi, num_dates)) * 0.4 + 1.0 # Yearly seasonality
    day_of_year_cos = np.cos(np.linspace(0, 4 * np.pi, num_dates)) * 0.1 + 1.0 # Quarterly
    
    for date_idx, date in enumerate(date_range):
        for sku in skus:
            for store in retail_stores:
                base_sales = random.randint(0, 100)
                
                demand_factor = 1.0
                demand_factor *= day_of_year_sin[date_idx] * day_of_year_cos[date_idx]
                
                if random.random() < 0.05: # 5% chance of a promo day
                    demand_factor *= random.uniform(1.5, 2.5)
                
                sales_quantity = max(0, int(base_sales * demand_factor) + random.randint(-10, 10))
                
                sales_data.append({
                    "Date": date,
                    "SKU_ID": sku,
                    "Location": store,
                    "Sales_Quantity": sales_quantity,
                    "Price": round(random.uniform(50, 500), 2),
                    "Customer_Segment": random.choice(["Retail", "Wholesale", "Online"]),
                    "Sales_Channel": random.choice(DEFAULT_SALES_CHANNELS)
                })
    df_sales = pd.DataFrame(sales_data)

    # Inventory Data
    inventory_data = []
    for item in all_items:
        for loc in all_sim_locations:
            inventory_data.append({
                "Date": date_range.min(), # Initial inventory at the start date
                "SKU_ID": item,
                "Location": loc,
                "Current_Stock": random.randint(100, 1000)
            })
    df_inventory = pd.DataFrame(inventory_data)

    # Lead Times Data & Network Definition
    lead_times_data = []
    
    if network_type == "Single-Echelon":
        # All stores are supplied directly from factories
        for store in retail_stores:
            for sku in skus:
                supplier = random.choice(factories)
                lead_times_data.append({
                    "Item_ID": sku,
                    "Item_Type": "Finished_Good",
                    "From_Location": supplier,
                    "To_Location": store,
                    "Lead_Time_Days": random.randint(3, 14),
                    "Min_Order_Quantity": random.choice([50, 100, 200]),
                    "Order_Multiple": random.choice([10, 20, 50])
                })
    else: # Multi-Echelon
        # Factories supply DCs
        for dc in distribution_centers:
            for sku in skus:
                supplier = random.choice(factories)
                lead_times_data.append({
                    "Item_ID": sku,
                    "Item_Type": "Finished_Good",
                    "From_Location": supplier,
                    "To_Location": dc,
                    "Lead_Time_Days": random.randint(5, 20),
                    "Min_Order_Quantity": random.choice([200, 500, 1000]),
                    "Order_Multiple": random.choice([50, 100])
                })
        # DCs supply retail stores
        for store in retail_stores:
            for sku in skus:
                supplier = random.choice(distribution_centers)
                lead_times_data.append({
                    "Item_ID": sku,
                    "Item_Type": "Finished_Good",
                    "From_Location": supplier,
                    "To_Location": store,
                    "Lead_Time_Days": random.randint(2, 7),
                    "Min_Order_Quantity": random.choice([20, 50, 100]),
                    "Order_Multiple": random.choice([10, 20])
                })
    
    # Add component lead times to the data
    for component in components:
        lead_times_data.append({
            "Item_ID": component,
            "Item_Type": "Component",
            "From_Location": f"Vendor-{random.randint(1, 3)}",
            "To_Location": random.choice(factories),
            "Lead_Time_Days": random.randint(10, 30),
            "Min_Order_Quantity": random.choice([1000, 2000, 5000]),
            "Order_Multiple": random.choice([100, 500])
        })
        
    df_lead_times = pd.DataFrame(lead_times_data)

    # BOM Data
    bom_data = []
    for sku in skus:
        for i in range(random.randint(1, num_components)):
            component = random.choice(components)
            bom_data.append({
                "Parent_SKU_ID": sku,
                "Component_ID": component,
                "Quantity_Required": random.randint(1, 5),
                "Component_Type": "Raw_Material",
                "Shelf_Life_Days": random.randint(DEFAULT_MAX_SKU_SHELF_LIFE_DAYS - 200, DEFAULT_MAX_SKU_SHELF_LIFE_DAYS)
            })
    df_bom = pd.DataFrame(bom_data)
    df_bom.drop_duplicates(subset=["Parent_SKU_ID", "Component_ID"], inplace=True)
    
    # Global Config Data
    global_config_data = {
        "Parameter": ["Holding_Cost_Per_Unit_Per_Day", "Ordering_Cost_Per_Order", "Stockout_Cost_Per_Unit"],
        "Value": [0.05, 50.0, 10.0]
    }
    df_global_config = pd.DataFrame(global_config_data)

    # Dummy actual orders and shipments for historical fill rate calculation
    actual_orders_data = []
    for index, row in df_sales.iterrows():
        if row["Sales_Quantity"] > 0:
            actual_orders_data.append({
                "Date": row["Date"],
                "Order_ID": f"ORD-{index}",
                "SKU_ID": row["SKU_ID"],
                "Ordered_Quantity": row["Sales_Quantity"],
                "Sales_Channel": row["Sales_Channel"]
            })
    df_actual_orders = pd.DataFrame(actual_orders_data)

    actual_shipments_data = []
    for index, row in df_actual_orders.iterrows():
        shipped_quantity = min(row["Ordered_Quantity"], random.randint(0, row["Ordered_Quantity"] + 20))
        actual_shipments_data.append({
            "Date": row["Date"] + timedelta(days=random.randint(0, 3)),
            "Order_ID": row["Order_ID"],
            "SKU_ID": row["SKU_ID"],
            "Shipped_Quantity": shipped_quantity,
            "Sales_Channel": row["Sales_Channel"]
        })
    df_actual_shipments = pd.DataFrame(actual_shipments_data)


    return {
        "sales_data.csv": df_sales,
        "inventory_data.csv": df_inventory,
        "lead_times_data.csv": df_lead_times,
        "bom_data.csv": df_bom,
        "global_config.csv": df_global_config,
        "actual_orders.csv": df_actual_orders,
        "actual_shipments.csv": df_actual_shipments
    }

def get_csv_download_link(df, filename):
    """
    Generates a single download link for a pandas DataFrame as a CSV file.
    """
    csv_string = df.to_csv(index=False)
    b64_csv = base64.b64encode(csv_string.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64_csv}" download="{filename}">Download {filename}</a>'
    return href

def _validate_dataframe_schema(df, expected_schema, df_name):
    """
    Validates a DataFrame against an expected schema.
    expected_schema = {'column_name': pd.api.types.is_datetime64_any_dtype, ...}
    Returns True if valid, False otherwise and prints errors to Streamlit.
    """
    errors = []
    # Check for missing columns
    for col, dtype_checker in expected_schema.items():
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")
        else:
            # Check data type (flexible for now, mainly datetime)
            if 'Date' in col and not pd.api.types.is_datetime64_any_dtype(df[col]):
                 try:
                     df[col] = pd.to_datetime(df[col])
                 except Exception:
                     errors.append(f"Column '{col}' is not in a valid datetime format.")
            elif not dtype_checker(df[col]):
                # More generic dtype checking could be added here
                # For now, this is a basic check.
                pass 
    
    if errors:
        st.error(f"Schema validation failed for '{df_name}':")
        for error in errors:
            st.error(f"- {error}")
        return False
    return True

def calculate_safety_stock_kings(df_demand, lead_time_days, service_level):
    """
    Calculates safety stock using King's method.
    """
    if df_demand.empty or df_demand['Sales_Quantity'].isnull().all():
        return 0
    
    # Ensure lead_time_days is treated as an integer for calculations
    lead_time_days = int(lead_time_days)

    df_demand['Date'] = pd.to_datetime(df_demand['Date'])
    df_daily_demand = df_demand.groupby('Date')['Sales_Quantity'].sum().reset_index()
    std_dev_demand = df_daily_demand['Sales_Quantity'].std()
    
    # Handle NaN std_dev_demand if all values are the same or not enough data
    if pd.isna(std_dev_demand):
        std_dev_demand = 0
    
    z_score = norm.ppf(service_level)
    safety_stock = z_score * math.sqrt(lead_time_days) * std_dev_demand
    return max(0, int(safety_stock))

def calculate_safety_stock_avg_max(df_demand, lead_time_days):
    """
    Calculates safety stock using the Avg Max method.
    """
    if df_demand.empty or df_demand['Sales_Quantity'].isnull().all():
        return 0

    # Ensure lead_time_days is treated as an integer for calculations
    lead_time_days = int(lead_time_days)

    df_demand['Date'] = pd.to_datetime(df_demand['Date'])
    df_daily_demand = df_demand.groupby('Date')['Sales_Quantity'].sum().reset_index()
    
    max_daily_demand = df_daily_demand['Sales_Quantity'].max()
    avg_daily_demand = df_daily_demand['Sales_Quantity'].mean()
    
    # Handle NaN values if data is sparse
    if pd.isna(max_daily_demand): max_daily_demand = 0
    if pd.isna(avg_daily_demand): avg_daily_demand = 0

    safety_stock = (max_daily_demand * lead_time_days) - (avg_daily_demand * lead_time_days)
    return max(0, int(safety_stock))

def calculate_historical_fill_rate(df_actual_orders, df_actual_shipments):
    """
    Calculates the historical fill rate based on actual orders and shipments.
    """
    if df_actual_orders.empty or df_actual_shipments.empty:
        return 0.0

    df_merged = pd.merge(df_actual_orders, df_actual_shipments, on=["Order_ID", "SKU_ID", "Sales_Channel"], suffixes=('_order', '_ship'))
    total_ordered = df_merged['Ordered_Quantity'].sum()
    total_shipped = df_merged['Shipped_Quantity'].sum()

    if total_ordered == 0:
        return 100.0 # Avoid division by zero
    
    fill_rate = (total_shipped / total_ordered) * 100
    return fill_rate

def forecast_demand(df_sales, forecast_model, forecast_days, ma_window_size=7):
    """
    Generates a forecast for future sales demand using the selected model.
    """
    df_sales['Date'] = pd.to_datetime(df_sales['Date'])
    df_daily_demand = df_sales.groupby(['Date', 'SKU_ID', 'Location'])['Sales_Quantity'].sum().reset_index()
    
    forecast_results = []
    
    unique_skus = df_daily_demand['SKU_ID'].unique()
    unique_locations = df_daily_demand['Location'].unique()
    
    for sku in unique_skus:
        for location in unique_locations:
            df_subset = df_daily_demand[(df_daily_demand['SKU_ID'] == sku) & (df_daily_demand['Location'] == location)].set_index('Date').sort_index()
            df_subset = df_subset.asfreq('D', fill_value=0) # Ensure all dates are present
            
            last_date = df_subset.index.max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            
            if forecast_model == "Moving Average":
                window_size = ma_window_size
                df_subset['Moving_Average'] = df_subset['Sales_Quantity'].rolling(window=window_size).mean()
                last_avg = df_subset['Moving_Average'].iloc[-1] if not df_subset['Moving_Average'].isnull().all() else 0
                forecast_values = [last_avg] * forecast_days
            elif forecast_model == "Moving Median":
                window_size = ma_window_size
                df_subset['Moving_Median'] = df_subset['Sales_Quantity'].rolling(window=window_size).median()
                last_median = df_subset['Moving_Median'].iloc[-1] if not df_subset['Moving_Median'].isnull().all() else 0
                forecast_values = [last_median] * forecast_days
            elif forecast_model in ["Random Forest", "XGBoost"]:
                # Simplistic feature engineering for ML models
                df_subset['dayofweek'] = df_subset.index.dayofweek
                df_subset['dayofyear'] = df_subset.index.dayofyear
                
                features = ['dayofweek', 'dayofyear']
                target = 'Sales_Quantity'
                
                # Split data - using a simple split for demonstration
                if len(df_subset) > 100: # Ensure enough data for training
                    train_size = int(len(df_subset) * 0.8)
                    train_data = df_subset.iloc[:train_size]
                    
                    X_train, y_train = train_data[features], train_data[target]
                    
                    if forecast_model == "Random Forest":
                        model = RandomForestRegressor(random_state=42)
                    else:
                        model = XGBRegressor(random_state=42)
                        
                    model.fit(X_train, y_train)
                    
                    future_df = pd.DataFrame(index=future_dates)
                    future_df['dayofweek'] = future_df.index.dayofweek
                    future_df['dayofyear'] = future_df.index.dayofyear
                    
                    forecast_values = model.predict(future_df[features])
                else:
                    forecast_values = [df_subset['Sales_Quantity'].mean() if not df_subset.empty else 0] * forecast_days # Fallback to mean if not enough data
            
            for date, forecast_qty in zip(future_dates, forecast_values):
                forecast_results.append({
                    'Date': date,
                    'SKU_ID': sku,
                    'Location': location,
                    'Sales_Quantity': max(0, int(forecast_qty)) # Ensure non-negative integer quantity
                })
    
    return pd.DataFrame(forecast_results)

def suggest_indent_orders(df_inventory_latest, df_forecast, df_lead_times, safety_stock_method, service_level, current_simulation_date, precalculated_reorder_policy):
    """
    Suggests indent orders (internal transfers or vendor orders) based on current inventory,
    forecasted demand, lead times, MOQ, and order multiples.
    Uses precalculated reorder policy for efficiency.
    """
    suggestions = []

    # Iterate through all unique (To_Location, Item_ID) pairs that have a defined reorder policy
    # These are the "demand points" in the network (stores, DCs, factories needing components)
    for (location, item_id), policy in precalculated_reorder_policy.items():
        # Get current inventory
        current_stock = df_inventory_latest.get((location, item_id), 0)

        # Retrieve precalculated policy values
        lead_time_days = policy['lead_time']
        min_order_qty = policy['min_order']
        order_multiple = policy['order_mult']
        supplier = policy['supplier']
        rop = policy['rop']
        ss = policy['ss']

        # If current stock is below reorder point, suggest an order
        if current_stock <= rop:
            # Calculate demand during lead time from forecast
            forecast_end_date_for_lead_time = current_simulation_date + timedelta(days=int(lead_time_days))
            
            # Identify the relevant demand for this location/item.
            # If it's a store, its demand is direct sales forecast.
            # If it's a DC, its demand is aggregate of stores it supplies.
            # If it's a component for a factory, its demand is based on projected production.
            
            demand_during_lead_time_forecast = 0
            if 'Store' in location:
                df_item_location_forecast = df_forecast[(df_forecast['SKU_ID'] == item_id) & 
                                                        (df_forecast['Location'] == location)]
                demand_during_lead_time_forecast = df_item_location_forecast[
                    (df_item_location_forecast['Date'] > current_simulation_date) & 
                    (df_item_location_forecast['Date'] <= forecast_end_date_for_lead_time)
                ]['Sales_Quantity'].sum()
            elif 'DC' in location:
                # Aggregate forecast demand from all stores supplied by this DC for this item
                downstream_stores = df_lead_times[
                    (df_lead_times['From_Location'] == location) & 
                    (df_lead_times['Item_ID'] == item_id) &
                    (df_lead_times['Item_Type'] == 'Finished_Good') # DCs supply finished goods
                ]['To_Location'].unique()
                
                demand_during_lead_time_forecast = df_forecast[
                    (df_forecast['Date'] > current_simulation_date) & 
                    (df_forecast['Date'] <= forecast_end_date_for_lead_time) &
                    (df_forecast['SKU_ID'] == item_id) &
                    (df_forecast['Location'].isin(downstream_stores))
                ]['Sales_Quantity'].sum() if len(downstream_stores) > 0 else 0
            elif 'Factory' in location and df_lead_times[(df_lead_times['To_Location']==location) & (df_lead_times['Item_ID']==item_id)]['Item_Type'].iloc[0] == 'Component':
                # For components at factories, demand is derived from forecasted finished good production.
                # This requires summing up forecasted SKU demand that uses this component, then multiplying by quantity required.
                # This is a simplified approach for demonstration: assume component demand is tied to finished goods.
                parent_skus = df_bom[df_bom['Component_ID'] == item_id]['Parent_SKU_ID'].unique()
                if len(parent_skus) > 0:
                    component_qty_map = df_bom.set_index(['Parent_SKU_ID', 'Component_ID'])['Quantity_Required'].to_dict()
                    for parent_sku in parent_skus:
                        # Assuming the factory produces all relevant SKUs that use this component
                        avg_sku_demand = df_forecast[(df_forecast['SKU_ID'] == parent_sku) & 
                                                     (df_forecast['Location'] == location)]['Sales_Quantity'].mean()
                        if pd.isna(avg_sku_demand): avg_sku_demand = 0

                        qty_per_sku = component_qty_map.get((parent_sku, item_id), 0)
                        demand_during_lead_time_forecast += avg_sku_demand * lead_time_days * qty_per_sku # Simplified projection

            # Target inventory to cover demand during lead time plus safety stock
            target_inventory = demand_during_lead_time_forecast + ss
            
            # Calculate raw quantity to order
            raw_order_qty = max(0, target_inventory - current_stock)

            # Adjust quantity to meet Min_Order_Quantity and Order_Multiple
            if raw_order_qty > 0:
                order_qty = max(min_order_qty, raw_order_qty)
                order_qty = math.ceil(order_qty / order_multiple) * order_multiple if order_multiple > 0 else order_qty
            else:
                order_qty = 0 # No order needed if raw_order_qty is 0 or less
            
            if order_qty > 0:
                order_type = "Vendor Order"
                # Infer order type based on supplier location
                if "DC-" in supplier or "Factory-" in supplier or "Store-" in supplier: # Check if supplier is internal
                    order_type = "Internal Transfer"

                suggestions.append({
                    "Date_of_Suggestion": current_simulation_date,
                    "Item_ID": item_id,
                    "Location_Needing_Order": location,
                    "Supplier_Location": supplier,
                    "Order_Type": order_type,
                    "Suggested_Order_Quantity": order_qty,
                    "Current_Stock": current_stock,
                    "Reorder_Point": int(rop),
                    "Safety_Stock": int(ss),
                    "Lead_Time_Days": lead_time_days,
                    "Min_Order_Quantity": min_order_qty,
                    "Order_Multiple": order_multiple
                })
    
    return pd.DataFrame(suggestions)


# --- Refactored Simulation Helper Functions ---

def _process_arrivals(current_date, inventory_levels, incoming_shipments, simulation_events):
    """Processes incoming shipments for the current day."""
    if current_date in incoming_shipments:
        for (location, item), shipments in incoming_shipments[current_date].items():
            for quantity in shipments:
                inventory_levels[(location, item)] = inventory_levels.get((location, item), 0) + quantity
                simulation_events.append({
                    "Date": current_date,
                    "Type": "Shipment_Arrival",
                    "Item_ID": item,
                    "Location": location,
                    "Quantity": quantity,
                    "Description": f"Shipment of {quantity} {item} arrived at {location}."
                })
        del incoming_shipments[current_date] # Clear arrivals for the day


def _process_daily_sales_demand(current_date, inventory_levels, daily_demand_df, reorder_policy, 
                                 simulation_events, cost_metrics, df_forecast, incoming_shipments):
    """Processes customer sales at retail stores and triggers their reorders."""
    
    # Filter for sales demand relevant to the current date
    current_day_sales = daily_demand_df[daily_demand_df['Date'] == current_date]
    
    for _, row in current_day_sales.iterrows():
        location = row['Location']
        sku = row['SKU_ID']
        demand_qty = row['Sales_Quantity']
        
        # Only process for retail stores and where there is actual demand (not just forecast entry)
        if 'Store' in location and demand_qty > 0:
            cost_metrics['total_sales_demand'] += demand_qty
            current_stock = inventory_levels.get((location, sku), 0)
            
            shipped_qty = min(current_stock, demand_qty)
            lost_sales = demand_qty - shipped_qty
            
            inventory_levels[(location, sku)] = current_stock - shipped_qty
            cost_metrics['total_lost_sales'] += lost_sales
            cost_metrics['total_stockout_cost'] += lost_sales * cost_metrics['stockout_cost']
            
            simulation_events.append({
                "Date": current_date,
                "Type": "Sales_Demand",
                "Item_ID": sku,
                "Location": location,
                "Quantity": demand_qty,
                "Description": f"Customer demand for {demand_qty} {sku} at {location}. {shipped_qty} fulfilled, {lost_sales} lost sales."
            })
            
            # Check reorder point for this retail store SKU
            policy = reorder_policy.get((location, sku))
            if policy and inventory_levels.get((location, sku), 0) <= policy['rop']:
                lead_time = policy['lead_time']
                min_order_qty = policy['min_order']
                order_multiple = policy['order_mult']
                supplier = policy['supplier']
                ss = policy['ss']

                # Calculate quantity needed based on forecast for lead time + safety stock
                forecast_period_start = current_date + timedelta(days=1)
                forecast_period_end = current_date + timedelta(days=int(lead_time))
                
                demand_during_lead_time_forecast = df_forecast[
                    (df_forecast['Date'] >= forecast_period_start) & 
                    (df_forecast['Date'] <= forecast_period_end) &
                    (df_forecast['SKU_ID'] == sku) &
                    (df_forecast['Location'] == location)
                ]['Sales_Quantity'].sum()
                
                raw_order_qty = max(0, (demand_during_lead_time_forecast + ss) - inventory_levels.get((location, sku), 0))
                
                order_qty = max(min_order_qty, raw_order_qty)
                order_qty = math.ceil(order_qty / order_multiple) * order_multiple if order_multiple > 0 else order_qty
                
                if order_qty > 0:
                    cost_metrics['total_ordering_cost'] += cost_metrics['ordering_cost']
                    
                    # Schedule the shipment from supplier to the current location (store)
                    arrival_date = current_date + timedelta(days=int(lead_time)) # Use int for timedelta
                    incoming_shipments.setdefault(arrival_date, {}).setdefault((location, sku), []).append(order_qty)
                    
                    simulation_events.append({
                        "Date": current_date,
                        "Type": "Reorder_Placed",
                        "Item_ID": sku,
                        "Location": location,
                        "Quantity": order_qty,
                        "Description": f"Reorder of {order_qty} {sku} placed with {supplier}. Expected arrival: {arrival_date.strftime('%Y-%m-%d')}."
                    })

def _process_upstream_orders_and_reorders(current_date, inventory_levels, reorder_policy, 
                                           incoming_shipments, simulation_events, 
                                           cost_metrics, df_forecast, df_lead_times, 
                                           factory_production_requests):
    """
    Processes internal orders at DCs/Factories (not direct sales) and triggers their upstream reorders.
    Also populates `factory_production_requests`.
    """
    
    # Locations to process: DCs and Factories (these are the 'To_Location' in lead times for upstream supply)
    # Iterate through unique (To_Location, Item_ID) where 'To_Location' is a DC or Factory
    for (loc, item_id), policy in reorder_policy.items():
        if 'DC' not in loc and 'Factory' not in loc:
            continue # Already handled stores, or this item/location doesn't have an upstream supplier in lead_times.

        # Check reorder point for this DC/Factory item
        if inventory_levels.get((loc, item_id), 0) <= policy['rop']:
            lead_time = policy['lead_time']
            min_order_qty = policy['min_order']
            order_multiple = policy['order_mult']
            supplier = policy['supplier']
            ss = policy['ss']

            forecast_period_start = current_date + timedelta(days=1)
            forecast_period_end = current_date + timedelta(days=int(lead_time))
            
            # Determine demand for this upstream location/item
            demand_during_lead_time_forecast = 0
            if 'DC' in loc: # DC's demand is aggregate of stores it supplies
                downstream_locations = df_lead_times[
                    (df_lead_times['From_Location'] == loc) & 
                    (df_lead_times['Item_ID'] == item_id)
                ]['To_Location'].unique() # Get all stores supplied by this DC for this item
                
                demand_during_lead_time_forecast = df_forecast[
                    (df_forecast['Date'] >= forecast_period_start) & 
                    (df_forecast['Date'] <= forecast_period_end) &
                    (df_forecast['SKU_ID'] == item_id) &
                    (df_forecast['Location'].isin(downstream_locations))
                ]['Sales_Quantity'].sum() if len(downstream_locations) > 0 else 0
            
            elif 'Factory' in loc: # Factory's demand for components or for finished goods production
                item_type = df_lead_times[(df_lead_times['To_Location']==loc) & (df_lead_times['Item_ID']==item_id)]['Item_Type'].iloc[0]
                
                if item_type == 'Component': # Factory needs component for production
                    # This logic needs to be more robust for component demand. For simplicity,
                    # base component demand on overall forecast for parent SKUs using this component.
                    parent_skus = df_bom[df_bom['Component_ID'] == item_id]['Parent_SKU_ID'].unique()
                    component_demand_data = []
                    for p_sku in parent_skus:
                        qty_req = df_bom[(df_bom['Parent_SKU_ID'] == p_sku) & (df_bom['Component_ID'] == item_id)]['Quantity_Required'].iloc[0]
                        # Assuming factory produces SKUs for overall forecast demand
                        sku_forecast = df_forecast[(df_forecast['SKU_ID'] == p_sku) & (df_forecast['Location'] == loc)]
                        if not sku_forecast.empty:
                            component_demand_data.append(sku_forecast[['Date', 'Sales_Quantity']].copy())
                            component_demand_data[-1]['Sales_Quantity'] *= qty_req

                    if component_demand_data:
                        # Combine and sum up sales quantities for the component's demand
                        temp_df = pd.concat(component_demand_data)
                        temp_df = temp_df[(temp_df['Date'] >= forecast_period_start) & (temp_df['Date'] <= forecast_period_end)]
                        demand_during_lead_time_forecast = temp_df['Sales_Quantity'].sum()
                    else:
                        demand_during_lead_time_forecast = 0
                else: # Factory needs to produce a finished good itself (likely based on DC orders)
                     # Sum demand from all DCs this factory supplies for this finished good
                    downstream_locations = df_lead_times[
                        (df_lead_times['From_Location'] == loc) & 
                        (df_lead_times['Item_ID'] == item_id)
                    ]['To_Location'].unique()
                    
                    demand_during_lead_time_forecast = df_forecast[
                        (df_forecast['Date'] >= forecast_period_start) & 
                        (df_forecast['Date'] <= forecast_period_end) &
                        (df_forecast['SKU_ID'] == item_id) &
                        (df_forecast['Location'].isin(downstream_locations))
                    ]['Sales_Quantity'].sum() if len(downstream_locations) > 0 else 0
            
            raw_order_qty = max(0, (demand_during_lead_time_forecast + ss) - inventory_levels.get((loc, item_id), 0))
            
            order_qty = max(min_order_qty, raw_order_qty)
            order_qty = math.ceil(order_qty / order_multiple) * order_multiple if order_multiple > 0 else order_qty
            
            if order_qty > 0:
                cost_metrics['total_ordering_cost'] += cost_metrics['ordering_cost']
                
                if 'Factory' in supplier and 'Finished_Good' == df_lead_times[(df_lead_times['To_Location']==loc) & (df_lead_times['Item_ID']==item_id)]['Item_Type'].iloc[0]:
                    # This is a factory producing a finished good (not ordering a component)
                    factory_production_requests[(supplier, item_id)] = factory_production_requests.get((supplier, item_id), 0) + order_qty
                    simulation_events.append({
                        "Date": current_date,
                        "Type": "Production_Request",
                        "Item_ID": item_id,
                        "Location": loc, # Origin of the request
                        "Quantity": order_qty,
                        "Description": f"{loc} triggered production request of {order_qty} {item_id} at {supplier}."
                    })
                else: # It's an order to an external vendor (components for factory, or finished goods for DC)
                    arrival_date = current_date + timedelta(days=int(lead_time))
                    incoming_shipments.setdefault(arrival_date, {}).setdefault((loc, item_id), []).append(order_qty)
                    simulation_events.append({
                        "Date": current_date,
                        "Type": "Upstream_Reorder_Placed",
                        "Item_ID": item_id,
                        "Location": loc,
                        "Quantity": order_qty,
                        "Description": f"{loc} placed a reorder of {order_qty} {item_id} with {supplier}. Expected arrival: {arrival_date.strftime('%Y-%m-%d')}."
                    })


def _simulate_production(current_date, inventory_levels, factory_production_requests, bom_map, bom_quantity_map, 
                         simulation_events, bom_check, reorder_policy, cost_metrics, df_lead_times, df_forecast, incoming_shipments):
    """
    Processes production requests at factories.
    Also triggers component reorders if necessary after consumption.
    """
    
    # Create a copy of requests to avoid modifying dict during iteration
    requests_to_process = factory_production_requests.copy()
    factory_production_requests.clear() # Clear for next day's requests
    
    for (factory_loc, sku_to_produce), requested_qty in requests_to_process.items():
        if 'Factory' not in factory_loc:
            continue # Only factories produce

        can_produce_qty = requested_qty 

        if bom_check and sku_to_produce in bom_map:
            for component in bom_map[sku_to_produce]:
                qty_required_per_unit = bom_quantity_map.get((sku_to_produce, component), 0)
                total_qty_required = qty_required_per_unit * requested_qty
                
                current_component_stock = inventory_levels.get((factory_loc, component), 0)
                
                if current_component_stock < total_qty_required:
                    producible_by_component = current_component_stock // qty_required_per_unit if qty_required_per_unit > 0 else 0
                    can_produce_qty = min(can_produce_qty, producible_by_component)
                    
                    if can_produce_qty < requested_qty:
                        simulation_events.append({
                            "Date": current_date,
                            "Type": "Production_Constraint_Alert",
                            "Item_ID": sku_to_produce,
                            "Location": factory_loc,
                            "Quantity": requested_qty - can_produce_qty,
                            "Description": f"Production of {sku_to_produce} at {factory_loc} limited by component {component}. Only {can_produce_qty} units can be produced. Requested: {requested_qty}."
                        })
                        
        if can_produce_qty > 0:
            # Deduct components
            if bom_check and sku_to_produce in bom_map:
                for component in bom_map[sku_to_produce]:
                    qty_required_per_unit = bom_quantity_map.get((sku_to_produce, component), 0)
                    inventory_levels[(factory_loc, component)] -= (qty_required_per_unit * can_produce_qty)
                    
            # Add finished goods
            inventory_levels[(factory_loc, sku_to_produce)] = inventory_levels.get((factory_loc, sku_to_produce), 0) + can_produce_qty
            
            simulation_events.append({
                "Date": current_date,
                "Type": "Production_Completion",
                "Item_ID": sku_to_produce,
                "Location": factory_loc,
                "Quantity": can_produce_qty,
                "Description": f"Factory {factory_loc} produced {can_produce_qty} of {sku_to_produce}."
            })

            # After production, check reorder point for components at this factory
            if bom_check and sku_to_produce in bom_map:
                for component in bom_map[sku_to_produce]:
                    policy = reorder_policy.get((factory_loc, component))
                    if policy and inventory_levels.get((factory_loc, component), 0) <= policy['rop']:
                        lead_time = policy['lead_time']
                        min_order_qty = policy['min_order']
                        order_multiple = policy['order_mult']
                        supplier = policy['supplier']
                        ss = policy['ss']

                        # Simplified component demand: sum of avg daily production needs for all SKUs using this component
                        avg_comp_daily_consumption = 0
                        for p_sku in df_bom[df_bom['Component_ID'] == component]['Parent_SKU_ID'].unique():
                            qty_req_per_sku = df_bom[(df_bom['Parent_SKU_ID'] == p_sku) & (df_bom['Component_ID'] == component)]['Quantity_Required'].iloc[0]
                            # Assuming forecast for finished good represents potential production demand at factory
                            avg_fg_demand = df_forecast[(df_forecast['SKU_ID'] == p_sku) & (df_forecast['Location'] == factory_loc)]['Sales_Quantity'].mean()
                            if pd.isna(avg_fg_demand): avg_fg_demand = 0
                            avg_comp_daily_consumption += avg_fg_demand * qty_req_per_sku

                        raw_order_qty_comp = max(0, (avg_comp_daily_consumption * lead_time + ss) - inventory_levels.get((factory_loc, component), 0))

                        order_qty_comp = max(min_order_qty, raw_order_qty_comp)
                        order_qty_comp = math.ceil(order_qty_comp / order_multiple) * order_multiple if order_multiple > 0 else order_qty_comp

                        if order_qty_comp > 0:
                            cost_metrics['total_ordering_cost'] += cost_metrics['ordering_cost']
                            arrival_date_comp = current_date + timedelta(days=int(lead_time))
                            incoming_shipments.setdefault(arrival_date_comp, {}).setdefault((factory_loc, component), []).append(order_qty_comp)
                            simulation_events.append({
                                "Date": current_date,
                                "Type": "Component_Reorder_Placed",
                                "Item_ID": component,
                                "Location": factory_loc,
                                "Quantity": order_qty_comp,
                                "Description": f"Factory {factory_loc} reordered {order_qty_comp} of component {component} from {supplier}. Expected arrival: {arrival_date_comp.strftime('%Y-%m-%d')}."
                            })

def _calculate_daily_holding_cost(inventory_levels, holding_cost, cost_metrics):
    """Calculates daily holding cost based on end-of-day inventory."""
    for (location, item), stock in inventory_levels.items():
        if stock > 0: 
            cost_metrics['total_holding_cost'] += stock * holding_cost

@st.cache_data(show_spinner=False)
def run_full_simulation(
    df_sales, 
    df_inventory, 
    df_lead_times, 
    df_bom, 
    df_global_config, 
    start_date, 
    end_date, 
    safety_stock_method, 
    service_level, 
    bom_check,
    forecast_model,
    ma_window_size
):
    """
    Runs a time-series simulation of the entire supply chain network.
    Returns simulation results and the forecast DataFrame.
    Caches the output for performance.
    """
    st.info("Running full supply chain simulation...")

    # Initialize data structures
    inventory_levels = df_inventory.set_index(['Location', 'SKU_ID'])['Current_Stock'].to_dict()
    inventory_history = []
    simulation_events = []
    
    incoming_shipments = {} # Key: arrival_date -> {(location, item): [quantity, quantity]}
    factory_production_requests = {} # Key: (factory_loc, sku_id): quantity_to_produce

    # Get costs from global config
    holding_cost = df_global_config.loc[df_global_config['Parameter'] == 'Holding_Cost_Per_Unit_Per_Day', 'Value'].iloc[0]
    ordering_cost = df_global_config.loc[df_global_config['Parameter'] == 'Ordering_Cost_Per_Order', 'Value'].iloc[0]
    stockout_cost = df_global_config.loc[df_global_config['Parameter'] == 'Stockout_Cost_Per_Unit', 'Value'].iloc[0]

    # Initialize mutable cost metrics dictionary
    cost_metrics = {
        'total_holding_cost': 0.0,
        'total_ordering_cost': 0.0,
        'total_stockout_cost': 0.0,
        'total_sales_demand': 0,
        'total_lost_sales': 0,
        'holding_cost': holding_cost,
        'ordering_cost': ordering_cost,
        'stockout_cost': stockout_cost
    }
    
    # Pre-process lead times and BOM
    bom_map = df_bom.groupby('Parent_SKU_ID')['Component_ID'].apply(list).to_dict()
    bom_quantity_map = df_bom.set_index(['Parent_SKU_ID', 'Component_ID'])['Quantity_Required'].to_dict()

    # Generate forecast demand for the simulation period
    forecast_start_date = df_sales['Date'].max() + timedelta(days=1) if not df_sales.empty and not df_sales['Date'].empty else start_date
    forecast_end_date = end_date
    forecast_days = (forecast_end_date - forecast_start_date).days + 1
    
    df_forecast = pd.DataFrame()
    if forecast_days > 0 and not df_sales.empty: # Only forecast if there's historical sales data to learn from
        st.info(f"Generating a {forecast_days}-day demand forecast using {forecast_model}...")
        df_forecast = forecast_demand(df_sales, forecast_model, forecast_days, ma_window_size)
    elif df_sales.empty:
        st.warning("Cannot generate forecast as historical sales data is empty.")
    else:
        st.info("No future forecast period defined. Simulation runs only on historical data.")
        
    # Combine historical and forecast data for simulation
    # Ensure 'Sales_Quantity' column exists in df_sales before concatenating
    if 'Sales_Quantity' not in df_sales.columns and not df_sales.empty:
        df_sales['Sales_Quantity'] = 0 # Or handle appropriately if sales data might truly lack quantities

    df_sim_demand = pd.concat([df_sales, df_forecast]).sort_values('Date').reset_index(drop=True)
    
    # Pre-calculate Safety Stock and Reorder Points for all relevant item-location pairs
    reorder_policy = {} 
    # Iterate through all unique (To_Location, Item_ID) pairs from df_lead_times
    # These are the nodes that might need to place orders
    for idx, row in df_lead_times.iterrows():
        location = row['To_Location']
        item_id = row['Item_ID']
        lead_time_days = int(row['Lead_Time_Days']) # Ensure int
        min_order_qty = int(row['Min_Order_Quantity']) # Ensure int
        order_multiple = int(row['Order_Multiple']) # Ensure int
        from_location = row['From_Location'] # The supplier

        # Determine the relevant demand for SS/ROP calculation based on location type
        df_demand_for_ss = pd.DataFrame()
        if 'Store' in location:
            df_demand_for_ss = df_sim_demand[(df_sim_demand['SKU_ID'] == item_id) & (df_sim_demand['Location'] == location)]
        elif 'DC' in location:
            # For DCs, demand is driven by stores they supply. Sum up forecast demand from downstream.
            downstream_locations = df_lead_times[
                (df_lead_times['From_Location'] == location) & (df_lead_times['Item_ID'] == item_id)
            ]['To_Location'].unique()
            df_demand_for_ss = df_sim_demand[
                (df_sim_demand['SKU_ID'] == item_id) & (df_sim_demand['Location'].isin(downstream_locations))
            ]
        elif 'Factory' in location:
            item_type_at_factory = row['Item_Type'] # Is it a component or a finished good being supplied?
            if item_type_at_factory == 'Component':
                # Component demand depends on finished good production. Simplistic: tie to forecast of parent SKUs.
                parent_skus = df_bom[df_bom['Component_ID'] == item_id]['Parent_SKU_ID'].unique()
                component_demand_data = []
                for p_sku in parent_skus:
                    qty_req = df_bom[(df_bom['Parent_SKU_ID'] == p_sku) & (df_bom['Component_ID'] == item_id)]['Quantity_Required'].iloc[0]
                    # Assuming factory produces SKUs for overall forecast demand
                    sku_forecast = df_sim_demand[(df_sim_demand['SKU_ID'] == p_sku) & (df_sim_demand['Location'] == location)]
                    if not sku_forecast.empty:
                        component_demand_data.append(sku_forecast[['Date', 'Sales_Quantity']].copy())
                        component_demand_data[-1]['Sales_Quantity'] *= qty_req
                if component_demand_data:
                    df_demand_for_ss = pd.concat(component_demand_data)
                    df_demand_for_ss = df_demand_for_ss.groupby('Date')['Sales_Quantity'].sum().reset_index()
                
            else: # Factory supplying a Finished_Good
                # Demand from DCs it supplies.
                downstream_locations = df_lead_times[
                    (df_lead_times['From_Location'] == location) & (df_lead_times['Item_ID'] == item_id)
                ]['To_Location'].unique()
                df_demand_for_ss = df_sim_demand[
                    (df_sim_demand['SKU_ID'] == item_id) & (df_sim_demand['Location'].isin(downstream_locations))
                ]
            
        ss = 0
        if not df_demand_for_ss.empty:
            if safety_stock_method == "King's Method":
                ss = calculate_safety_stock_kings(df_demand_for_ss, lead_time_days, service_level / 100)
            else:
                ss = calculate_safety_stock_avg_max(df_demand_for_ss, lead_time_days)

        avg_daily_demand = df_demand_for_ss['Sales_Quantity'].mean() if not df_demand_for_ss.empty else 0
        rop = avg_daily_demand * lead_time_days + ss

        reorder_policy[(location, item_id)] = {
            'ss': ss,
            'rop': rop,
            'min_order': min_order_qty,
            'order_mult': order_multiple,
            'supplier': from_location,
            'lead_time': lead_time_days
        }

    # Simulation loop
    for current_date in pd.date_range(start=start_date, end=end_date, freq='D'):
        # 1. Process arrivals (incoming shipments)
        _process_arrivals(current_date, inventory_levels, incoming_shipments, simulation_events)

        # 2. Process daily demand at retail stores and trigger their reorders
        # This function directly schedules the shipment from the supplier into `incoming_shipments`
        _process_daily_sales_demand(current_date, inventory_levels, df_sim_demand, reorder_policy, 
                                    simulation_events, cost_metrics, df_forecast, incoming_shipments)

        # 3. Process reorders for DCs and Factories (for items they need to order from upstream)
        # This function also populates `factory_production_requests` if a factory needs to produce.
        _process_upstream_orders_and_reorders(current_date, inventory_levels, reorder_policy, 
                                              incoming_shipments, simulation_events, 
                                              cost_metrics, df_forecast, df_lead_times, 
                                              factory_production_requests)
        
        # 4. Simulate actual production at factories based on accumulated requests
        _simulate_production(current_date, inventory_levels, factory_production_requests, bom_map, 
                             bom_quantity_map, simulation_events, bom_check, reorder_policy, 
                             cost_metrics, df_lead_times, df_forecast, incoming_shipments) # Pass all needed params


        # 5. Calculate daily holding costs
        _calculate_daily_holding_cost(inventory_levels, holding_cost, cost_metrics)
            
        # 6. Record inventory levels for plotting
        for (location, item), stock in inventory_levels.items():
            inventory_history.append({
                "Date": current_date,
                "Location": location,
                "Item_ID": item,
                "Stock": stock
            })
            
    # Finalize results
    df_inventory_history = pd.DataFrame(inventory_history)
    df_simulation_events = pd.DataFrame(simulation_events)

    return {
        "df_inventory_history": df_inventory_history,
        "df_simulation_events": df_simulation_events,
        "total_holding_cost": cost_metrics['total_holding_cost'],
        "total_ordering_cost": cost_metrics['total_ordering_cost'],
        "total_stockout_cost": cost_metrics['total_stockout_cost'],
        "total_sales_demand": cost_metrics['total_sales_demand'],
        "total_lost_sales": cost_metrics['total_lost_sales'],
        "df_forecast": df_forecast,
        "latest_inventory_levels": inventory_levels,
        "precalculated_reorder_policy": reorder_policy # Return for use in indent suggestions
    }


# --- UI Layout ---
st.set_page_config(layout="wide", page_title="Supply Chain App")

st.title("Advanced Supply Chain Intelligence App")
st.markdown("Analyze demand, simulate your supply chain, and optimize inventory policy with custom data.")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# Data Source Selection
data_source = st.sidebar.radio("Select Data Source", ("Generated Sample Data", "Upload Custom Data"))

uploaded_data = {}
validation_passed = True # Flag to track overall data validation status

if data_source == "Upload Custom Data":
    st.sidebar.subheader("Upload Your CSV Files")
    sample_data = generate_realistic_data(DEFAULT_NUM_SKUS, DEFAULT_NUM_COMPONENTS_PER_SKU, 1, 1, 1, "Multi-Echelon")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Download Templates")
    required_files_info = {
        "sales_data.csv": SALES_SCHEMA, 
        "inventory_data.csv": INVENTORY_SCHEMA, 
        "lead_times_data.csv": LEAD_TIMES_SCHEMA, 
        "bom_data.csv": BOM_SCHEMA
    }
    optional_files_info = {
        "actual_orders.csv": ACTUAL_ORDERS_SCHEMA, 
        "actual_shipments.csv": ACTUAL_SHIPMENTS_SCHEMA, 
        "global_config.csv": GLOBAL_CONFIG_SCHEMA
    }
    
    for filename in required_files_info.keys():
        if filename in sample_data:
            st.sidebar.markdown(get_csv_download_link(sample_data[filename], filename), unsafe_allow_html=True)
    for filename in optional_files_info.keys():
        if filename in sample_data:
            st.sidebar.markdown(get_csv_download_link(sample_data[filename], filename), unsafe_allow_html=True)
    st.sidebar.markdown("---")

    st.sidebar.markdown("#### Upload Your Files")
    all_files_info = {**required_files_info, **optional_files_info}
    
    for filename, schema in all_files_info.items():
        is_optional = filename in optional_files_info
        label = f"{filename} (Optional)" if is_optional else filename
        
        uploaded_file = st.sidebar.file_uploader(label, type="csv", key=filename)
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if not _validate_dataframe_schema(df, schema, filename):
                    validation_passed = False
                uploaded_data[filename] = df
            except Exception as e:
                st.error(f"Error loading or parsing {filename}: {e}")
                uploaded_data[filename] = pd.DataFrame()
                validation_passed = False
        elif not is_optional:
            st.warning(f"Required file '{filename}' is missing.")
            uploaded_data[filename] = pd.DataFrame()
            validation_passed = False
            
    # Check if all REQUIRED files are present and valid
    if not all(filename in uploaded_data and not uploaded_data[filename].empty for filename in required_files_info.keys()):
        validation_passed = False # Set to false if any required file is missing or empty
        st.sidebar.warning("Please upload all required files to run the simulation.")
        
    run_simulation_button = st.sidebar.button("Run Simulation with Uploaded Data", disabled=not validation_passed)

else: # Generated Sample Data
    st.sidebar.subheader("Sample Data Configuration")
    network_type = st.sidebar.selectbox("Supply Chain Network Type", ["Multi-Echelon", "Single-Echelon"])
    num_factories = st.sidebar.slider("Number of Factories", 1, 5, 2 if network_type == "Multi-Echelon" else 1)
    num_dcs = st.sidebar.slider("Number of Distribution Centers", 1, 5, 2) if network_type == "Multi-Echelon" else 0
    num_stores = st.sidebar.slider("Number of Retail Stores", 1, 10, 5)
    
    run_simulation_button = st.sidebar.button("Run Simulation with Sample Data")

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Parameters")
simulation_days = st.sidebar.slider("Simulation Duration (days)", 30, 365, 90)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecasting Parameters")
forecast_model = st.sidebar.selectbox("Forecasting Model", ["Moving Average", "Moving Median", "Random Forest", "XGBoost"])

if forecast_model in ["Moving Average", "Moving Median"]:
    ma_window_size = st.sidebar.slider("Moving Average/Median Window Size (days)", 1, 30, 7)
else:
    ma_window_size = 7 # Default value if not using MA/MM, though not strictly used by ML models

st.sidebar.markdown("---")
st.sidebar.subheader("Inventory Policy")
safety_stock_method = st.sidebar.selectbox("Safety Stock Method", ["King's Method", "Avg Max Method"])
service_level = st.sidebar.slider("Desired Service Level (%)", min_value=70, max_value=99, value=95) if safety_stock_method == "King's Method" else None

st.sidebar.markdown("---")
bom_check = st.sidebar.checkbox("BOM Check (Enable for production constraints)", value=True)

# Main content
if run_simulation_button and validation_passed: # Only run if validation passed for uploaded data
    st.header("Simulation Results")
    
    with st.spinner("Preparing and running simulation... This may take a moment."):
        if data_source == "Generated Sample Data":
            st.success("Using generated sample data.")
            all_dfs = generate_realistic_data(DEFAULT_NUM_SKUS, DEFAULT_NUM_COMPONENTS_PER_SKU, num_factories, num_dcs, num_stores, network_type)
        else:
            st.success("Using uploaded custom data.")
            all_dfs = uploaded_data
        
        # Load data from the dictionary, providing empty DFs for optional missing ones
        df_sales = all_dfs.get("sales_data.csv", pd.DataFrame())
        df_inventory = all_dfs.get("inventory_data.csv", pd.DataFrame())
        df_lead_times = all_dfs.get("lead_times_data.csv", pd.DataFrame())
        df_bom = all_dfs.get("bom_data.csv", pd.DataFrame())
        df_global_config = all_dfs.get("global_config.csv", pd.DataFrame({
            "Parameter": ["Holding_Cost_Per_Unit_Per_Day", "Ordering_Cost_Per_Order", "Stockout_Cost_Per_Unit"],
            "Value": [0.05, 50.0, 10.0] # Default values if global_config.csv is missing
        }))
        df_actual_orders = all_dfs.get("actual_orders.csv", pd.DataFrame())
        df_actual_shipments = all_dfs.get("actual_shipments.csv", pd.DataFrame())

        # Determine simulation start and end dates
        sim_start_date = df_sales['Date'].min() if not df_sales.empty and not df_sales['Date'].empty else DEFAULT_START_DATE
        sim_end_date = (df_sales['Date'].max() if not df_sales.empty and not df_sales['Date'].empty else DEFAULT_START_DATE) + timedelta(days=simulation_days)

        # Run the full simulation
        simulation_results = run_full_simulation(
            df_sales,
            df_inventory,
            df_lead_times,
            df_bom,
            df_global_config,
            sim_start_date,
            sim_end_date,
            safety_stock_method,
            service_level,
            bom_check,
            forecast_model,
            ma_window_size
        )
        
        st.markdown("---")
        st.subheader("Simulation Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        
        total_cost = simulation_results['total_holding_cost'] + simulation_results['total_ordering_cost'] + simulation_results['total_stockout_cost']
        service_level_perc = 100 * (1 - (simulation_results['total_lost_sales'] / simulation_results['total_sales_demand'])) if simulation_results['total_sales_demand'] > 0 else 100
        
        col1.metric("Total Cost", f"${total_cost:,.2f}")
        col2.metric("Total Lost Sales", f"{simulation_results['total_lost_sales']:,}")
        col3.metric("Service Level", f"{service_level_perc:,.2f}%")
        
        with st.expander("Cost Breakdown"):
            st.write(f"**Total Holding Cost:** ${simulation_results['total_holding_cost']:,.2f}")
            st.write(f"**Total Ordering Cost:** ${simulation_results['total_ordering_cost']:,.2f}")
            st.write(f"**Total Stockout Cost:** ${simulation_results['total_stockout_cost']:,.2f}")

        st.markdown("---")
        
        st.subheader("Inventory Levels Over Time")
        df_plot_inv_hist = simulation_results['df_inventory_history']
        all_locations_inv = df_plot_inv_hist['Location'].unique()
        all_items_inv = df_plot_inv_hist['Item_ID'].unique()
        
        selected_location_inv = st.selectbox("Select Location for Inventory", all_locations_inv)
        selected_item_inv = st.selectbox("Select SKU/Component for Inventory", all_items_inv)
        
        df_plot_inv = df_plot_inv_hist[(df_plot_inv_hist['Location'] == selected_location_inv) & (df_plot_inv_hist['Item_ID'] == selected_item_inv)]
        
        if not df_plot_inv.empty:
            fig_inv = px.line(df_plot_inv, x="Date", y="Stock", title=f"Inventory Level for {selected_item_inv} at {selected_location_inv}")
            st.plotly_chart(fig_inv, use_container_width=True)
        else:
            st.info("No inventory data to display for the selected item and location.")
        
        st.markdown("---")
        st.subheader("Demand Forecasts")
        df_forecast = simulation_results['df_forecast']
        if not df_forecast.empty:
            all_locations_forecast = df_forecast['Location'].unique()
            all_items_forecast = df_forecast['SKU_ID'].unique()

            selected_location_forecast = st.selectbox("Select Location for Forecast", all_locations_forecast)
            selected_item_forecast = st.selectbox("Select SKU/Component for Forecast", all_items_forecast)
            
            df_plot_forecast = df_forecast[
                (df_forecast['Date'] >= sim_start_date) & # Show forecast only from simulation start date
                (df_forecast['Location'] == selected_location_forecast) & 
                (df_forecast['SKU_ID'] == selected_item_forecast)
            ]
            
            if not df_plot_forecast.empty:
                # Combine historical sales with forecast for a richer plot
                df_historical_sales_for_plot = df_sales[
                    (df_sales['Location'] == selected_location_forecast) & 
                    (df_sales['SKU_ID'] == selected_item_forecast)
                ].copy()
                df_historical_sales_for_plot['Type'] = 'Historical Sales'
                df_plot_forecast['Type'] = 'Forecasted Demand'
                
                df_combined_demand = pd.concat([df_historical_sales_for_plot.rename(columns={'Sales_Quantity': 'Demand'}), 
                                                df_plot_forecast.rename(columns={'Sales_Quantity': 'Demand'})])
                
                fig_forecast = px.line(df_combined_demand, x="Date", y="Demand", color='Type', 
                                       title=f"Demand for {selected_item_forecast} at {selected_location_forecast}",
                                       markers=True)
                
                # FIX: Add check for df_sales and its 'Date' column before adding vline
                if not df_sales.empty and 'Date' in df_sales.columns and not df_sales['Date'].empty:
                    fig_forecast.add_vline(x=df_sales['Date'].max(), line_width=2, line_dash="dash", line_color="red", annotation_text="Forecast Start")
                else:
                    st.warning("Cannot show forecast start line as historical sales data is empty or missing dates.")
                
                st.plotly_chart(fig_forecast, use_container_width=True)
            else:
                st.info("No forecast data to display for the selected item and location.")
        else:
            st.info("No demand forecast was generated for the simulation period.")


        st.markdown("---")
        st.subheader("Suggested Indent Orders")
        # Get the latest inventory levels from the simulation results
        latest_inventory_levels_dict = simulation_results['latest_inventory_levels']
        
        # Use the end date of the simulation as the current date for order suggestion
        current_date_for_indent = sim_end_date 

        df_indent_suggestions = suggest_indent_orders(
            latest_inventory_levels_dict, 
            df_forecast, 
            df_lead_times, 
            safety_stock_method, 
            service_level,
            current_date_for_indent,
            simulation_results['precalculated_reorder_policy'] # Pass precalculated policy
        )
        
        if not df_indent_suggestions.empty:
            st.dataframe(df_indent_suggestions, use_container_width=True)

            st.markdown("##### Summary of Suggested Orders by Type")
            order_type_summary = df_indent_suggestions.groupby('Order_Type')['Suggested_Order_Quantity'].sum().reset_index()
            fig_order_type = px.bar(order_type_summary, x='Order_Type', y='Suggested_Order_Quantity', 
                                    title='Total Suggested Orders by Type')
            st.plotly_chart(fig_order_type, use_container_width=True)

            st.markdown("##### Summary of Suggested Orders by Supplier")
            supplier_summary = df_indent_suggestions.groupby('Supplier_Location')['Suggested_Order_Quantity'].sum().reset_index()
            fig_supplier = px.bar(supplier_summary, x='Supplier_Location', y='Suggested_Order_Quantity', 
                                  title='Total Suggested Orders by Supplier')
            st.plotly_chart(fig_supplier, use_container_width=True)

            st.warning("Note on Shelf Life and Max Capacity: The current order suggestion logic does not explicitly account for SKU shelf life expiration or maximum warehouse/storage capacity. These factors would require more complex inventory tracking and constraint modeling.")

        else:
            st.info("No indent orders suggested based on current inventory and forecast. Inventory levels are sufficient.")

        st.markdown("---")
        st.subheader("Detailed Simulation Events & Alerts")
        with st.expander("View Event Log"):
            if not simulation_results['df_simulation_events'].empty:
                st.dataframe(simulation_results['df_simulation_events'], use_container_width=True)
            else:
                st.info("No events to display.")

        # New: Export Results Section
        st.markdown("---")
        st.subheader("📥 Export Results")
        with st.expander("Download Simulation Data"):
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.markdown(get_csv_download_link(simulation_results['df_inventory_history'], "inventory_history.csv"), unsafe_allow_html=True)
            with col_dl2:
                st.markdown(get_csv_download_link(simulation_results['df_simulation_events'], "simulation_events.csv"), unsafe_allow_html=True)


else: # Initial load or if validation failed for uploaded data
    st.info("Configure your parameters in the sidebar and click 'Run Simulation' to begin.")
    st.subheader("Data Table Schemas")
    st.markdown("""
        The application relies on several CSV files for historical data and configuration. You can upload your own data based on these templates. A sample data template is available for download in the sidebar.
        
        **Required Files:**
        - `sales_data.csv`
        - `inventory_data.csv`
        - `lead_times_data.csv`
        - `bom_data.csv`

        **Optional Files:**
        - `actual_orders.csv`
        - `actual_shipments.csv`
        - `global_config.csv`
        
    """)

st.markdown("---")
st.header("🚀 How the Simulation Works")
st.markdown("""
This application runs a comprehensive, daily simulation of your entire supply chain. It's a closed-loop system where each day's events affect the next. Here's a simple breakdown of the process:

1.  **Initial Setup 📊:**
    - The simulation starts by loading all your data (sales, inventory, lead times, costs, and BOM).
    - It then generates a demand forecast for the future period using your selected model.
    - Finally, it pre-calculates the **Reorder Points (ROP)** and **Safety Stock (SS)** for every item at every location, based on historical demand and your desired service level. These are the key inventory policy levers.

2.  **Daily Simulation Loop (1 day at a time) 🗓️:**
    - **Step A: Arrivals:** Any shipments scheduled to arrive on the current day are added to the inventory of their destination location.
    - **Step B: Customer Demand:** The simulation processes customer demand at the retail stores. It fulfills what it can from current stock and logs any lost sales (stockouts).
    - **Step C: Reorders:** For every location (retail store, DC, or factory), the app checks if the inventory of any item has dropped below its pre-calculated Reorder Point. If so, it places an order with the upstream supplier, which will arrive after the specified lead time.
    - **Step D: Production:** If a factory receives a request to produce a finished good, it checks its inventory for the necessary components (based on the BOM). It then consumes the components to produce the finished goods, adding them to its inventory.
    - **Step E: Cost Calculation:** At the end of each day, the simulation calculates and aggregates daily costs: a holding cost for every item in inventory and a stockout cost for every lost sale.

3.  **Final Results and Insights ✨:**
    - After the simulation is complete, all the daily data is aggregated.
    - The application presents key performance indicators like **Total Cost**, **Service Level**, and **Total Lost Sales**.
    - It generates interactive charts to visualize **inventory levels** and **demand forecasts** over the entire period.
    - Finally, it provides a list of **suggested orders** based on the final day's inventory position and the forward-looking forecast.
""")

st.markdown("---")
st.header("❓ Frequently Asked Questions")
with st.expander("How does the simulation work?"):
    st.markdown("""
        The simulation runs daily, modeling the entire supply chain from customer demand to factory production.
        
        1.  **Demand Fulfillment:** It starts at the retail store level, fulfilling customer demand for each SKU from available stock.
        2.  **Reorder Logic:** If a location's inventory (store, DC, or factory for components/finished goods) falls below its reorder point, it places an order with its upstream supplier.
        3.  **Upstream Demand:** These reorders create demand for upstream nodes (DCs and Factories).
        4.  **Production:** Factories process production requests, consuming components based on the Bill of Materials (BOM) if enabled. Production might be constrained by component availability.
    """)

with st.expander("What does the BOM check do?"):
    st.markdown("""
        When enabled, the `Bill of Materials` (BOM) check adds a crucial layer of realism to the simulation.
        A factory will only be able to fulfill a production request if it has all the necessary components for that SKU in its inventory. If a component is missing, the production quantity will be limited, which can lead to a delayed fulfillment and a potential stockout at the downstream location. This accurately reflects real-world production constraints.
    """)

with st.expander("How is the cost data used?"):
    st.markdown("""
        The application uses the cost data from `global_config.csv` to calculate the total supply chain cost for the simulation period.
        * **Total Holding Cost:** Calculated daily for every unit of inventory in the network.
        * **Total Ordering Cost:** A fixed cost is added for every order placed (e.g., a reorder from a store to a DC).
        * **Total Stockout Cost:** In the event of a stockout, a penalty cost is applied for every unit of unfulfilled demand.
    """)
    
with st.expander("What are the different forecasting models?"):
    st.markdown("""
        The app offers four methods for forecasting future demand based on historical sales data:
        * **Moving Average:** A simple model that predicts the next period's demand by taking the average of sales over a specified historical window.
        * **Moving Median:** Similar to the moving average, but uses the median value, which can be more robust to outliers in sales data.
        * **Random Forest:** A powerful machine aine learning model that uses an ensemble of decision trees to predict future demand based on patterns in your historical data.
        * **XGBoost:** Another advanced machine learning model, known for its performance and speed. It uses a gradient boosting framework to build a robust predictive model.
    """)
    
with st.expander("What data inputs are required?"):
    st.markdown("""
        The application requires four primary CSV files to run:
        * **`sales_data.csv`**: Contains historical sales transactions, including `Date`, `SKU_ID`, `Location`, and `Sales_Quantity`.
        * **`inventory_data.csv`**: Contains the initial `Current_Stock` levels for each `SKU_ID` and `Location` at the simulation's start date.
        * **`lead_times_data.csv`**: Defines the supply chain network, including `From_Location`, `To_Location`, `Item_ID`, `Lead_Time_Days`, and order parameters like `Min_Order_Quantity`.
        * **`bom_data.csv`**: The Bill of Materials, which links parent `SKU_ID`s to their `Component_ID`s and `Quantity_Required`.
        
        Optional files include `actual_orders.csv`, `actual_shipments.csv`, and `global_config.csv` for calculating historical fill rate and configuring costs.
    """)