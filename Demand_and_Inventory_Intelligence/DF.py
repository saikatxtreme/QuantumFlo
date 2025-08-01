# DFv13.py - Demand and Inventory Intelligence Streamlit App
# Features: Full multi-echelon simulation, detailed cost analysis, BOM integration,
#           comprehensive reporting, and now includes a forecast model selector and FAQ.

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

@st.cache_data
def generate_realistic_data(num_skus, num_components, num_factories, num_dcs, num_stores, network_type):
    """
    Generates a complete set of realistic dummy dataframes for the app.
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

def calculate_safety_stock_kings(df_demand, lead_time_days, service_level):
    """
    Calculates safety stock using King's method.
    """
    if df_demand.empty:
        return 0
    
    df_demand['Date'] = pd.to_datetime(df_demand['Date'])
    df_daily_demand = df_demand.groupby('Date')['Sales_Quantity'].sum().reset_index()
    std_dev_demand = df_daily_demand['Sales_Quantity'].std()
    
    z_score = norm.ppf(service_level)
    safety_stock = z_score * math.sqrt(lead_time_days) * std_dev_demand
    return max(0, int(safety_stock))

def calculate_safety_stock_avg_max(df_demand, lead_time_days):
    """
    Calculates safety stock using the Avg Max method.
    """
    if df_demand.empty:
        return 0

    df_demand['Date'] = pd.to_datetime(df_demand['Date'])
    df_daily_demand = df_demand.groupby('Date')['Sales_Quantity'].sum().reset_index()
    
    max_daily_demand = df_daily_demand['Sales_Quantity'].max()
    avg_daily_demand = df_daily_demand['Sales_Quantity'].mean()
    
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

@st.cache_data
def forecast_demand(df_sales, forecast_model, forecast_days, window_size=7):
    """
    Generates a forecast for future sales demand using the selected model.
    """
    df_sales['Date'] = pd.to_datetime(df_sales['Date'])
    df_daily_demand = df_sales.groupby(['Date', 'SKU_ID', 'Location'])['Sales_Quantity'].sum().reset_index()
    
    forecast_results = []
    
    # Placeholder for different forecasting models
    unique_skus = df_daily_demand['SKU_ID'].unique()
    unique_locations = df_daily_demand['Location'].unique()
    
    for sku in unique_skus:
        for location in unique_locations:
            df_subset = df_daily_demand[(df_daily_demand['SKU_ID'] == sku) & (df_daily_demand['Location'] == location)].set_index('Date').sort_index()
            df_subset = df_subset.asfreq('D', fill_value=0)
            
            last_date = df_subset.index.max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
            
            if forecast_model == "Moving Average":
                df_subset['Moving_Average'] = df_subset['Sales_Quantity'].rolling(window=window_size).mean()
                last_avg = df_subset['Moving_Average'].iloc[-1] if not df_subset['Moving_Average'].isnull().all() else 0
                forecast_values = [last_avg] * forecast_days
            elif forecast_model == "Moving Median":
                df_subset['Moving_Median'] = df_subset['Sales_Quantity'].rolling(window=window_size).median()
                last_median = df_subset['Moving_Median'].iloc[-1] if not df_subset['Moving_Median'].isnull().all() else 0
                forecast_values = [last_median] * forecast_days
            elif forecast_model in ["Random Forest", "XGBoost"]:
                # Simplistic feature engineering for ML models
                df_subset['dayofweek'] = df_subset.index.dayofweek
                df_subset['dayofyear'] = df_subset.index.dayofyear
                
                features = ['dayofweek', 'dayofyear']
                target = 'Sales_Quantity'
                
                # Split data
                if len(df_subset) > 100:
                    train_size = int(len(df_subset) * 0.8)
                    train_data = df_subset.iloc[:train_size]
                    
                    X_train, y_train = train_data[features], train_data[target]
                    
                    if forecast_model == "Random Forest":
                        model = RandomForestRegressor()
                    else:
                        model = XGBRegressor()
                        
                    model.fit(X_train, y_train)
                    
                    future_df = pd.DataFrame(index=future_dates)
                    future_df['dayofweek'] = future_df.index.dayofweek
                    future_df['dayofyear'] = future_df.index.dayofyear
                    
                    forecast_values = model.predict(future_df[features])
                else:
                    forecast_values = [0] * forecast_days
            
            for date, forecast_qty in zip(future_dates, forecast_values):
                forecast_results.append({
                    'Date': date,
                    'SKU_ID': sku,
                    'Location': location,
                    'Sales_Quantity': max(0, int(forecast_qty))
                })
    
    return pd.DataFrame(forecast_results)

@st.cache_data
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
    forecast_window_size # Added forecast_window_size
):
    """
    Runs a time-series simulation of the entire supply chain network.
    """
    st.info("Running full supply chain simulation...")

    # Initialize data structures
    all_locations = df_inventory['Location'].unique()
    all_items = df_inventory['SKU_ID'].unique()
    
    inventory_levels = df_inventory.set_index(['Location', 'SKU_ID'])['Current_Stock'].to_dict()
    inventory_history = []
    simulation_events = []
    
    # Store incoming shipments
    incoming_shipments = {} # Key: (location, item), Value: list of (arrival_date, quantity)
    
    # Get costs from global config
    holding_cost = df_global_config.loc[df_global_config['Parameter'] == 'Holding_Cost_Per_Unit_Per_Day', 'Value'].iloc[0]
    ordering_cost = df_global_config.loc[df_global_config['Parameter'] == 'Ordering_Cost_Per_Order', 'Value'].iloc[0]
    stockout_cost = df_global_config.loc[df_global_config['Parameter'] == 'Stockout_Cost_Per_Unit', 'Value'].iloc[0]

    # Initialize costs
    total_holding_cost = 0
    total_ordering_cost = 0
    total_stockout_cost = 0
    total_sales_demand = 0
    total_lost_sales = 0
    
    # Pre-process lead times and BOM
    lead_times_map = df_lead_times.set_index(['To_Location', 'Item_ID'])[['From_Location', 'Lead_Time_Days', 'Min_Order_Quantity', 'Order_Multiple']].to_dict('index')
    bom_map = df_bom.groupby('Parent_SKU_ID')['Component_ID'].apply(list).to_dict()
    bom_quantity_map = df_bom.set_index(['Parent_SKU_ID', 'Component_ID'])['Quantity_Required'].to_dict()

    # Pre-process sales data for quick lookup
    df_sales['Date'] = pd.to_datetime(df_sales['Date'])
    
    # Generate forecast demand for the simulation period
    forecast_start_date = df_sales['Date'].max() + timedelta(days=1)
    forecast_end_date = end_date
    forecast_days = (forecast_end_date - forecast_start_date).days + 1
    
    df_forecast = pd.DataFrame() # Initialize df_forecast
    if forecast_days > 0:
        st.info(f"Generating a {forecast_days}-day demand forecast using {forecast_model}...")
        df_forecast = forecast_demand(df_sales, forecast_model, forecast_days, forecast_window_size) # Pass window_size
    else:
        st.info("No forecast needed as simulation period is within historical data.")
        
    
    # Combine historical and forecast data for simulation
    df_sim_demand = pd.concat([df_sales, df_forecast]).sort_values('Date').reset_index(drop=True)
    sim_demand_by_date = df_sim_demand.groupby('Date')

    # Simulation loop
    for current_date in pd.date_range(start=start_date, end=end_date, freq='D'):
        # 1. Update inventory from incoming shipments
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
        
        # 2. Process demand at the retail store level (bottom-up approach)
        demand_for_today = {} # Key: (location, item), Value: quantity
        if current_date in sim_demand_by_date.groups:
            daily_demand_df = sim_demand_by_date.get_group(current_date)
            for _, row in daily_demand_df.iterrows():
                location = row['Location']
                sku = row['SKU_ID']
                demand_qty = row['Sales_Quantity']
                
                total_sales_demand += demand_qty
                
                # Check for inventory and fulfill demand
                current_stock = inventory_levels.get((location, sku), 0)
                shipped_qty = min(current_stock, demand_qty)
                lost_sales = demand_qty - shipped_qty
                
                inventory_levels[(location, sku)] = current_stock - shipped_qty
                total_lost_sales += lost_sales
                total_stockout_cost += lost_sales * stockout_cost
                
                simulation_events.append({
                    "Date": current_date,
                    "Type": "Sales_Demand",
                    "Item_ID": sku,
                    "Location": location,
                    "Quantity": demand_qty,
                    "Description": f"Customer demand for {demand_qty} {sku} at {location}. {shipped_qty} fulfilled, {lost_sales} lost sales."
                })
                
                # Check reorder point for stores
                lead_time_df_store = df_lead_times[(df_lead_times['To_Location'] == location) & (df_lead_times['Item_ID'] == sku)]
                if not lead_time_df_store.empty:
                    lead_time = lead_time_df_store.iloc[0]['Lead_Time_Days']
                    
                    # Calculate reorder point and order quantity
                    if safety_stock_method == "King's Method":
                        ss = calculate_safety_stock_kings(df_sim_demand[(df_sim_demand['SKU_ID'] == sku) & (df_sim_demand['Location'] == location)], lead_time, service_level / 100)
                    else:
                        ss = calculate_safety_stock_avg_max(df_sim_demand[(df_sim_demand['SKU_ID'] == sku) & (df_sim_demand['Location'] == location)], lead_time)
                    
                    # Assume ROP = Daily Avg Demand * Lead Time + Safety Stock
                    avg_daily_demand = df_sim_demand[(df_sim_demand['SKU_ID'] == sku) & (df_sim_demand['Location'] == location)]['Sales_Quantity'].mean() or 0
                    reorder_point = avg_daily_demand * lead_time + ss
                    
                    if inventory_levels.get((location, sku), 0) <= reorder_point:
                        min_order_qty = lead_time_df_store.iloc[0]['Min_Order_Quantity']
                        order_multiple = lead_time_df_store.iloc[0]['Order_Multiple']
                        
                        order_qty = max(min_order_qty, (avg_daily_demand * lead_time) + ss)
                        order_qty = math.ceil(order_qty / order_multiple) * order_multiple
                        
                        supplier = lead_time_df_store.iloc[0]['From_Location']
                        arrival_date = current_date + timedelta(days=int(lead_time))
                        
                        # Place order with the upstream location
                        demand_for_today[(supplier, sku)] = demand_for_today.get((supplier, sku), 0) + order_qty
                        total_ordering_cost += ordering_cost
                        
                        simulation_events.append({
                            "Date": current_date,
                            "Type": "Reorder_Placed",
                            "Item_ID": sku,
                            "Location": location,
                            "Quantity": order_qty,
                            "Description": f"Reorder of {order_qty} {sku} placed with {supplier}. Expected arrival: {arrival_date.strftime('%Y-%m-%d')}."
                        })
                
        # 3. Process demand at DCs (from store orders)
        for (location, item), demand_qty in demand_for_today.items():
            if 'DC' in location:
                # Check for inventory at DC and fulfill order
                current_stock = inventory_levels.get((location, item), 0)
                shipped_qty = min(current_stock, demand_qty)
                
                inventory_levels[(location, item)] = current_stock - shipped_qty
                
                # Schedule shipment
                lead_time_df_dc = df_lead_times[(df_lead_times['From_Location'] == location) & (df_lead_times['Item_ID'] == item)]
                if not lead_time_df_dc.empty:
                    lead_time = lead_time_df_dc.iloc[0]['Lead_Time_Days']
                    destination_location = lead_time_df_dc.iloc[0]['To_Location'] # This logic is simplistic, assuming one-to-one
                    arrival_date = current_date + timedelta(days=int(lead_time))
                    
                    incoming_shipments.setdefault(arrival_date, {}).setdefault((destination_location, item), []).append(shipped_qty)

                simulation_events.append({
                    "Date": current_date,
                    "Type": "DC_Order_Fulfillment",
                    "Item_ID": item,
                    "Location": location,
                    "Quantity": shipped_qty,
                    "Description": f"DC {location} fulfilled {shipped_qty} of order for {item}."
                })
                
                # Check reorder point for DCs
                lead_time_df_dc_up = df_lead_times[(df_lead_times['To_Location'] == location) & (df_lead_times['Item_ID'] == item)]
                if not lead_time_df_dc_up.empty:
                    lead_time_up = lead_time_df_dc_up.iloc[0]['Lead_Time_Days']
                    
                    # Assume ROP logic for DCs is similar to stores but based on aggregate demand
                    avg_daily_demand_dc = demand_qty # This is a simplification
                    reorder_point = avg_daily_demand_dc * lead_time_up
                    
                    if inventory_levels.get((location, item), 0) <= reorder_point:
                        min_order_qty = lead_time_df_dc_up.iloc[0]['Min_Order_Quantity']
                        order_multiple = lead_time_df_dc_up.iloc[0]['Order_Multiple']
                        
                        order_qty = max(min_order_qty, (avg_daily_demand_dc * lead_time_up) + random.randint(50, 100))
                        order_qty = math.ceil(order_qty / order_multiple) * order_multiple
                        
                        supplier = lead_time_df_dc_up.iloc[0]['From_Location']
                        arrival_date = current_date + timedelta(days=int(lead_time_up))
                        
                        # We'll just add it to incoming shipments for now as factories have different logic
                        
                        incoming_shipments.setdefault(arrival_date, {}).setdefault((location, item), []).append(order_qty)
                        total_ordering_cost += ordering_cost
                        
                        simulation_events.append({
                            "Date": current_date,
                            "Type": "DC_Reorder",
                            "Item_ID": item,
                            "Location": location,
                            "Quantity": order_qty,
                            "Description": f"DC {location} placed a reorder of {order_qty} {item} with {supplier}. Expected arrival: {arrival_date.strftime('%Y-%m-%d')}."
                        })
        
        # 4. Process factory production (based on orders from DCs)
        for (location, item), demand_qty in demand_for_today.items():
            if 'Factory' in location:
                # Check BOM if enabled
                can_produce = True
                if bom_check and item in bom_map:
                    for component in bom_map[item]:
                        qty_needed = bom_quantity_map.get((item, component), 0) * demand_qty
                        if inventory_levels.get((location, component), 0) < qty_needed:
                            can_produce = False
                            simulation_events.append({
                                "Date": current_date,
                                "Type": "Production_Hold",
                                "Item_ID": item,
                                "Location": location,
                                "Quantity": demand_qty,
                                "Description": f"Production of {item} held at {location} due to insufficient component {component}. Needed: {qty_needed}, In Stock: {inventory_levels.get((location, component), 0)}."
                            })
                            break
                
                if can_produce:
                    # Deduct components from inventory
                    if item in bom_map:
                        for component in bom_map[item]:
                            qty_needed = bom_quantity_map.get((item, component), 0) * demand_qty
                            inventory_levels[(location, component)] = inventory_levels.get((location, component), 0) - qty_needed
                            
                    # Add finished goods to inventory
                    inventory_levels[(location, item)] = inventory_levels.get((location, item), 0) + demand_qty
                    
                    simulation_events.append({
                        "Date": current_date,
                        "Type": "Production_Run",
                        "Item_ID": item,
                        "Location": location,
                        "Quantity": demand_qty,
                        "Description": f"Factory {location} produced {demand_qty} of {item}."
                    })
        
        # 5. Calculate daily costs
        for (location, item), stock in inventory_levels.items():
            total_holding_cost += stock * holding_cost
            
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
        "total_holding_cost": total_holding_cost,
        "total_ordering_cost": total_ordering_cost,
        "total_stockout_cost": total_stockout_cost,
        "total_sales_demand": total_sales_demand,
        "total_lost_sales": total_lost_sales,
        "df_forecast": df_forecast # Return the forecast dataframe
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
if data_source == "Upload Custom Data":
    st.sidebar.subheader("Upload Your CSV Files")
    sample_data = generate_realistic_data(DEFAULT_NUM_SKUS, DEFAULT_NUM_COMPONENTS_PER_SKU, 1, 1, 1, "Multi-Echelon")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Download Templates")
    required_files = ["sales_data.csv", "inventory_data.csv", "lead_times_data.csv", "bom_data.csv"]
    optional_files = ["actual_orders.csv", "actual_shipments.csv", "global_config.csv"]
    
    for filename in required_files + optional_files:
        if filename in sample_data:
            st.sidebar.markdown(get_csv_download_link(sample_data[filename], filename), unsafe_allow_html=True)
    st.sidebar.markdown("---")

    st.sidebar.markdown("#### Upload Your Files")
    all_files = required_files + optional_files
    
    for filename in all_files:
        is_optional = filename in optional_files
        label = f"{filename} (Optional)" if is_optional else filename
        
        uploaded_file = st.sidebar.file_uploader(label, type="csv", key=filename)
        if uploaded_file is not None:
            try:
                uploaded_data[filename] = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
                uploaded_data[filename] = pd.DataFrame()
        elif not is_optional:
            st.warning(f"Required file '{filename}' is missing.")
            uploaded_data[filename] = pd.DataFrame()
            
    if all(uploaded_data.get(f) is not None and not uploaded_data.get(f).empty for f in required_files):
        run_simulation_button = st.sidebar.button("Run Simulation with Uploaded Data")
    else:
        st.sidebar.warning("Please upload all required files to run the simulation.")
        run_simulation_button = False
else:
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

# New slider for window size
if forecast_model in ["Moving Average", "Moving Median"]:
    forecast_window_size = st.sidebar.slider("Window Size (days)", 1, 30, 7)
else:
    forecast_window_size = 7 # Default or unused for other models

st.sidebar.markdown("---")
st.sidebar.subheader("Inventory Policy")
safety_stock_method = st.sidebar.selectbox("Safety Stock Method", ["King's Method", "Avg Max Method"])
service_level = st.sidebar.slider("Desired Service Level (%)", min_value=70, max_value=99, value=95) if safety_stock_method == "King's Method" else None

st.sidebar.markdown("---")
bom_check = st.sidebar.checkbox("BOM Check (Enable for production constraints)", value=True)

# Main content
if run_simulation_button:
    st.header("Simulation Results")
    
    with st.spinner("Preparing and running simulation... This may take a moment."):
        if data_source == "Generated Sample Data":
            st.success("Using generated sample data.")
            all_dfs = generate_realistic_data(DEFAULT_NUM_SKUS, DEFAULT_NUM_COMPONENTS_PER_SKU, num_factories, num_dcs, num_stores, network_type)
        else:
            st.success("Using uploaded custom data.")
            all_dfs = uploaded_data
        
        # Load data from the dictionary
        df_sales = all_dfs.get("sales_data.csv", pd.DataFrame())
        df_inventory = all_dfs.get("inventory_data.csv", pd.DataFrame())
        df_lead_times = all_dfs.get("lead_times_data.csv", pd.DataFrame())
        df_bom = all_dfs.get("bom_data.csv", pd.DataFrame())
        df_global_config = all_dfs.get("global_config.csv", pd.DataFrame())
        df_actual_orders = all_dfs.get("actual_orders.csv", pd.DataFrame())
        df_actual_shipments = all_dfs.get("actual_shipments.csv", pd.DataFrame())

        # Validate that required data is present
        required_files = ["sales_data.csv", "inventory_data.csv", "lead_times_data.csv", "bom_data.csv"]
        if not all(all_dfs.get(f) is not None and not all_dfs.get(f).empty for f in required_files):
            st.error("Cannot run simulation. One or more required data files are missing or empty.")
        else:
            # Determine simulation start and end dates
            sim_start_date = df_sales['Date'].min()
            sim_end_date = sim_start_date + timedelta(days=simulation_days - 1)

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
                forecast_window_size # Pass forecast_window_size
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

            st.subheader("Forecasted Demand Data")
            if not simulation_results['df_forecast'].empty:
                st.dataframe(simulation_results['df_forecast'], use_container_width=True)
            else:
                st.info("No forecasted demand data to display (perhaps the simulation period is entirely within historical data).")
            
            st.markdown("---")
            
            st.subheader("Inventory Levels Over Time")
            all_locations = simulation_results['df_inventory_history']['Location'].unique()
            all_items = simulation_results['df_inventory_history']['Item_ID'].unique()
            
            selected_location = st.selectbox("Select Location", all_locations)
            selected_item = st.selectbox("Select SKU/Component", all_items)
            
            df_plot = simulation_results['df_inventory_history']
            df_plot = df_plot[(df_plot['Location'] == selected_location) & (df_plot['Item_ID'] == selected_item)]
            
            if not df_plot.empty:
                fig = px.line(df_plot, x="Date", y="Stock", title=f"Inventory Level for {selected_item} at {selected_location}")
                
                # Add forecast start line
                if not df_sales.empty and 'Date' in df_sales.columns:
                    fig.add_vline(x=df_sales['Date'].max().to_pydatetime(), line_dash="dash", line_color="red", annotation_text="Forecast Start", annotation_position="top right")
                else:
                    st.warning("Cannot show forecast start line as historical sales data is empty or 'Date' column is missing.")

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No inventory data to display for the selected item and location.")
                
            st.markdown("---")
            st.subheader("Detailed Simulation Events & Alerts")
            with st.expander("View Event Log"):
                if not simulation_results['df_simulation_events'].empty:
                    st.dataframe(simulation_results['df_simulation_events'], use_container_width=True)
                else:
                    st.info("No events to display.")


else:
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
st.header("❓ Frequently Asked Questions")
with st.expander("How does the simulation work?"):
    st.markdown("""
        The simulation runs daily, modeling the entire supply chain from customer demand to factory production.
        
        1.  **Demand Fulfillment:** It starts at the retail store level, fulfilling customer demand for each SKU from available stock.
        2.  **Reorder Logic:** If a store's inventory falls below its reorder point, it places an order with its supplying distribution center (DC).
        3.  **Upstream Demand:** These orders become the demand for the DCs. The same logic applies from the DCs to the factories, creating a multi-echelon demand cascade.
        4.  **Production:** Factories fulfill orders from DCs. If `BOM Check` is enabled, they will only produce if all required components are in stock.
    """)

with st.expander("What does the BOM check do?"):
    st.markdown("""
        When enabled, the `Bill of Materials` (BOM) check adds a crucial layer of realism to the simulation.
        A factory will only be able to fulfill an order if it has all the necessary components for that SKU in its `component_inventory`. If a component is missing, the order cannot be fulfilled, which can lead to a delayed shipment and a potential stockout at the downstream location. This accurately reflects real-world production constraints.
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
        * **Random Forest:** A powerful machine learning model that uses an ensemble of decision trees to predict future demand based on patterns in your historical data.
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