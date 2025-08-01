# DF_v9.py - Demand and Inventory Intelligence Streamlit App with Advanced Multi-Echelon and BOM Integration

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
import graphviz # Added for network visualization

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
DEFAULT_DISTRIBUTION_CENTERS = ["DC-A", "DC-B"]
DEFAULT_RETAIL_STORES = ["Store-1", "Store-2", "Store-3", "Store-4"]
DEFAULT_COMPONENT_SUPPLIERS = ["Supplier-X", "Supplier-Y"]

# --- Default Cost Parameters (used if global_config.csv is not uploaded) ---
DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY = 0.10
DEFAULT_ORDERING_COST_PER_ORDER = 50.00
DEFAULT_STOCKOUT_COST_PER_UNIT = 10.00
DEFAULT_SERVICE_LEVEL = 95 # Default to 95% service level

# --- Helper Functions ---

def generate_dummy_data():
    """Generates a realistic set of dummy data with a multi-echelon structure."""
    st.info("Generating realistic sample data for Mobile & Wearables products with a multi-echelon network...")
    
    # SKU Master Data
    sku_data = []
    for i in range(DEFAULT_NUM_SKUS):
        sku_id = f"Smartwatch-{i+1}"
        retail_price = round(random.uniform(150, 400), 2)
        shelf_life = random.randint(200, DEFAULT_MAX_SKU_SHELF_LIFE_DAYS)
        lead_time = random.randint(7, DEFAULT_MAX_LEAD_TIME_DAYS)
        safety_stock_factor = random.uniform(1.2, 1.5)
        
        sku_data.append([sku_id, retail_price, shelf_life, lead_time, safety_stock_factor])
    
    skus_df = pd.DataFrame(
        sku_data,
        columns=['SKU_ID', 'Retail_Price', 'Shelf_Life_Days', 'Lead_Time_Days', 'Safety_Stock_Factor']
    )
    
    # Sales Data
    dates = pd.date_range(start=DEFAULT_START_DATE, end=DEFAULT_END_DATE)
    sales_data = []
    
    for _, sku_row in skus_df.iterrows():
        sku_id = sku_row['SKU_ID']
        base_demand = random.randint(10, 50)
        
        for date in dates:
            for region in DEFAULT_REGIONS:
                for store in DEFAULT_RETAIL_STORES:
                    sales_channel = random.choice(DEFAULT_SALES_CHANNELS)
                    
                    month = date.month
                    seasonality_multiplier = 1.0
                    if month in [10, 11, 12]:
                        seasonality_multiplier = random.uniform(1.2, 1.8)
                    
                    trend = (date - DEFAULT_START_DATE).days / 365 * 0.1
                    
                    is_promotion = random.random() < 0.1
                    promotion_effect = 1.0
                    promotion_discount = 0.0
                    if is_promotion:
                        promotion_effect = random.uniform(1.2, 1.5)
                        promotion_discount = random.uniform(0.1, 0.3)
                    
                    competitor_price = sku_row['Retail_Price'] * random.uniform(0.9, 1.1)
                    online_ad_spend = random.uniform(100, 500) if random.random() > 0.7 else 0
                    
                    demand = int(base_demand * seasonality_multiplier * (1 + trend) * promotion_effect + random.gauss(0, 5))
                    demand = max(0, demand)
                    
                    sales_data.append({
                        'Date': date,
                        'SKU_ID': sku_id,
                        'Sales_Channel': sales_channel,
                        'Region': region,
                        'Location': store,
                        'Demand_Quantity': demand,
                        'Promotion_Discount_Rate': promotion_discount,
                        'Online_Ad_Spend': online_ad_spend,
                        'Competitor_Price': competitor_price
                    })
    sales_df = pd.DataFrame(sales_data)

    # Inventory Data
    inventory_data = []
    all_locations = DEFAULT_FACTORIES + DEFAULT_DISTRIBUTION_CENTERS + DEFAULT_RETAIL_STORES
    for _, sku_row in skus_df.iterrows():
        sku_id = sku_row['SKU_ID']
        for date in dates:
            for location in all_locations:
                if location in DEFAULT_FACTORIES:
                    location_type = 'Factory'
                    on_hand = random.randint(2000, 5000)
                elif location in DEFAULT_DISTRIBUTION_CENTERS:
                    location_type = 'Distribution Center'
                    on_hand = random.randint(500, 1500)
                else: # Retail Store
                    location_type = 'Retail Store'
                    on_hand = random.randint(50, 200)

                inventory_data.append({
                    'Date': date,
                    'SKU_ID': sku_id,
                    'Location': location,
                    'Location_Type': location_type,
                    'On_Hand_Inventory': on_hand
                })
    inventory_df = pd.DataFrame(inventory_data)

    # Component Inventory
    component_inventory_data = []
    component_ids = [f"Component-C{i+1}" for i in range(DEFAULT_NUM_SKUS * DEFAULT_NUM_COMPONENTS_PER_SKU)]
    for comp in component_ids:
        for date in dates:
            on_hand = random.randint(1000, 5000)
            component_inventory_data.append({
                'Date': date,
                'Component_ID': comp,
                'Location': DEFAULT_FACTORIES[0],
                'Location_Type': 'Factory',
                'On_Hand_Inventory': on_hand
            })
    component_inventory_df = pd.DataFrame(component_inventory_data)

    # BOM Data
    bom_data = []
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
    
    # Network Structure Data
    network_structure = []
    for dc in DEFAULT_DISTRIBUTION_CENTERS:
        factory = random.choice(DEFAULT_FACTORIES)
        network_structure.append({
            'Source_Location': factory,
            'Source_Location_Type': 'Factory',
            'Destination_Location': dc,
            'Destination_Location_Type': 'Distribution Center',
            'Transit_Time_Days': random.randint(2, 5)
        })
    for store in DEFAULT_RETAIL_STORES:
        dc = random.choice(DEFAULT_DISTRIBUTION_CENTERS)
        network_structure.append({
            'Source_Location': dc,
            'Source_Location_Type': 'Distribution Center',
            'Destination_Location': store,
            'Destination_Location_Type': 'Retail Store',
            'Transit_Time_Days': random.randint(1, 3)
        })
    network_df = pd.DataFrame(network_structure)

    st.session_state.sales_df = sales_df
    st.session_state.skus_df = skus_df
    st.session_state.bom_df = bom_df
    st.session_state.inventory_df = inventory_df
    st.session_state.network_df = network_df
    st.session_state.component_inventory_df = component_inventory_df
    st.success("Sample data loaded successfully!")

def create_template_df(data_type):
    """Generates a template DataFrame for a given data type."""
    if data_type == "Sales":
        return pd.DataFrame({
            'Date': [datetime.now().strftime('%Y-%m-%d')],
            'SKU_ID': ['Smartwatch-1'],
            'Sales_Channel': ['Direct-to-Consumer'],
            'Region': ['North America'],
            'Location': ['Store-1'],
            'Demand_Quantity': [100],
            'Promotion_Discount_Rate': [0.15],
            'Online_Ad_Spend': [250.50],
            'Competitor_Price': [200.00]
        })
    elif data_type == "Inventory":
        return pd.DataFrame({
            'Date': [datetime.now().strftime('%Y-%m-%d')],
            'SKU_ID': ['Smartwatch-1'],
            'Location': ['Store-1'],
            'Location_Type': ['Retail Store'],
            'On_Hand_Inventory': [500]
        })
    elif data_type == "Component Inventory":
        return pd.DataFrame({
            'Date': [datetime.now().strftime('%Y-%m-%d')],
            'Component_ID': ['Component-C1'],
            'Location': ['Factory-A'],
            'Location_Type': ['Factory'],
            'On_Hand_Inventory': [2000]
        })
    elif data_type == "Network":
        return pd.DataFrame({
            'Source_Location': ['DC-A'],
            'Source_Location_Type': ['Distribution Center'],
            'Destination_Location': ['Store-1'],
            'Destination_Location_Type': ['Retail Store'],
            'Transit_Time_Days': [2]
        })
    elif data_type == "SKU Master":
        return pd.DataFrame({
            'SKU_ID': ['Smartwatch-1'],
            'Retail_Price': [350.00],
            'Shelf_Life_Days': [365],
            'Lead_Time_Days': [14],
            'Safety_Stock_Factor': [1.3]
        })
    elif data_type == "BOM":
        return pd.DataFrame({
            'SKU_ID': ['Smartwatch-1'],
            'Component_ID': ['Component-C1'],
            'Quantity_Per_SKU': [2],
            'Cost_Per_Unit': [15.00]
        })
    elif data_type == "Global Costs":
        return pd.DataFrame({
            'Holding_Cost_Per_Unit_Per_Day': [0.10],
            'Ordering_Cost_Per_Order': [50.00],
            'Stockout_Cost_Per_Unit': [10.00]
        })
        
# --- Forecasting Model Functions (re-used from v8) ---
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

def auto_select_best_model(df, models_to_test):
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

def calculate_reorder_point(historical_demand_df, lead_time_days, service_level, safety_stock_method, safety_stock_factor=None):
    """
    Calculates a realistic reorder point based on the selected method.
    """
    if historical_demand_df.empty or historical_demand_df['Demand_Quantity'].isnull().all():
        return 0
    
    lead_time_days = int(lead_time_days)
    avg_demand = historical_demand_df['Demand_Quantity'].mean()
    
    if safety_stock_method == "Statistical (Reorder Point)":
        std_demand = historical_demand_df['Demand_Quantity'].std()
        avg_demand_during_lead_time = avg_demand * lead_time_days
        std_demand_during_lead_time = std_demand * np.sqrt(lead_time_days)
        service_level_z_score = norm.ppf(service_level / 100)
        safety_stock = service_level_z_score * std_demand_during_lead_time
        reorder_point = math.ceil(avg_demand_during_lead_time + safety_stock)
    elif safety_stock_method == "King's Method":
        safety_stock = avg_demand * safety_stock_factor
        reorder_point = math.ceil((avg_demand * lead_time_days) + safety_stock)
    else:
        reorder_point = math.ceil(avg_demand * lead_time_days)
        
    return reorder_point

def run_multi_echelon_simulation(sales_df, inventory_df, component_inventory_df, network_df, skus_df, bom_df, reorder_df, holding_cost, ordering_cost, stockout_cost, enable_bom_check):
    """
    Simulates inventory levels and calculates KPIs for a multi-echelon network.
    This function performs a day-by-day simulation, propagating orders up the supply chain.
    """
    kpi_data = []
    
    # Corrected method for getting unique, sorted dates to avoid 'AttributeError'
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    all_dates = sales_df['Date'].sort_values().unique()
    
    # Initialize a dictionary to hold daily inventory levels and orders
    current_inventory = {}
    current_component_inventory = {}
    
    total_holding_cost = 0
    total_ordering_cost = 0
    total_stockout_cost = 0
    
    # Get all locations and SKUs
    all_locations = list(inventory_df['Location'].unique())
    all_skus = list(inventory_df['SKU_ID'].unique())
    all_components = list(component_inventory_df['Component_ID'].unique())
    
    # Initialize inventory levels for the start of the simulation
    initial_inventory_df = inventory_df.loc[pd.to_datetime(inventory_df['Date']) == all_dates.min()]
    for _, row in initial_inventory_df.iterrows():
        current_inventory[(row['SKU_ID'], row['Location'])] = row['On_Hand_Inventory']
    
    if enable_bom_check:
        initial_component_df = component_inventory_df.loc[pd.to_datetime(component_inventory_df['Date']) == all_dates.min()]
        for _, row in initial_component_df.iterrows():
            current_component_inventory[row['Component_ID']] = row['On_Hand_Inventory']
            
    # Identify the echelons
    echelons = network_df['Destination_Location_Type'].unique().tolist() + network_df['Source_Location_Type'].unique().tolist()
    echelons = sorted(list(set(echelons)), reverse=True) # Sort so we process from downstream to upstream

    # Simulation state variables
    daily_stockout_tracker = {sku: {loc: 0 for loc in all_locations} for sku in all_skus}
    daily_orders_placed = {}
    open_orders = [] # list of tuples: (arrival_date, sku, quantity, location)
    
    # Main simulation loop
    for date in all_dates:
        
        # Process newly arrived orders
        arrived_orders = [o for o in open_orders if o[0] <= date]
        for arrival_date, sku, quantity, location in arrived_orders:
            current_inventory[(sku, location)] += quantity
        open_orders = [o for o in open_orders if o[0] > date]
        
        # Propagate demand and place new orders from downstream to upstream
        for echelon in echelons:
            locations_in_echelon = inventory_df[inventory_df['Location_Type'] == echelon]['Location'].unique()
            
            for location in locations_in_echelon:
                # Get the daily demand for this location
                if echelon == 'Retail Store':
                    location_demand = sales_df.loc[
                        (sales_df['Date'] == date) & (sales_df['Location'] == location)
                    ]
                    for _, row in location_demand.iterrows():
                        sku = row['SKU_ID']
                        demand_qty = row['Demand_Quantity']
                        
                        # Fulfill demand
                        if current_inventory.get((sku, location), 0) < demand_qty:
                            daily_stockout_tracker[sku][location] += (demand_qty - current_inventory.get((sku, location), 0))
                            current_inventory[(sku, location)] = 0
                        else:
                            current_inventory[(sku, location)] -= demand_qty
                            
                        # Place new order if below reorder point
                        reorder_point_df = reorder_df.loc[(reorder_df['SKU_ID'] == sku) & (reorder_df['Location'] == location)]
                        if not reorder_point_df.empty:
                            reorder_point = reorder_point_df['Reorder_Point'].iloc[0]
                            if current_inventory.get((sku, location), 0) <= reorder_point:
                                order_quantity = reorder_point * 2 # Simple order-up-to policy
                                
                                # Determine supplier and lead time
                                supplier_row = network_df.loc[network_df['Destination_Location'] == location]
                                if not supplier_row.empty:
                                    supplier_location = supplier_row['Source_Location'].iloc[0]
                                    lead_time = supplier_row['Transit_Time_Days'].iloc[0]
                                    arrival_date = date + timedelta(days=lead_time)
                                    open_orders.append((arrival_date, sku, order_quantity, location))
                                    total_ordering_cost += ordering_cost
                                    
                # For other echelons, demand comes from downstream locations
                else:
                    # Collect orders from downstream locations
                    downstream_locations = network_df.loc[network_df['Source_Location'] == location]['Destination_Location'].unique()
                    for sku in all_skus:
                        # Check if a demand order was placed for this SKU from a downstream location
                        demand_qty = sum(daily_orders_placed.get((sku, downstream_loc), 0) for downstream_loc in downstream_locations)
                        
                        if demand_qty > 0:
                            # Fulfill demand
                            if current_inventory.get((sku, location), 0) < demand_qty:
                                daily_stockout_tracker[sku][location] += (demand_qty - current_inventory.get((sku, location), 0))
                                current_inventory[(sku, location)] = 0
                            else:
                                current_inventory[(sku, location)] -= demand_qty
                                
                            # Place new order if below reorder point
                            reorder_point_df = reorder_df.loc[(reorder_df['SKU_ID'] == sku) & (reorder_df['Location'] == location)]
                            if not reorder_point_df.empty:
                                reorder_point = reorder_point_df['Reorder_Point'].iloc[0]
                                if current_inventory.get((sku, location), 0) <= reorder_point:
                                    order_quantity = reorder_point * 2
                                    
                                    # Determine supplier and lead time
                                    supplier_row = network_df.loc[network_df['Destination_Location'] == location]
                                    if not supplier_row.empty:
                                        supplier_location = supplier_row['Source_Location'].iloc[0]
                                        lead_time = supplier_row['Transit_Time_Days'].iloc[0]
                                        arrival_date = date + timedelta(days=lead_time)
                                        open_orders.append((arrival_date, sku, order_quantity, location))
                                        total_ordering_cost += ordering_cost
        
        # Calculate daily costs and KPIs
        daily_inventory_level = sum(current_inventory.values())
        daily_stockout_units = sum(sum(loc_stockouts.values()) for loc_stockouts in daily_stockout_tracker.values())
        daily_holding_cost = daily_inventory_level * holding_cost
        daily_stockout_cost = daily_stockout_units * stockout_cost
        
        total_holding_cost += daily_holding_cost
        total_stockout_cost += daily_stockout_cost
        
        kpi_data.append({
            'Date': date,
            'Total_Inventory_Level': daily_inventory_level,
            'Total_Holding_Cost': total_holding_cost,
            'Total_Ordering_Cost': total_ordering_cost,
            'Total_Stockout_Cost': total_stockout_cost,
            'Total_Cost': total_holding_cost + total_ordering_cost + total_stockout_cost
        })
        
    return pd.DataFrame(kpi_data), daily_stockout_tracker
    
    
# --- Streamlit App ---

st.set_page_config(
    page_title="Demand and Inventory Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Demand and Inventory Intelligence App (v9)")
st.caption("Advanced Multi-Echelon & BOM Integration")

st.markdown("""
This app simulates and optimizes inventory and demand planning using various models.
""")

# --- State Management ---
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = pd.DataFrame()
if 'inventory_df' not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame()
if 'bom_df' not in st.session_state:
    st.session_state.bom_df = pd.DataFrame()
if 'skus_df' not in st.session_state:
    st.session_state.skus_df = pd.DataFrame()
if 'network_df' not in st.session_state:
    st.session_state.network_df = pd.DataFrame()
if 'component_inventory_df' not in st.session_state:
    st.session_state.component_inventory_df = pd.DataFrame()
if 'reorder_df' not in st.session_state:
    st.session_state.reorder_df = pd.DataFrame()
if 'costs_df' not in st.session_state:
    st.session_state.costs_df = pd.DataFrame()
if 'kpis_df' not in st.session_state:
    st.session_state.kpis_df = pd.DataFrame()
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if 'all_stockout_rates_df' not in st.session_state:
    st.session_state.all_stockout_rates_df = pd.DataFrame()
if 'reorder_points_df' not in st.session_state:
    st.session_state.reorder_points_df = pd.DataFrame()
if 'forecast_history' not in st.session_state:
    st.session_state.forecast_history = {}


# --- Sidebar for Data Upload and Configuration ---
st.sidebar.header("Data Configuration")
selected_app_mode = st.sidebar.radio("Choose App Mode", ["Manual Data Upload", "Run with Sample Data"])

if selected_app_mode == "Manual Data Upload":
    st.sidebar.subheader("Upload Your Data")
    
    sales_file = st.sidebar.file_uploader("Upload Sales Data (CSV)", type=["csv"], key="sales_file")
    inventory_file = st.sidebar.file_uploader("Upload Inventory Data (CSV)", type=["csv"], key="inventory_file")
    component_inventory_file = st.sidebar.file_uploader("Upload Component Inventory Data (CSV)", type=["csv"], key="component_inventory_file")
    bom_file = st.sidebar.file_uploader("Upload Bill of Materials (BOM) (CSV)", type=["csv"], key="bom_file")
    network_file = st.sidebar.file_uploader("Upload Supply Network (CSV)", type=["csv"], key="network_file")
    sku_master_file = st.sidebar.file_uploader("Upload SKU Master Data (CSV)", type=["csv"], key="sku_master_file")
    reorder_file = st.sidebar.file_uploader("Upload Reorder Policies (CSV)", type=["csv"], key="reorder_file")
    costs_file = st.sidebar.file_uploader("Upload Global Cost Data (CSV)", type=["csv"], key="costs_file")

    if st.sidebar.button("Load Uploaded Files"):
        with st.sidebar.spinner("Loading files..."):
            if sales_file:
                st.session_state.sales_df = pd.read_csv(sales_file)
            if inventory_file:
                st.session_state.inventory_df = pd.read_csv(inventory_file)
            if component_inventory_file:
                st.session_state.component_inventory_df = pd.read_csv(component_inventory_file)
            if bom_file:
                st.session_state.bom_df = pd.read_csv(bom_file)
            if network_file:
                st.session_state.network_df = pd.read_csv(network_file)
            if sku_master_file:
                st.session_state.skus_df = pd.read_csv(sku_master_file)
            if reorder_file:
                st.session_state.reorder_df = pd.read_csv(reorder_file)
            if costs_file:
                st.session_state.costs_df = pd.read_csv(costs_file)
        st.sidebar.success("Files loaded successfully!")

    if st.sidebar.button("Download Sample Data Templates"):
        st.sidebar.write("Downloading sample data templates...")
        for data_type in ["Sales", "Inventory", "Component Inventory", "Network", "SKU Master", "BOM", "Global Costs"]:
            template_df = create_template_df(data_type)
            csv = template_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label=f"Download {data_type} Template",
                data=csv,
                file_name=f"{data_type.lower().replace(' ', '_')}_template.csv",
                mime="text/csv",
            )
        st.sidebar.success("All templates generated!")


elif selected_app_mode == "Run with Sample Data":
    st.sidebar.subheader("Sample Data Generation")
    
    num_skus_input = st.sidebar.number_input("Number of SKUs", min_value=1, max_value=50, value=DEFAULT_NUM_SKUS)
    start_date_input = st.sidebar.date_input("Start Date", value=DEFAULT_START_DATE)
    end_date_input = st.sidebar.date_input("End Date", value=DEFAULT_END_DATE)
    
    if st.sidebar.button("Generate and Load Sample Data"):
        with st.sidebar.spinner("Generating and loading sample data..."):
            generate_dummy_data()
        st.experimental_rerun()


# --- Display Uploaded Data ---
if not st.session_state.sales_df.empty:
    st.subheader("Uploaded Data Preview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if not st.session_state.sales_df.empty:
            st.write("**Sales Data**")
            st.dataframe(st.session_state.sales_df.head(), use_container_width=True)
    with col2:
        if not st.session_state.inventory_df.empty:
            st.write("**Inventory Data**")
            st.dataframe(st.session_state.inventory_df.head(), use_container_width=True)
    with col3:
        if not st.session_state.bom_df.empty:
            st.write("**BOM Data**")
            st.dataframe(st.session_state.bom_df.head(), use_container_width=True)
    with col4:
        if not st.session_state.network_df.empty:
            st.write("**Network Data**")
            st.dataframe(st.session_state.network_df.head(), use_container_width=True)

    if not st.session_state.reorder_df.empty:
        st.write("Reorder Policies:")
        st.dataframe(st.session_state.reorder_df.head(), use_container_width=True)

# --- Forecasting Section ---
st.header("Forecasting")
if not st.session_state.sales_df.empty:
    
    forecast_col1, forecast_col2 = st.columns(2)
    with forecast_col1:
        all_skus_for_forecast = st.session_state.sales_df['SKU_ID'].unique()
        selected_sku_for_forecast = st.selectbox(
            "Select SKU for Forecasting",
            all_skus_for_forecast,
            key="selected_sku_forecast"
        )
    with forecast_col2:
        forecast_horizon = st.number_input(
            "Forecast Horizon (Days)",
            min_value=1,
            value=365
        )
    
    st.subheader("Model Selection")
    
    # Model configuration
    model_col1, model_col2 = st.columns(2)
    
    with model_col1:
        forecast_model = st.selectbox(
            "Choose Forecasting Model",
            ["Auto-Select", "XGBoost", "Random Forest", "Moving Average", "Moving Median"],
            key="selected_model_forecast"
        )
    
    with model_col2:
        if forecast_model in ["XGBoost", "Random Forest"]:
            n_estimators = st.number_input("Number of Estimators", min_value=10, value=100)
            max_depth = st.number_input("Max Depth", min_value=1, value=5)
            model_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        elif forecast_model in ["Moving Average", "Moving Median"]:
            window_size = st.number_input("Window Size", min_value=1, value=7)
            model_params = {'window_size': window_size}
        else:
            model_params = {}
    
    if st.button("Run Forecasting"):
        with st.spinner(f"Running {forecast_model} forecast for {selected_sku_for_forecast}..."):
            sku_sales_df = st.session_state.sales_df[st.session_state.sales_df['SKU_ID'] == selected_sku_for_forecast].copy()
            sku_sales_df['Date'] = pd.to_datetime(sku_sales_df['Date'])
            
            # Aggregate to daily demand
            daily_demand = sku_sales_df.groupby('Date').agg(Demand_Quantity=('Demand_Quantity', 'sum')).reset_index()
            
            # Join with other data for features
            # NOTE: For a real app, this would require careful feature engineering.
            # We'll use a simplified merge here.
            
            # Placeholder for merging, assuming dummy data has daily entries for these features
            # A more robust solution would require resampling and forward-filling
            
            # Create a simple dataframe for forecasting
            forecast_df_base = daily_demand.rename(columns={'Date': 'ds', 'Demand_Quantity': 'y'})
            
            if "Promotion_Discount_Rate" in sku_sales_df.columns:
                promo_df = sku_sales_df[['Date', 'Promotion_Discount_Rate']].drop_duplicates().set_index('Date').resample('D').mean().fillna(0).reset_index()
                forecast_df_base = pd.merge(forecast_df_base, promo_df, left_on='ds', right_on='Date', how='left').drop(columns='Date')
            else:
                forecast_df_base['Promotion_Discount_Rate'] = 0.0

            if "Online_Ad_Spend" in sku_sales_df.columns:
                ad_spend_df = sku_sales_df[['Date', 'Online_Ad_Spend']].drop_duplicates().set_index('Date').resample('D').sum().fillna(0).reset_index()
                forecast_df_base = pd.merge(forecast_df_base, ad_spend_df, left_on='ds', right_on='Date', how='left').drop(columns='Date')
            else:
                forecast_df_base['Online_Ad_Spend'] = 0.0

            if "Competitor_Price" in sku_sales_df.columns:
                comp_price_df = sku_sales_df[['Date', 'Competitor_Price']].drop_duplicates().set_index('Date').resample('D').mean().fillna(method='ffill').reset_index()
                forecast_df_base = pd.merge(forecast_df_base, comp_price_df, left_on='ds', right_on='Date', how='left').drop(columns='Date')
            else:
                forecast_df_base['Competitor_Price'] = 0.0

            forecast_df_base = forecast_df_base.fillna(0)

            
            # Handle Auto-Select
            if forecast_model == "Auto-Select":
                models_to_test = ["XGBoost", "Random Forest", "Moving Average", "Moving Median"]
                best_model_name = auto_select_best_model(forecast_df_base, models_to_test)
                # Re-run the best model to get the full forecast
                if best_model_name == "Moving Average" or best_model_name == "Moving Median":
                    model_params = {'window_size': 7}
                else:
                    model_params = {'n_estimators': 100, 'max_depth': 3}
                forecast_results, mae, rmse = run_forecasting(forecast_df_base, best_model_name, forecast_horizon, model_params)
            else:
                forecast_results, mae, rmse = run_forecasting(forecast_df_base, forecast_model, forecast_horizon, model_params)
            
            future_dates = pd.date_range(start=forecast_df_base['ds'].max() + timedelta(days=1), periods=forecast_horizon)
            
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted_Demand': forecast_results
            })
            
            st.session_state.forecast_df = forecast_df
            st.session_state.forecast_history[selected_sku_for_forecast] = forecast_df
            
            # Display metrics
            if mae is not None and rmse is not None:
                st.markdown(f"**Forecast Metrics:**")
                st.markdown(f"**Mean Absolute Error (MAE):** `{mae:.2f}`")
                st.markdown(f"**Root Mean Squared Error (RMSE):** `{rmse:.2f}`")

            st.success("Forecasting complete!")

# --- Forecasting Results Visualization ---
if not st.session_state.forecast_df.empty:
    st.subheader("Forecast Results")
    
    # Plot historical demand and forecast
    historical_demand_plot = st.session_state.sales_df.groupby(['Date', 'SKU_ID'])['Demand_Quantity'].sum().reset_index()
    historical_demand_plot['Date'] = pd.to_datetime(historical_demand_plot['Date'])
    historical_demand_plot = historical_demand_plot[historical_demand_plot['SKU_ID'] == st.session_state.selected_sku_forecast]
    
    # Combine historical and forecast data for plotting
    historical_demand_plot = historical_demand_plot.rename(columns={'Demand_Quantity': 'Demand'})
    forecast_plot = st.session_state.forecast_df.rename(columns={'Forecasted_Demand': 'Demand'})
    
    historical_demand_plot['Type'] = 'Historical'
    forecast_plot['Type'] = 'Forecast'

    # Filter out dates where forecast overlaps with historical data
    max_historical_date = historical_demand_plot['Date'].max()
    forecast_plot = forecast_plot[forecast_plot['Date'] > max_historical_date]
    
    combined_df = pd.concat([historical_demand_plot, forecast_plot])
    
    fig = px.line(
        combined_df,
        x='Date',
        y='Demand',
        color='Type',
        title=f"Historical Demand vs. Forecast for {st.session_state.selected_sku_forecast}",
        labels={'Demand': 'Demand (Units)'},
        color_discrete_map={'Historical': 'green', 'Forecast': 'red'}
    )
    
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# --- Simulation and Analysis ---
st.header("Simulation and Analysis")

if not st.session_state.sales_df.empty:
    
    # Sidebar for Simulation Parameters
    with st.expander("Simulation Parameters"):
        st.subheader("Simulation Options")
        
        selected_sc_model = st.radio(
            "Choose Simulation Model",
            ["Multi-Echelon", "Reorder Point"],
            help="""
            - **Multi-Echelon**: A simulation that considers the entire supply chain network and BOM,
              calculating KPIs based on inventory movements and reorder policies.
            - **Reorder Point**: A simple model that calculates stockout rates based on a single
              reorder point and lead time.
            """
        )
        
        enable_bom_check = st.checkbox("Enable Bill of Materials (BOM) Production Check", value=True)
        
        st.subheader("Reorder Point Calculation")
        
        safety_stock_method = st.selectbox(
            "Safety Stock Calculation Method",
            ["Statistical (Reorder Point)", "King's Method", "No Safety Stock"]
        )
        
        safety_stock_factor = None
        if safety_stock_method == "King's Method":
            safety_stock_factor = st.slider("Safety Stock Factor", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    run_simulation_button = st.button("Run Simulation and Analysis")

    if run_simulation_button:
        with st.spinner("Running simulation and analysis..."):
            
            # Validate required dataframes
            if selected_sc_model == "Multi-Echelon" and (st.session_state.network_df.empty or st.session_state.reorder_df.empty or st.session_state.bom_df.empty or st.session_state.inventory_df.empty):
                st.error("Please upload all required dataframes for Multi-Echelon model: Sales, Inventory, BOM, Network, and Reorder Policies.")
                st.stop()
            
            # Default to dummy costs if not provided
            if st.session_state.costs_df.empty:
                costs_df = create_template_df("Global Costs")
            else:
                costs_df = st.session_state.costs_df
            
            holding_cost = costs_df.loc[0, 'Holding_Cost_Per_Unit_Per_Day'] if not costs_df.empty else DEFAULT_HOLDING_COST_PER_UNIT_PER_DAY
            ordering_cost = costs_df.loc[0, 'Ordering_Cost_Per_Order'] if not costs_df.empty else DEFAULT_ORDERING_COST_PER_ORDER
            stockout_cost = costs_df.loc[0, 'Stockout_Cost_Per_Unit'] if not costs_df.empty else DEFAULT_STOCKOUT_COST_PER_UNIT
            
            # Generate Reorder Points if not provided
            if st.session_state.reorder_df.empty:
                st.info("Generating Reorder Points based on your data and selected method...")
                
                reorder_points_list = []
                unique_sku_locations = st.session_state.sales_df[['SKU_ID', 'Location']].drop_duplicates().to_records(index=False)
                
                for sku, location in unique_sku_locations:
                    historical_demand = st.session_state.sales_df[
                        (st.session_state.sales_df['SKU_ID'] == sku) &
                        (st.session_state.sales_df['Location'] == location)
                    ].copy()
                    
                    historical_demand['Date'] = pd.to_datetime(historical_demand['Date'])
                    
                    # Assume a default lead time if not available
                    lead_time = st.session_state.skus_df.loc[st.session_state.skus_df['SKU_ID'] == sku, 'Lead_Time_Days'].iloc[0] if not st.session_state.skus_df.empty else 14
                    
                    reorder_point = calculate_reorder_point(
                        historical_demand, 
                        lead_time_days=lead_time,
                        service_level=DEFAULT_SERVICE_LEVEL,
                        safety_stock_method=safety_stock_method,
                        safety_stock_factor=safety_stock_factor
                    )
                    
                    # Simple Order Quantity heuristic
                    order_quantity = math.ceil(historical_demand['Demand_Quantity'].mean() * 30)
                    
                    reorder_points_list.append({
                        'SKU_ID': sku,
                        'Location': location,
                        'Reorder_Point': reorder_point,
                        'Order_Quantity': order_quantity
                    })
                
                st.session_state.reorder_df = pd.DataFrame(reorder_points_list)
                st.write("Generated Reorder Policies:")
                st.dataframe(st.session_state.reorder_df, use_container_width=True)

            # Run the simulation
            if selected_sc_model == "Multi-Echelon":
                st.session_state.kpis_df, daily_stockout_tracker = run_multi_echelon_simulation(
                    st.session_state.sales_df, 
                    st.session_state.inventory_df, 
                    st.session_state.component_inventory_df,
                    st.session_state.network_df,
                    st.session_state.skus_df,
                    st.session_state.bom_df,
                    st.session_state.reorder_df,
                    holding_cost,
                    ordering_cost,
                    stockout_cost,
                    enable_bom_check
                )
            
                st.success("Simulation and analysis complete!")
            else:
                st.warning("Reorder Point simulation model is not yet implemented. Please select Multi-Echelon.")
                st.stop()


# --- Results and Visualizations ---
if not st.session_state.kpis_df.empty:
    st.header("Simulation Results")
    
    # KPI Plots
    st.subheader("Overall KPIs (Simulation)")
    
    # Total Cost Plot
    fig_cost = px.line(
        st.session_state.kpis_df,
        x="Date",
        y="Total_Cost",
        title="Cumulative Total Cost over Time",
        labels={"Total_Cost": "Cumulative Total Cost ($)", "Date": "Date"},
        color_discrete_sequence=['red']
    )
    fig_cost.update_layout(hovermode="x unified")
    st.plotly_chart(fig_cost, use_container_width=True)
    
    # Inventory Level Plot
    fig_inv = px.line(
        st.session_state.kpis_df,
        x="Date",
        y="Total_Inventory_Level",
        title="Total Inventory Level over Time",
        labels={"Total_Inventory_Level": "Total Inventory Level (Units)", "Date": "Date"},
        color_discrete_sequence=['blue']
    )
    fig_inv.update_layout(hovermode="x unified")
    st.plotly_chart(fig_inv, use_container_width=True)

    # Detailed costs
    st.subheader("Detailed Costs")
    cost_df_long = st.session_state.kpis_df[['Date', 'Total_Holding_Cost', 'Total_Ordering_Cost', 'Total_Stockout_Cost']]
    cost_df_long = cost_df_long.melt(id_vars='Date', var_name='Cost Type', value_name='Cumulative Cost')
    
    fig_costs = px.line(
        cost_df_long,
        x="Date",
        y="Cumulative Cost",
        color="Cost Type",
        title="Cumulative Costs Breakdown",
        labels={"Cumulative Cost": "Cost ($)", "Date": "Date"}
    )
    fig_costs.update_layout(hovermode="x unified")
    st.plotly_chart(fig_costs, use_container_width=True)


    # Network Visualization
    st.subheader("Supply Chain Network Visualization")
    if not st.session_state.network_df.empty:
        g = graphviz.Digraph(comment='Supply Chain Network')
        
        # Add nodes
        for _, row in st.session_state.network_df.iterrows():
            g.node(row['Source_Location'], row['Source_Location'])
            g.node(row['Destination_Location'], row['Destination_Location'])
        
        # Add edges
        for _, row in st.session_state.network_df.iterrows():
            g.edge(row['Source_Location'], row['Destination_Location'], label=f"{row['Transit_Time_Days']} days")
            
        st.graphviz_chart(g)
    else:
        st.warning("No network data found to visualize.")
