# DFv12.py - Demand and Inventory Intelligence Streamlit App
# Features: Configurable Single/Multi-Echelon Network, realistic data generation,
#           individual CSV upload/download, and safety stock method selection.

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

def generate_realistic_data(num_skus, num_components, num_factories, num_dcs, num_stores, network_type):
    """
    Generates a complete set of realistic dummy dataframes for the app.
    Now includes options for single or multi-echelon networks.
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
                
                # Apply trend, seasonality, and randomness
                demand_factor = 1.0
                demand_factor *= day_of_year_sin[date_idx] * day_of_year_cos[date_idx]
                
                # Apply promotion effect
                if random.random() < 0.05: # 5% chance of a promo day
                    demand_factor *= random.uniform(1.5, 2.5)
                
                sales_quantity = max(0, int(base_sales * demand_factor) + random.randint(-10, 10))
                
                sales_data.append({
                    "Date": date,
                    "SKU_ID": sku,
                    "Location": store, # Sales now tied to a specific retail location
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
    sample_data = generate_realistic_data(1, 1, 1, 1, 1, "Multi-Echelon")
    
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
st.sidebar.subheader("Safety Stock Parameters")
safety_stock_method = st.sidebar.selectbox("Safety Stock Method", ["King's Method", "Avg Max Method"])
service_level = st.sidebar.slider("Desired Service Level (%)", min_value=70, max_value=99, value=95) if safety_stock_method == "King's Method" else None

st.sidebar.markdown("---")
bom_check = st.sidebar.checkbox("BOM Check (Enable for production constraints)", value=True)

# Main content
if run_simulation_button:
    st.header("Simulation Results")
    
    with st.spinner("Preparing and running simulation..."):
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

        # Display Data Tables
        st.subheader("Uploaded/Generated Data Templates")
        with st.expander("View All Data Tables"):
            for filename, df in all_dfs.items():
                st.markdown(f"#### {filename}")
                st.dataframe(df.head(), use_container_width=True)
                if df.empty:
                    st.info("DataFrame is empty.")
                st.markdown("---")

        if df_sales.empty:
            st.error("Cannot run simulation. `sales_data.csv` is missing or empty.")
        else:
            st.markdown("---")
            st.subheader("Historical Insights")
            
            # Historical Fill Rate
            if not df_actual_orders.empty and not df_actual_shipments.empty:
                fill_rate = calculate_historical_fill_rate(df_actual_orders, df_actual_shipments)
                st.info(f"**Historical Fill Rate:** {fill_rate:.2f}%")
            else:
                st.info("Historical fill rate could not be calculated. Please upload `actual_orders.csv` and `actual_shipments.csv`.")
            
            # --- Simulation and Analysis Logic Here ---
            st.subheader("Inventory Policy Calculations")
            # This is where the selected safety stock method would be applied.
            # Example: Calculating safety stock for a specific SKU/location
            if safety_stock_method == "King's Method":
                # Assuming lead time of 7 days for example purposes
                ss_example = calculate_safety_stock_kings(df_sales, 7, service_level / 100)
                st.write(f"Example Safety Stock (King's Method, 7-day lead time): {ss_example} units")
            else:
                # Assuming lead time of 7 days for example purposes
                ss_example = calculate_safety_stock_avg_max(df_sales, 7)
                st.write(f"Example Safety Stock (Avg Max Method, 7-day lead time): {ss_example} units")
            
            # Here would be the full simulation run
            st.warning("Full simulation and detailed results are not yet implemented in this version.")


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
